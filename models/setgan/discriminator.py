import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import pickle

from models.styleganxl.training.diffaug import DiffAugment
from models.styleganxl.training.networks_stylegan2 import FullyConnectedLayer
from models.styleganxl.pg_modules.blocks import conv2d, DownBlock, DownBlockPatch
from models.styleganxl.pg_modules.projector import F_RandomProj
from models.styleganxl.feature_networks.constants import VITS

from torch_utils.misc import copy_params_and_buffers

from models.setgan.set import MultiSetTransformerEncoder, MultiSetTransformer
from setgan.utils import to_images, to_imgset, to_set

class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, out_features=64, head=None, patch=False):
        super().__init__()

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], out_features, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def load_weights(self, source):
        rg = list(self.parameters())[0].requires_grad
        self.requires_grad_(False)
        for layer, src_layer in zip(self.main[:-1], source.main[:-1]):
            copy_params_and_buffers(src_layer, layer)
        self.requires_grad_(rg)

    def forward(self, x, c):
        return self.main(x)


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        latent_size=512,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        patch=False,
        set_kwargs={},
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4, 5]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDisc

        set_kwargs.update({
            'x_size': latent_size*4,
            'y_size': latent_size*4,
            'latent_size': latent_size,
            'hidden_size': latent_size,
            'output_size': 1,
            'decoder_layers': 0,
            'weight_sharing': 'none',
            'ln': True,
            'dropout': 0
        })

        mini_discs = []
        set_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            disc_i = Disc(nc=cin, start_sz=start_sz, end_sz=8, out_features=latent_size//4, patch=patch)
            set_i = MultiSetTransformer(**set_kwargs)
            mini_discs += [str(i), disc_i],
            set_discs += [str(i), set_i],

        self.mini_discs = nn.ModuleDict(mini_discs)
        self.set_discs = nn.ModuleDict(set_discs)

    def load_weights(self, source):
        for k in self.mini_discs.keys():
            disc, src_disc = self.mini_discs[k], source.mini_discs[k]
            disc.load_weights(src_disc)

    def forward(self, r_features, x_features, rec=False):
        all_logits = []
        for k in self.mini_discs.keys():
            disc, set = self.mini_discs[k], self.set_discs[k]
            x_flat = to_images(x_features[k])
            r_flat = to_images(r_features[k])
            x_enc = disc(x_flat, None)
            r_enc = disc(r_flat, None)
            print(x_enc.size())
            x_enc = to_set(x_enc.view(x_flat.size(0), -1), initial_set=x_features[k])
            r_enc = to_set(r_enc.view(r_flat.size(0), -1), initial_set=r_features[k])
            logits = torch.ones(r_features[k].size(0),1)#set(r_enc, x_enc)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits

class ProjectedSetDiscriminator(torch.nn.Module):
    def __init__(
        self,
        backbones,
        diffaug=True,
        interp224=True,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.backbones = backbones
        self.diffaug = diffaug
        self.interp224 = interp224

        # get backbones and multi-scale discs
        feature_networks, discriminators = [], []

        for i, bb_name in enumerate(backbones):

            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )

            feature_networks.append([bb_name, feat])
            discriminators.append([bb_name, disc])

        self.feature_networks = nn.ModuleDict(feature_networks)
        self.discriminators = nn.ModuleDict(discriminators)

    def load_weights(self, source):
        rg = list(self.parameters())[0].requires_grad
        self.requires_grad_(False)
        for k in self.feature_networks.keys():
            feat, disc = self.feature_networks[k], self.discriminators[k]
            src_feat, src_disc = source.feature_networks[k], source.discriminators[k]
            copy_params_and_buffers(src_feat, feat)
            disc.load_weights(src_disc)
            
        self.requires_grad_(rg)

    def train(self, mode=True):
        self.feature_networks = self.feature_networks.train(False)
        self.discriminators = self.discriminators.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, r, x):
        logits = []
        xflat = to_images(x)
        rflat = to_images(r)
        for bb_name, feat in self.feature_networks.items():

            # apply augmentation (x in [-1, 1])
            x_aug = DiffAugment(xflat, policy='color,translation,cutout') if self.diffaug else xflat
            r_aug = rflat   #come back to this

            # transform to [0,1]
            x_aug = x_aug.add(1).div(2)
            r_aug = r_aug.add(1).div(2)

            # apply F-specific normalization
            x_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(x_aug)
            r_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(r_aug)

            # upsample if smaller, downsample if larger + VIT
            if self.interp224 or bb_name in VITS:
                x_n = F.interpolate(x_n, 224, mode='bilinear', align_corners=False)
                r_n = F.interpolate(r_n, 224, mode='bilinear', align_corners=False)

            # forward pass
            x_features = feat(x_n)
            r_features = feat(r_n)
            for k in x_features.keys():
                x_features[k] = to_imgset(x_features[k], initial_set=x)
                r_features[k] = to_imgset(r_features[k], initial_set=r)
            logits += self.discriminators[bb_name](r_features, x_features)

        return logits
