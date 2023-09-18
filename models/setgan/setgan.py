import torch
from torch import nn
import os

from configs.paths_config import model_paths
from models.setgan.encoder.encoders import restyle_e4e_encoders
from models.stylegan3.model import SG3Generator
from utils import common

import models.styleganxl.dnnlib as dnnlib
import models.styleganxl.legacy as legacy
from models.styleganxl.torch_utils import misc

from models.setgan.set import SetTransformerDecoder
from models.stylegan3.networks_stylegan3 import FullyConnectedLayer
from models.styleganxl.training.networks_stylegan3_resetting import SuperresGenerator, Generator
from setgan.utils import to_images, to_imgset, to_set
from models.setgan.restyle import Restyle

import pickle

class StyleAttention(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.n_styles = opts.n_styles
        self.opts = opts
        attns = []
        style_concats = []
        use_set_decoder = opts.use_set_decoder if hasattr(opts, 'use_set_decoder') else True
        for i in range(self.n_styles):
            attns_i = SetTransformerDecoder(opts.latent, opts.latent, opts.latent*2, opts.latent, opts.n_heads, opts.attn_layers, ln=True, 
                                      activation_fct=nn.LeakyReLU, self_attn=use_set_decoder, dropout=0, use_temperature=opts.use_temperature)
            style_concats.append(FullyConnectedLayer(opts.latent*2, opts.latent))
            attns.append(attns_i)
        self.attns = nn.ModuleList(attns)

        if not opts.disable_style_concat:
            self.style_concats = nn.ModuleList(style_concats)
            for layer in self.style_concats:
                with torch.no_grad():
                    torch.nn.init.normal_(layer.weight[:, :self.opts.style_dim], std=0.2)
                    torch.nn.init.eye_(layer.weight[:, self.opts.style_dim:])
    
    def forward(self, z, s):
        transformed_codes = []
        for i in range(self.n_styles):
            codes_i = self.attns[i](s[:,:,i], z[:,:,i])
            if not self.opts.disable_style_concat:
                codes_i = torch.cat([codes_i, s[:,:,i]], dim=-1)
                codes_i = self.style_concats[i](codes_i.view(-1, codes_i.size(-1))).view(*codes_i.size()[:-1], -1)
            else:
                codes_i = codes_i + s[:,:,i]
            #codes_i = codes_i.view(-1, codes_i.size(-1))
            transformed_codes.append(codes_i)
        transformed_codes = torch.stack(transformed_codes, dim=2)
        return transformed_codes


class SetGAN(nn.Module):

    def __init__(self, opts):
        super(SetGAN, self).__init__()
        self.opts=opts
        # Define architecture
        #self.n_styles = opts.n_styles
        #self.encoder = self.set_encoder()
        #self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        if self.opts.decoder_ckpt is not None:
            print(f"Loading StyleGAN3 generator from path: {self.opts.decoder_ckpt}")
            with open(self.opts.decoder_ckpt, "rb") as f:
                self.decoder = pickle.load(f)['G_ema'].cpu()
            print('Done!')
        else:
            self.decoder = dnnlib.util.construct_class_by_name(**opts.decoder_kwargs)
        self.n_styles = self.decoder.num_ws
        opts.n_styles = self.n_styles
        opts.encoder_kwargs.n_styles = self.n_styles
        self.encoder = dnnlib.util.construct_class_by_name(**opts.encoder_kwargs)

        self.opts.style_dim = self.decoder.w_dim
        self.style_attn = StyleAttention(opts)

        # Load weights if needed

        if opts.path_stem is not None:
            with dnnlib.util.open_url(self.path_stem) as f:
                G_stem = legacy.load_network_pkl(f)['G_ema']
            misc.copy_params_and_buffers(G_stem.encoder, self.encoder, require_all=False)
            misc.copy_params_and_buffers(G_stem.style_attn, self.style_attn, require_all=False)
        else:
            self.load_weights()

        self.mean_center = opts.mean_center

        self.freeze_params()

        self.update_latent_avg()

        if self.opts.restyle_mode == 'encoder':
            self.encoder = Restyle(self.encoder, self.decoder, self.latent_avg, self.avg_image, self.opts.n_styles, iters=self.opts.restyle_iters)

    def freeze_params(self):
        if self.opts.use_pretrained:
            for parameter in self.decoder.mapping.parameters():
                parameter.requires_grad_(False)

            if self.opts.freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)

            if self.opts.freeze_decoder:
                for parameter in self.decoder.parameters():
                    parameter.requires_grad_(False)

    def update_latent_avg(self):
        self.latent_avg = self.decoder.mapping.w_avg.cpu()
        with torch.no_grad():
            self.avg_image = self.decoder.synthesis(self.latent_avg.repeat(self.n_styles, 1).unsqueeze(0))[0]
        self.avg_image = self.avg_image.float().detach()

        if self.opts.restyle_mode == 'encoder':
            self.encoder.latent_avg = self.latent_avg
            self.encoder.avg_image = self.avg_image

    def load_weights(self):
        if self.opts.encoder_ckpt is not None:
            encoder_ckpt = torch.load(self.opts.encoder_ckpt, map_location='cpu')
            self.encoder.load_state_dict(self._get_keys(encoder_ckpt, 'encoder'), strict=True)
        else:
            if self.opts.encoder_type == 'ProgressiveBackboneEncoder':
                print('Loading encoders weights from irse50!')
                encoder_ckpt = torch.load(model_paths['ir_se50'])
                # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
                if self.opts.encoder_kwargs.input_nc != 3:
                    shape = encoder_ckpt['input_layer.0.weight'].shape
                    altered_input_layer = torch.randn(shape[0], self.opts.encoder_kwargs.input_nc, shape[2], shape[3], dtype=torch.float32)
                    altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                    encoder_ckpt['input_layer.0.weight'] = altered_input_layer
        

                               
        '''
        if self.opts.checkpoint_path is not None and os.path.exists(self.opts.checkpoint_path):
            print(f'Loading ReStyle e4e from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self._get_keys(ckpt, 'encoder'), strict=True)
            self.decoder = SG3Generator(checkpoint_path=self.opts.stylegan_weights).decoder.cpu()
            self.decoder.load_state_dict(self._get_keys(ckpt, 'decoder', remove=["synthesis.input.transform"]), strict=False)
            #self._load_latent_avg(ckpt)
        else:
            encoder_ckpt = self._get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            self.decoder = SG3Generator(checkpoint_path=self.opts.stylegan_weights).decoder.cpu()
            #self.latent_avg = self.decoder.mapping.w_avg.cpu()
        '''

    def decode(self, x, transform=None, resize=True, update_emas=False, **kwargs):
        #if transform is not None:
        #    self.decoder.synthesis.input.transform = transform
        images = self.decoder.synthesis(x, update_emas=update_emas, **kwargs)
        #if resize:
        #    images = self.face_pool(images)
        return images

        
    def forward(self, x, s, latent=None, resize=True, input_code=False, landmarks_transform=None,
                return_latents=False, return_aligned_and_unaligned=False, update_emas=False, **kwargs):

        images, unaligned_images = None, None

        '''
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        '''
        bs, rs = x.size()[:2]
        cs = s.size(1)

        
        codes = self.encoder(to_images(x))
        if self.mean_center:
            codes = codes - self.latent_avg.repeat(codes.size(0), 1, 1).to(codes.device)
        codes = codes.view(bs, rs, *codes.size()[1:])

        if input_code:
            style_latents = s
        else:
            style_latents = self.decoder.mapping(s.view(-1, s.size(-1)), None, update_emas=update_emas) 
            if self.mean_center:
                style_latents = style_latents - self.latent_avg.repeat(style_latents.size(0), 1, 1).to(style_latents.device)
            style_latents = style_latents.view(*s.size()[:-1], *style_latents.size()[-2:])

        transformed_codes = self.style_attn(codes, style_latents)
        if self.mean_center:
            transformed_codes = transformed_codes + self.latent_avg.repeat(*transformed_codes.size()[:2], 1, 1).to(transformed_codes.device)
        decoder_inputs = transformed_codes.view(-1, *transformed_codes.size()[2:])
        
        #decoder_inputs = self.decoder.mapping(s.view(-1, s.size(-1)), None, update_emas=update_emas)

        # generate the aligned images
        '''
        identity_transform = common.get_identity_transform()
        if not self.opts.sgxl:
            identity_transform = torch.from_numpy(identity_transform).unsqueeze(0).repeat(x.shape[0], 1, 1).cuda().float()
        else:
            identity_transform = torch.from_numpy(identity_transform).cuda().float()
        self.decoder.synthesis.input.transform = identity_transform
        '''
        images = self.decode(decoder_inputs, resize=resize, update_emas=False, **kwargs)
        images = images.view(bs, cs, *images.size()[1:])

        # generate the unaligned image using the user-specified transforms
        '''
        if landmarks_transform is not None:
            images = self.decode(decoder_inputs, transform=landmarks_transform.float(), resize=resize)
            unaligned_images = unaligned_images.view(bs, cs, *unaligned_images.size()[2:])

        if landmarks_transform is not None and return_aligned_and_unaligned:
            return images, unaligned_images, transformed_codes
        '''

        if return_latents:
            return images, transformed_codes
        else:
            return images
        
    '''
    def set_opts(self, opts):
        self.opts = opts

    def _load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].cpu()
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def _get_encoder_checkpoint(self):
        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load(model_paths['ir_se50'])
        # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
        if self.opts.input_nc != 3:
            shape = encoder_ckpt['input_layer.0.weight'].shape
            altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
            encoder_ckpt['input_layer.0.weight'] = altered_input_layer
        return encoder_ckpt
    '''

    @staticmethod
    def _get_keys(d, name, remove=[]):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items()
                  if k[:len(name)] == name and k[len(name) + 1:] not in remove}
        return d_filt
