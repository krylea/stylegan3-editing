import torch
from torch import nn
import os

from configs.paths_config import model_paths
from inversion.models.encoders import restyle_e4e_encoders
from models.stylegan3.model import SG3Generator
from utils import common

from models.setgan.set import SetTransformerDecoder
from models.stylegan3.networks_stylegan3 import FullyConnectedLayer
from setgan.utils import to_images, to_imgset, to_set


class StyleAttention(nn.Module):
    def __init__(self, opts):
        self.n_styles = opts.n_styles
        attns = []
        style_concats = []
        use_set_decoder = opts.use_set_decoder if hasattr(opts, 'use_set_decoder') else True
        for i in range(self.n_styles):
            attns_i = SetTransformerDecoder(opts.latent, opts.latent, opts.latent*2, opts.latent, opts.n_heads, opts.attn_layers, ln=True, 
                                      activation_fct=nn.LeakyReLU, self_attn=use_set_decoder, dropout=0, use_temperature=opts.use_temperature)
            style_concats.append(FullyConnectedLayer(opts.latent*2, opts.latent))
            attns.append(attns_i)
        self.attns = nn.ModuleList(attns)

        if not self.opts.disable_style_concat:
            self.style_concats = nn.ModuleList(style_concats)
            for layer in self.style_concats:
                with torch.no_grad():
                    torch.nn.init.normal_(layer.weight[:, :self.decoder.style_dim], std=0.2)
                    torch.nn.init.eye_(layer.weight[:, self.decoder.style_dim:])
    
    def forward(self, z, s):
        transformed_codes = []
        for i in range(self.n_styles):
            codes_i = self.attns[i](s, z[:,:,i])
            if not self.args.disable_style_concat:
                codes_i = self.style_concats[i](torch.cat([codes_i, s], dim=-1))
            else:
                codes_i = codes_i + s
            #codes_i = codes_i.view(-1, codes_i.size(-1))
            transformed_codes.append(codes_i)
        transformed_codes = torch.stack(transformed_codes, dim=2)
        return transformed_codes


class SetGAN(nn.Module):

    def __init__(self, opts):
        super(SetGAN, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.n_styles = opts.n_styles
        self.encoder = self.set_encoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

        self.style_attn = StyleAttention(opts)

        for parameter in self.decoder.mapping.parameters():
            parameter.requires_grad_(False)


    def set_encoder(self):
        if self.opts.encoder_type == 'ProgressiveBackboneEncoder':
            encoder = restyle_e4e_encoders.ProgressiveBackboneEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetProgressiveBackboneEncoder':
            encoder = restyle_e4e_encoders.ResNetProgressiveBackboneEncoder(self.n_styles, self.opts)
        else:
            raise Exception(f'{self.opts.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None and os.path.exists(self.opts.checkpoint_path):
            print(f'Loading ReStyle e4e from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self._get_keys(ckpt, 'encoder'), strict=True)
            self.decoder = SG3Generator(checkpoint_path=self.opts.stylegan_weights).decoder
            self.decoder.load_state_dict(self._get_keys(ckpt, 'decoder', remove=["synthesis.input.transform"]), strict=False)
            self._load_latent_avg(ckpt)
        else:
            encoder_ckpt = self._get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            self.decoder = SG3Generator(checkpoint_path=self.opts.stylegan_weights).decoder.cuda()
            self.latent_avg = self.decoder.mapping.w_avg

    def decode(self, x, transform=None, resize=True, **kwargs):
        if transform is not None:
            self.decoder.synthesis.input.transform = transform
        images = self.decoder.synthesis(x, noise_mode='const', force_fp32=True, **kwargs)
        if resize:
            images = self.face_pool(images)
        return images
        
    def forward(self, x, s, latent=None, resize=True, input_code=False, landmarks_transform=None,
                return_latents=False, return_aligned_and_unaligned=False):

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
        bs, rs = images.size()[:2]
        cs = s.size(1)

        codes = self.encoder(to_images(x))
        #codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        codes = codes.view(bs, rs, *codes.size()[1:])
        style_latents = self.decoder.mapping(s.view(-1, s.size(-1)))
        style_latents = style_latents.view(*s.size()[:-1], style_latents.size(-1))

        transformed_codes = self.style_attn(codes, style_latents)
        transformed_codes = transformed_codes + self.latent_avg.repeat(*transformed_codes.size()[:2], 1, 1)
        decoder_inputs = transformed_codes.view(-1, *transformed_codes.size()[2:])

        # generate the aligned images
        '''
        identity_transform = common.get_identity_transform()
        if not self.opts.sgxl:
            identity_transform = torch.from_numpy(identity_transform).unsqueeze(0).repeat(x.shape[0], 1, 1).cuda().float()
        else:
            identity_transform = torch.from_numpy(identity_transform).cuda().float()
        self.decoder.synthesis.input.transform = identity_transform
        '''
        images = self.decode(decoder_inputs, resize=resize)
        images = images.view(bs, cs, *images.size()[2:])

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

    def set_opts(self, opts):
        self.opts = opts

    def _load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to("cuda")
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

    @staticmethod
    def _get_keys(d, name, remove=[]):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items()
                  if k[:len(name)] == name and k[len(name) + 1:] not in remove}
        return d_filt
