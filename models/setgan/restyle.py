import numpy as np
import torch
import torch.nn as nn

from setgan.utils import to_images

class Restyle(nn.Module):
    def __init__(self, encoder, decoder, latent_avg, avg_image, n_styles, iters=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_styles = n_styles
        self.latent_avg = latent_avg
        self.iters = iters

        self.latent_avg = self.latent_avg
        self.avg_image = avg_image
    
    def forward(self, x, iters=None, return_latents=False, return_intermediates=False):
        iters = iters if iters is not None else self.iters
        Y = [self.avg_image.unsqueeze(0).expand_as(x).to(x.device)]
        Z = [self.latent_avg.unsqueeze(0).to(x.device)]
        #z =  # latent codes
        for i in range(1, iters+1):
            z = Z[-1] + self.encoder(torch.cat([x, Y[-1]], dim=1)) # add output to previous z?
            y = self.decoder.synthesis(z)
            Y.append(y)
            Z.append(z)
        
        if return_intermediates:
            #if return_latents:
                #return Y, Z
            #else:
                #return Y
            return Z
        else:
            #if return_latents:
                #return Y[-1], Z[-1]
            #else:
                #return Y[-1]
            return Z[-1]


class ReSetGAN(nn.Module):
    def __init__(self, encoder, decoder, set_model, latent_avg):
        self.encoder = encoder
        self.decoder = decoder
        self.set_model = set_model
        self.latent_avg = latent_avg

    def forward(self, x, s, y0, iters=3):
        Y = [y0]
        z0 = self.latent_avg
        z_X = z0 + self.encoder(to_images(x))
        Z_Y = [z0]
        
        for t in range(1, iters+1):
            z_Y = Z_Y[-1] + self.encoder(to_images(Y[-1]))
            W = self.set_model(s, z_Y, z_X) # transformed latent vectors
            y = self.decoder(W)
            Y.append(y)
            Z_Y.append(z_Y)

        return Y, Z_Y

class ReSetGAN2(nn.Module):
    def __init__(self, model):
        self.model = model
    
    def forward(self, x_in, s, x_out0, y0, iters=3):
        Y = [y0]
        X_out = [x_out0]
        z0 = self.model.latent_avg
        Z_X = [z0]
        Z_Y = [z0]
        
        for t in range(1, iters+1):
            z_X = Z_X[-1] + self.model.encode(to_images(x_in), to_images(X_out[-1]))
            z_Y = Z_Y[-1] + self.model.encode(to_images(Y[-1]))
            W = self.model.set_model(s, z_Y, z_X) # transformed latent vectors
            y = self.model.decode(W)
            x_out = self.model.decode(z_X)
            Y.append(y)
            X_out.append(x_out)
            Z_Y.append(z_Y)
            Z_X.append(z_X)

        return Y, X_out, Z_Y, Z_X