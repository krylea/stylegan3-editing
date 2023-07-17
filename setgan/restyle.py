import numpy as np
import torch
import torch.nn as nn

from setgan.utils import to_images

class Restyle(nn.Module):
    def __init__(self, model):
        self.model = model
    
    def forward(self, x, y0, iters=3):
        Y = [y0]
        Z = [self.model.latent_avg]
        #z =  # latent codes
        for i in range(1, iters+1):
            z = Z[-1] + self.model.encode(x, Y[-1]) # add output to previous z?
            y = self.model.decode(z)
            Y.append(y)
            Z.append(z)
        return Y, Z


class ReSetGAN(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, x, s, y0, iters=3):
        Y = [y0]
        z0 = self.model.latent_avg
        z_X = z0 + self.model.encode(to_images(x))
        Z_Y = [z0]
        
        for t in range(1, iters+1):
            z_Y = Z_Y[-1] + self.model.encode(to_images(Y[-1]))
            W = self.model.set_model(s, z_Y, z_X) # transformed latent vectors
            y = self.model.decode(W)
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