
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from setgan.utils import masked_softmax


class MHA(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, bias=None, equi=False, use_temperature=False):
        super(MHA, self).__init__()
        if bias is None:
            bias = not equi
        self.latent_size = dim_V
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_Q, dim_V, bias=bias)
        self.w_k = nn.Linear(dim_K, dim_V, bias=bias)
        self.w_v = nn.Linear(dim_K, dim_V, bias=bias)
        self.w_o = nn.Linear(dim_V, dim_V, bias=bias)
        self.equi = equi
        
        self.temperature = nn.Parameter(torch.tensor(1.0)) if use_temperature else 1.0

    def forward(self, Q, K, mask=None, return_weights=False):
        Q_ = self.w_q(Q)
        K_, V_ = self.w_k(K), self.w_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.stack(Q_.split(dim_split, 2), 0)
        K_ = torch.stack(K_.split(dim_split, 2), 0)
        V_ = torch.stack(V_.split(dim_split, 2), 0)

        E = Q_.matmul(K_.transpose(2,3))/math.sqrt(self.latent_size) * self.temperature
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_)).split(1, 0), 3).squeeze(0))
        if return_weights:
            return O, A
        else:
            return O

class SetAttentionBlock(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, attn_size=None, ln=True, dropout=0.1, activation_fct=nn.ReLU):
        super(SetAttentionBlock, self).__init__()
        attn_size = attn_size if attn_size is not None else input_size
        self.attn = MHA(input_size, attn_size, latent_size, num_heads)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), activation_fct(), nn.Linear(hidden_size, latent_size))
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)
        else:
            self.ln0 = None
            self.ln1 = None

    def forward(self, Q, K, **kwargs):
        A1 = self.attn(Q, K, **kwargs)
        A1 = A1 if self.dropout is None else self.dropout(A1)
        X = Q + A1
        X = X if self.ln0 is None else self.ln0(X)
        FC = self.fc(X)
        FC = FC if self.dropout is None else self.dropout(FC)
        X = X + FC
        X = X if self.ln1 is None else self.ln1(X)
        return X


class SetEncoderBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, num_heads, ln=False, dropout=0.1, activation_fct=nn.ReLU):
        super().__init__()
        self.attn = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, ln=ln, dropout=dropout, activation_fct=activation_fct)

    def forward(self, X, mask=None):
        return self.attn(X, X, mask=mask)

class SetDecoderBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, encoder_size, num_heads, ln=False, dropout=0.1, activation_fct=nn.ReLU, self_attn=True, use_temperature=False):
        super().__init__()
        self.self_attn=self_attn
        if self_attn:
            self.attn1 = MHA(latent_size, latent_size, latent_size, num_heads, use_temperature=use_temperature)
        self.attn2 = MHA(latent_size, encoder_size, latent_size, num_heads, use_temperature=use_temperature)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), activation_fct(), nn.Linear(hidden_size, latent_size))
        if ln:
            if self_attn:
                self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)
            self.ln2 = nn.LayerNorm(latent_size)
        else:
            self.ln0 = None
            self.ln1 = None
            self.ln2 = None

    def forward(self, Q, K, return_weights=False, **kwargs):
        if self.self_attn:
            A1 = self.attn1(Q, Q, **kwargs)
            A1 = A1 if self.dropout is None else self.dropout(A1)
            X = Q + A1
            X = X if self.ln0 is None else self.ln0(X)
        else:
            X = Q
        A2, W = self.attn2(X, K, return_weights=True, **kwargs)
        A2 = A2 if self.dropout is None else self.dropout(A2)
        X = X + A2
        X = X if self.ln1 is None else self.ln1(X)
        FC = self.fc(X)
        FC = FC if self.dropout is None else self.dropout(FC)
        X = X + FC
        X = X if self.ln2 is None else self.ln2(X)
        if return_weights:
            return X, W
        else:
            return X

class SetTransformerEncoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, num_blocks, ln=False, dropout=0.1, activation_fct=nn.ReLU):
        super().__init__()
        self.proj = nn.Linear(input_size, latent_size) if input_size != latent_size else None
        for i in range(num_blocks):
            setattr(self, "block_%d"%i, SetEncoderBlock(latent_size, hidden_size, num_heads, ln=ln, dropout=dropout, activation_fct=activation_fct))
        self.num_blocks = num_blocks

    def forward(self, inputs, mask=None):
        inputs = inputs if self.proj is None else self.proj(inputs)
        for i in range(self.num_blocks):
            block = getattr(self, "block_%d"%i)
            inputs = block(inputs, mask=mask)
        return inputs

class SetTransformerDecoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, encoder_size, num_heads, num_blocks, ln=False, dropout=0.1, 
                 activation_fct=nn.ReLU, self_attn=True, use_temperature=False):
        super().__init__()
        self.proj = nn.Linear(input_size, latent_size) if input_size != latent_size else None
        self.blocks = nn.ModuleList(
            [
                SetDecoderBlock(latent_size, hidden_size, encoder_size, num_heads, ln=ln, dropout=dropout, 
                                activation_fct=activation_fct, self_attn=self_attn, use_temperature=use_temperature)
                for _ in range(num_blocks)
            ]
        )
        self.num_blocks = num_blocks

    def forward(self, inputs, encoder_outputs, mask=None, return_weights=False):
        inputs = inputs if self.proj is None else self.proj(inputs)
        for i in range(self.num_blocks):
            block = self.blocks[i]
            if return_weights and i == self.num_blocks-1:
                inputs, weights = block(inputs, encoder_outputs, mask=mask, return_weights=True)
            else:
                inputs = block(inputs, encoder_outputs, mask=mask)
        if return_weights:
            return inputs, weights
        else:
            return inputs


class MultiSetAttentionBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, num_heads, weight_sharing='none', ln=False, dropout=0, **kwargs):
        super().__init__()
        self._init_blocks(latent_size, hidden_size, num_heads, weight_sharing, ln=ln, dropout=dropout, **kwargs)
        self.fc_X = nn.Linear(2*latent_size, latent_size)
        self.fc_Y = nn.Linear(2*latent_size, latent_size)
        if ln:
            self.ln_x = nn.LayerNorm(latent_size)
            self.ln_y = nn.LayerNorm(latent_size)
        else:
            self.ln_x = None
            self.ln_y = None
        if dropout > 0:
            self.dropout=nn.Dropout(dropout)
        else:
            self.dropout=None

    def _init_blocks(self, latent_size, hidden_size, num_heads, weight_sharing='none', **kwargs):
        if weight_sharing == 'none':
            self.MAB_XX = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_YY = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_XY = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_YX = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads,  **kwargs)
        elif weight_sharing == 'cross':
            self.MAB_XX = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_YY = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            MAB_cross = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_XY = MAB_cross
            self.MAB_YX = MAB_cross
        elif weight_sharing == 'sym':
            MAB_cross = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            MAB_self = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_XX = MAB_self
            self.MAB_YY = MAB_self
            self.MAB_XY = MAB_cross
            self.MAB_YX = MAB_cross
        else:
            raise NotImplementedError("weight sharing must be none, cross or sym")

    def _get_masks(self, masks):
        if masks is not None:
            mask_xx, mask_xy, mask_yx, mask_yy = masks 
        else: 
            mask_xx, mask_xy, mask_yx, mask_yy = None,None,None,None
        return mask_xx, mask_xy, mask_yx, mask_yy

    def forward(self, X, Y, masks=None):
        mask_xx, mask_xy, mask_yx, mask_yy = self._get_masks(masks)
        XX = self.MAB_XX(X, X, mask=mask_xx)
        XY = self.MAB_XY(X, Y, mask=mask_xy)
        YX = self.MAB_YX(Y, X, mask=mask_yx)
        YY = self.MAB_YY(Y, Y, mask=mask_yy)
        X_merge = self.fc_X(torch.cat([XX, XY], dim=-1))
        Y_merge = self.fc_Y(torch.cat([YY, YX], dim=-1))
        if self.dropout is not None:
            X_merge = self.dropout(X_merge)
            Y_merge = self.dropout(Y_merge)
        X_out = X + X_merge
        Y_out = Y + Y_merge
        X_out = X_out if self.ln_x is None else self.ln_x(X_out)
        Y_out = Y_out if self.ln_y is None else self.ln_y(Y_out)
        return (X_out, Y_out)


class MultiSetTransformerEncoder(nn.Module):
    def __init__(self, x_size, y_size, latent_size, hidden_size, num_heads, num_blocks, weight_sharing='none', ln=False, dropout=0, **kwargs):
        super().__init__()
        if x_size != latent_size and x_size == y_size and weight_sharing != "none":
            proj = nn.Linear(x_size, latent_size)
            self.proj_x, self.proj_y = proj, proj
        else:
            self.proj_x = nn.Linear(x_size, latent_size) if x_size != latent_size else None
            self.proj_y = nn.Linear(y_size, latent_size) if y_size != latent_size else None
        self.blocks = nn.ModuleList(
            [
                MultiSetAttentionBlock(latent_size, hidden_size, num_heads, weight_sharing=weight_sharing, ln=ln, dropout=dropout, **kwargs)
                for _ in range(num_blocks)
            ]
        )
        self.num_blocks = num_blocks

    def forward(self, X, Y, masks=None):
        X = X if self.proj_x is None else self.proj_x(X)
        Y = Y if self.proj_y is None else self.proj_y(Y)
        for i in range(self.num_blocks):
            X, Y = self.blocks[i](X, Y, masks=masks)
        return X, Y


class PMA(nn.Module):
    def __init__(self, latent_size, hidden_size, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, latent_size))
        nn.init.xavier_uniform_(self.S)
        self.mab = SetAttentionBlock(latent_size, latent_size, hidden_size, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)



class NaiveMultiSetEncoder(nn.Module):
    def __init__(self, latent_size, hidden_size, num_blocks, weight_sharing='none', activation_fct=nn.ReLU, **encoder_kwargs):
        super().__init__()
        if weight_sharing == 'none':
            self.encoder1 = nn.Sequential(*[self._init_block(latent_size, hidden_size, activation_fct=activation_fct, **encoder_kwargs) for _ in range(num_blocks)])
            self.encoder2 = nn.Sequential(*[self._init_block(latent_size, hidden_size, activation_fct=activation_fct, **encoder_kwargs) for _ in range(num_blocks)])
        else:
            encoder = nn.Sequential(*[self._init_block(latent_size, hidden_size, **encoder_kwargs) for _ in range(num_blocks)])
            self.encoder1 = encoder
            self.encoder2 = encoder

    def _init_block(self, latent_size, hidden_size, activation_fct, **encoder_kwargs):
        pass

    def forward(self, X, Y):
        ZX = self.encoder1(X)
        ZY = self.encoder2(Y)
        return ZX, ZY

class NaiveSetTransformerEncoder(NaiveMultiSetEncoder):
    def __init__(self, latent_size, hidden_size, num_blocks, weight_sharing='none', activation_fct=nn.ReLU, ln=True, dropout=0):
        return super().__init__(latent_size, hidden_size, num_blocks, weight_sharing=weight_sharing, activation_fct=activation_fct, 
            ln=ln, dropout=dropout)

    def _init_block(self, latent_size, hidden_size, activation_fct, num_heads, ln, dropout):
        return SetEncoderBlock(latent_size, hidden_size, num_heads, ln=ln, dropout=dropout, activation_fct=activation_fct)



class MultiSetModel(nn.Module):
    def __init__(self, x_size, y_size, latent_size, hidden_size, output_size, pooling='mean', num_heads=8, concat_inputs=False, 
            activation_fct=nn.ReLU, decoder_layers=1, **encoder_kwargs):
        super().__init__()
        self.concat_inputs = concat_inputs
        output_latent_size = latent_size if not concat_inputs else latent_size*2

        self.encoder = self._make_encoder(x_size, y_size, latent_size, hidden_size, num_heads=num_heads, activation_fct=activation_fct, **encoder_kwargs)
        self.decoder = self._make_decoder(output_latent_size, hidden_size, output_size, decoder_layers, activation_fct)

        self.pooling = pooling
        if self.pooling == "pma":
            self.pool_x = PMA(output_latent_size, hidden_size, num_heads, 1, ln=True)
            self.pool_y = PMA(output_latent_size, hidden_size, num_heads, 1, ln=True)

        self.proj_x = nn.Linear(x_size, latent_size) if x_size != latent_size else None
        self.proj_y = nn.Linear(y_size, latent_size) if y_size != latent_size else None
        

        self.output_size=output_size
        # Store the latent size since it is needed by some transformation steps.
        self.latent_size=latent_size

    def _make_encoder(self, x_size, y_size, latent_size, hidden_size, num_heads=8, activation_fct=nn.ReLU, **encoder_kwargs):
        pass
    
    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers, activation_fct):
        if n_layers == 0:
            return nn.Linear(3*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), activation_fct()]
            return nn.Sequential(
                nn.Linear(3*latent_size, hidden_size),
                activation_fct(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, X, Y, use_encoder=True):
        if self.proj_x is not None:
            X = self.proj_x(X)
        if self.proj_y is not None:
            Y = self.proj_y(Y)

        if self.encoder is not None and use_encoder:
            ZX, ZY = self.encoder(X, Y) 
        else: 
            ZX, ZY = X,Y

        if self.concat_inputs:
            if self.encoder is not None and use_encoder:
                ZX = torch.cat([ZX, X], dim=-1)
                ZY = torch.cat([ZY, Y], dim=-1)
            else:
                ZX = torch.cat([torch.zeros_like(X), X], dim=-1)
                ZY = torch.cat([torch.zeros_like(Y), Y], dim=-1)

        if self.pooling == "pma":
            ZX = self.pool_x(ZX)
            ZY = self.pool_y(ZY)
        elif self.pooling == "max":
            ZX = torch.max(ZX, dim=1)
            ZY = torch.max(ZY, dim=1)
        elif self.pooling == "mean":
            ZX = torch.mean(ZX, dim=1)
            ZY = torch.mean(ZY, dim=1)
        
        out = torch.cat([ZX, ZY, ZX*ZY], dim=-1)
        out = self.decoder(out)
        if self.output_size == 1:
            out = out.squeeze(-1)
        return out
        

class MultiSetTransformer(MultiSetModel):
    def __init__(self, x_size, y_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=4, ln=True,
            weight_sharing='none', dropout=0.1, decoder_layers=1, pooling='pma', activation_fct=nn.ReLU, concat_inputs=False):
        return super().__init__(x_size, y_size, latent_size, hidden_size, output_size, num_heads=num_heads, num_blocks=num_blocks, 
            ln=ln, weight_sharing=weight_sharing, dropout=dropout, decoder_layers=decoder_layers, pooling=pooling, 
            activation_fct=activation_fct, concat_inputs=concat_inputs)

    def _make_encoder(self, x_size, y_size, latent_size, hidden_size, num_heads=4, num_blocks=4, ln=True, weight_sharing='none', dropout=0.1, activation_fct=nn.ReLU):
        return MultiSetTransformerEncoder(x_size, y_size, latent_size, hidden_size, num_heads, num_blocks, 
            weight_sharing=weight_sharing, ln=ln, dropout=dropout, activation_fct=activation_fct)

class SimpleSetModel(MultiSetModel):
    def __init__(self, x_size, y_size, latent_size, hidden_size, output_size, pooling='mean', num_heads=8, concat_inputs=False, 
            activation_fct=nn.ReLU, decoder_layers=1):
        return super().__init__(x_size, y_size, latent_size, hidden_size, output_size, num_heads=num_heads, 
            decoder_layers=decoder_layers, pooling=pooling, activation_fct=activation_fct, concat_inputs=False)
    
    def _make_encoder(self, x_size, y_size, latent_size, hidden_size, num_heads=8, activation_fct=nn.ReLU, **encoder_kwargs):
        return None

class NaiveSetTransformer(MultiSetModel):
    def __init__(self, x_size, y_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=4, ln=True,
            weight_sharing='none', dropout=0, decoder_layers=1, pooling='pma', activation_fct=nn.ReLU, concat_inputs=False):
        return super().__init__(x_size, y_size, latent_size, hidden_size, output_size, num_heads=num_heads, num_blocks=num_blocks, 
            ln=ln, weight_sharing=weight_sharing, dropout=dropout, decoder_layers=decoder_layers, pooling=pooling, 
            activation_fct=activation_fct, concat_inputs=concat_inputs)
    
    def _make_encoder(self, x_size, y_size, latent_size, hidden_size, num_heads=8, num_blocks=4, ln=True, dropout=0,
            weight_sharing='none',activation_fct=nn.ReLU):
        return NaiveSetTransformerEncoder(latent_size, hidden_size, num_blocks, weight_sharing=weight_sharing, ln=ln, dropout=dropout,
            activation_fct=activation_fct)



class SetVectorModel(nn.Module):
    def __init__(self, x_size, y_size, latent_size, hidden_size, output_size, pooling='mean', num_heads=8, concat_inputs=False, 
            activation_fct=nn.ReLU, decoder_layers=1, **encoder_kwargs):
        super().__init__()
        self.concat_inputs = concat_inputs
        output_latent_size = latent_size if not concat_inputs else latent_size*2

        self.encoder = self._make_encoder(x_size, y_size, latent_size, hidden_size, num_heads=num_heads, activation_fct=activation_fct, **encoder_kwargs)
        self.decoder = self._make_decoder(output_latent_size, hidden_size, output_size, decoder_layers, activation_fct)

        self.pooling = pooling
        if self.pooling == "pma":
            self.pool_x = PMA(output_latent_size, hidden_size, num_heads, 1, ln=True)
            self.pool_y = PMA(output_latent_size, hidden_size, num_heads, 1, ln=True)

        self.proj_x = nn.Linear(x_size, latent_size) if x_size != latent_size else None
        self.proj_y = nn.Linear(y_size, latent_size) if y_size != latent_size else None
        

        self.output_size=output_size
        # Store the latent size since it is needed by some transformation steps.
        self.latent_size=latent_size

    def _make_encoder(self, x_size, y_size, latent_size, hidden_size, num_heads=8, activation_fct=nn.ReLU, **encoder_kwargs):
        pass
    
    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers, activation_fct):
        if n_layers == 0:
            return nn.Linear(3*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), activation_fct()]
            return nn.Sequential(
                nn.Linear(3*latent_size, hidden_size),
                activation_fct(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, X, Y, use_encoder=True):
        if self.proj_x is not None:
            X = self.proj_x(X)
        if self.proj_y is not None:
            Y = self.proj_y(Y)

        ZY = Y
        if self.encoder is not None and use_encoder:
            ZX = self.encoder(X) 
        else: 
            ZX = X

        if self.concat_inputs:
            if self.encoder is not None and use_encoder:
                ZX = torch.cat([ZX, X], dim=-1)
            else:
                ZX = torch.cat([torch.zeros_like(X), X], dim=-1)

        if self.pooling == "pma":
            ZX = self.pool_x(ZX)
        elif self.pooling == "max":
            ZX = torch.max(ZX, dim=1)
        elif self.pooling == "mean":
            ZX = torch.mean(ZX, dim=1)
        
        out = torch.cat([ZX, ZY, ZX*ZY], dim=-1)
        out = self.decoder(out)
        if self.output_size == 1:
            out = out.squeeze(-1)
        return out