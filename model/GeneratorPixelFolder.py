__all__ = ['PixelFolder'
           ]

import math
import random

import torch
from torch import nn

from .blocks import ConstantInput, StyledConv, ToRGB, PixelNorm, EqualLinear, Unfold, LFF, PosFold
import tensor_transforms as tt


class PixelFolder(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        activation=None,
        **kwargs,
    ):
        super().__init__()

        self.size = size
        
        # ---------------- mappling block -----------------
        self.style_dim = style_dim
        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # ------------------ Synthesis block -----------------
        self.log_size = int(math.log(size, 2))

        self.num_fold_per_stage = 2
        self.num_stage = self.log_size // self.num_fold_per_stage - 1

        unfolded_res = 2**(self.log_size - (self.num_stage-1)*self.num_fold_per_stage)

        folded_res = unfolded_res // (2**self.num_fold_per_stage)
        folded_shape = (self.channels[folded_res], folded_res, folded_res)

        self.input = ConstantInput(self.channels[unfolded_res], size=unfolded_res)

        self.posfolders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.lffs = nn.ModuleList()

        for i in range(self.num_stage):
            unfolded_res = 2**(self.log_size - (self.num_stage-i-1)*self.num_fold_per_stage)
            unfolded_shape = (self.channels[unfolded_res], unfolded_res, unfolded_res)
            self.lffs.append(LFF(self.channels[unfolded_res]))
            self.posfolders.append(
                PosFold(folded_shape, unfolded_shape, use_const=True if i==0 else False)
            )
            in_channel = self.channels[folded_res]
            for fold in range(self.num_fold_per_stage):
                out_channel = self.channels[folded_res*(2**(fold+1))]
                self.convs.append(
                    StyledConv(
                    in_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.convs.append(
                    StyledConv(
                    out_channel//4, out_channel, 3, style_dim, blur_kernel=blur_kernel # due to the unfolding operation, the channel is reduced by 1/4. 
                    )
                )
                self.to_rgbs.append(ToRGB(out_channel, style_dim))
                in_channel = out_channel
            folded_res = unfolded_res
            folded_shape = unfolded_shape

        self.unfolder = Unfold()
        self.n_latent = (self.num_fold_per_stage * self.num_stage) * 2 + 1

    def forward(
        self,
        coords,  # fake argument
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        return_all_images=False
    ):
    # -------------- mapping blocks -----------
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        elif len(styles) > 2:
            latent = torch.stack(styles, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
    # -------------- synthesis blocks -----------
        images = []
        out = self.input(latent)
        skip = None
        
        for i in range(self.num_stage):

            b, _, h, w = out.shape
            if i > 0:
                h, w = h*2**self.num_fold_per_stage, w*2**self.num_fold_per_stage
            coord = tt.convert_to_coord_format(b, h, w, device=out.device)
            emb = self.lffs[i](coord)
            
            out = self.posfolders[i](emb, out, is_first=(i == 0))
            for fold in range(self.num_fold_per_stage):
                out = self.convs[i*self.num_fold_per_stage*2 + fold*2](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2]
                    )
                out = self.unfolder(out)
                out = self.convs[i*self.num_fold_per_stage*2 + fold*2 + 1](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 1]
                )
                skip = self.to_rgbs[i*self.num_fold_per_stage + fold](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 2], skip
                    )
            images.append(skip)

        image = skip

        if return_latents:
            return image, latent
        elif return_all_images:
            return image, images
        else:
            return image, None

