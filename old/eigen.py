class EigenResult():
    def __init__(self, result):
        if result is not None:
            self.evecs = torch.Tensor(result[0])
            self.evals = result[1]
            self.mass = torch.Tensor(result[2])
            self.L = result[3]

            #self.to_spectral_mtx = self.evecs.T
            #self.to_original_mtx = torch.diag(self.mass) @ self.evecs
            self.to_spectral_mtx = self.evecs.T @ torch.diag(self.mass)
            self.to_original_mtx = self.evecs

            x = self.to_original_mtx @ self.to_spectral_mtx
            print(x)
            print(torch.mean(torch.diag(x)))

    def to_spectral(self, x):
        if isinstance(x, tuple):
            return tuple([self.to_spectral(xx) for xx in x])
        else:
            self.orig_shape = x.shape
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]*x.shape[3]))
            #ret = torch.sum(self.to_spectral_mtx[None, :, :].to(x.device) * x[:, None], dim=2)
            ret = torch.matmul(self.to_spectral_mtx[None, :, :].to(x.device), x[:, :, None])[..., 0]

            ret = ret.reshape((self.orig_shape[0], self.orig_shape[1], -1))
            ret = ret.permute((0, 2, 1))

            return ret
            #return x
            #ret = torch.sum(self.to_spectral_mtx[None, :, :, None].to(x.device) * x[:, None], dim=2)
            #print(ret)
            #input([torch.sort(torch.abs(ret[..., i]), descending=True) for i in range(ret.shape[-1])])
            #return ret
            # (1, K, N, 1) * (B, 1, N, C)
            # (B, K, N, C)
            #x = x.reshape((x.shape[0], 176, 240, -1))
            #c = x.shape[-1]
            #x = torch.permute(x, (0, 3, 1, 2))
            """
            t = x - x.min()
            t = t / x.max()
            t = t[:, :3]
            torchvision.utils.save_image(t, "pre.png")
            """
            self.xy = (x.shape[2], x.shape[3])
            x = torch.fft.rfft2(x)
            x_pad = torch.zeros_like(x)
            #x_pad[:, :, :x.shape[2]//1, :x.shape[3]//1] += x[:, :, :x.shape[2]//1, :x.shape[3]//1]
            x_pad[:, :, :x.shape[2]//2, :x.shape[3]//2] += x[:, :, :x.shape[2]//2, :x.shape[3]//2]
            #x = torch.permute(x_pad, (0, 2, 3, 1))
            #x = x.reshape((x.shape[0], -1, c))
            return x_pad
    
    def to_original(self, x):
        if isinstance(x, tuple):
            return tuple([self.to_original(xx) for xx in x])
        else:
            x = x.reshape((self.orig_shape[0], -1, self.orig_shape[1]))
            x = x.permute((0, 2, 1))
            x = x.reshape((self.orig_shape[0]*self.orig_shape[1], -1))

            #ret = torch.sum(self.to_original_mtx[None, :, :].to(x.device) * x[:, None], dim=2)
            ret = torch.matmul(self.to_original_mtx[None, :, :].to(x.device), x[:, :, None])[..., 0]
            ret = ret.reshape(self.orig_shape)

            return ret
            #return x
            #return torch.sum(self.to_original_mtx[None, :, :, None].to(x.device) * x[:, None], dim=2)
            # (1, N, K, 1) * (B, 1, K, C)
            # (B, N, C)
            #x = x.reshape((-1, self.xy[0], self.xy[1], x.shape[-1]))
            #x = torch.permute(x, (0, 3, 1, 2))
            x = torch.fft.irfft2(x, s=self.xy)
            """
            t = x - x.min()
            t = t / x.max()
            t = t[:, :3]
            torchvision.utils.save_image(t, "post.png")
            """
            #x = torch.permute(x, (0, 2, 3, 1))
            #x = x.reshape((x.shape[0], -1, x.shape[-1]))
            return x

class LaplacianBlock(nn.Module):
    def __init__(self, eigs, dims, patchify=None, unpatchify=None):
        super(LaplacianBlock, self).__init__()
        self.eigs = eigs
        self.diffusion = eigenutils.LaplacianDiffusionBlock(dims)

        #self.model_patchify = patchify
        #self.model_unpatchify = unpatchify

    """
    def unpatchify(self, x):
        if isinstance(x, tuple):
            ret = [self.unpatchify(xx) for xx in x]
            return tuple([r[0] for r in ret]), tuple([r[1] for r in ret])
        else:
            x = self.model_unpatchify(x)
            x = torch.permute(x, (0, 2, 3, 1))
            h, w = x.shape[1], x.shape[2]
            x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))
            return x, (h, w)

    def patchify(self, x, hw):
        if isinstance(x, tuple):
            return tuple([self.patchify(xx, hwhw) for xx, hwhw in zip(x, hw)])
        else:
            x = torch.reshape(x, (x.shape[0], hw[0], hw[1], x.shape[-1]))
            x = torch.permute(x, (0, 3, 1, 2))
            x = self.model_patchify(x)
            return x
    """

    def forward(self, x):
        return x, 0. 
        #x = self.block(*args)

        #x_, hw = self.unpatchify(x)
        k = self.eigs.to_spectral(x)
        x_ = self.eigs.to_original(k)
        #return x_

        diffused = self.diffusion(k, torch.Tensor(self.eigs.evals).to(k.device))
        diffused = self.eigs.to_original(diffused)

        # maintain high frequencies but penalize them
        return diffused + (x - x_), torch.nn.functional.mse_loss(x - x_, torch.zeros_like(x))

class IdentityBlock(nn.Module):
    def __init__(self, d):
        super(IdentityBlock, self).__init__()

    def forward(self, x, eigs):
        return x, 0.
