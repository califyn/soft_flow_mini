import numpy as np
import scipy as sp
import scipy.sparse.linalg as sla
import robust_laplacian as rl
import potpourri3d as pp3d
import scipy

from glob import glob

# I think the bottom line 10 is meant to be commented out
# Laplacian Eigenspectrum
# Computes first num_eigs eigenfunctions + eigenvalues
# L = evecs @ diag(evals) @ transpose(evecs) @ diag(mass)
# Forward fourier transform: matmul by transpose(evecs) @ diag(mass)
# Inverse fourier transform: matmul by evecs

def get_edgeaware_grid_eigs( H, W, num_eigs, image):
  C = image.shape[0]
  img_to_flat = np.reshape(np.arange(H*W), (H, W))
  image = image.cpu().numpy()

  row = np.concatenate((img_to_flat[1:, :].flatten(),
                   img_to_flat[:-1, :].flatten(),
                   img_to_flat[:, 1:].flatten(),
                   img_to_flat[:, :-1].flatten()), axis=0)
  col = np.concatenate((img_to_flat[:-1, :].flatten(),
                   img_to_flat[1:, :].flatten(),
                   img_to_flat[:, :-1].flatten(),
                   img_to_flat[:, 1:].flatten()), axis=0)
  val = np.concatenate(((image[:, 1:, :] - image[:, :-1, :]).reshape((C, -1)),
                   (image[:, :-1, :] - image[:, 1:, :]).reshape((C, -1)),
                   (image[:, :, 1:] - image[:, :, :-1]).reshape((C, -1)),
                   (image[:, :, :-1] - image[:, :, 1:]).reshape((C, -1))), axis=1)
  val = np.abs(val)
  val = np.sum(val, axis=0) # sum across channels
  val = np.exp(-1 * val/100) # sigma=100.... seems like pretty mild

  sort_idx = np.argsort(row)
  row, col, val = row[sort_idx], col[sort_idx], val[sort_idx]

  adj_mtx = scipy.sparse.csr_matrix((val, (row, col)), shape=(H*W, H*W))
  L = scipy.sparse.csgraph.laplacian(adj_mtx)
  M = None # I think this is ok?

  #L = L + sp.sparse.identity(L.shape[0])*1.0e-8

  # Compute first K eigenvectors
  evals, evecs = sla.eigsh(L, num_eigs, M, sigma=1e-8)

  #M = M.tocoo()
  #mass        = np.zeros(H*W, dtype=np.float32)
  #mass[M.row] = M.data
  mass        = np.ones(H*W, dtype=np.float32)

  return evecs, evals, mass, L.todense(order="C")


def get_grid_eigs( H, W, num_eigs ):

  # Build two triangulations of plane (split quads between TL & BR or TR & BL)
  I, J = np.meshgrid( np.arange(H), np.arange(W), indexing="ij" )
  maxD = max( H-1, W-1 )
  
  I = I * (maxD / (H-1))
  J = J * (maxD / (W-1))
  
  V   = np.reshape( np.concatenate((I[..., None], J[..., None]), axis=-1), (-1, 2) )
  ind = np.reshape( np.arange(V.shape[0]), (H, W) )

  T1 = []
  T2 = []

  for i in range( H-1 ):

    for j in range(W-1):
      t11 = [ind[i, j], ind[i+1, j+1], ind[i, j+1]]
      t12 = [ind[i, j], ind[i+1, j], ind[i+1, j+1]]

      t21 = [ind[i, j], ind[i+1, j], ind[i, j+1]]
      t22 = [ind[i+1, j+1], ind[i, j+1], ind[i+1, j]]

      T1.append( t11 )
      T1.append( t12 )

      T2.append( t21 )
      T2.append( t22 )

  T1 = np.asarray( T1, dtype=np.int32 )
  T2 = np.asarray( T2, dtype=np.int32 )

  V = np.concatenate( (V, np.zeros_like(V[:, 0, None])), axis=-1 )

  # Compute and average Laplacians
  L1, M1 = rl.mesh_laplacian( V, T1 )
  L2, M2 = rl.mesh_laplacian( V, T2 )
 
  #L1        = pp3d.cotan_laplacian(V, T1, denom_eps=1.0e-10)
  #massvec1  = pp3d.vertex_areas(V, T1)
  #massvec1 += 1.0e-8 * np.mean(massvec1)
  #M1        = sp.sparse.diags(massvec1)
  
  #L2        = pp3d.cotan_laplacian(V, T2, denom_eps=1.0e-10)
  #massvec2  = pp3d.vertex_areas(V, T2)
  #massvec2 += 1.0e-8 * np.mean(massvec2)
  #M2        = sp.sparse.diags(massvec2)
     
  L = 0.5 * (L1 + L2) # Unnecessary, can just take one or the other
  M = 0.5 * (M1 + M2) # Potentially necessary

  L = L + sp.sparse.identity(L.shape[0])*1.0e-8

  # Compute first K eigenvectors
  evals, evecs = sla.eigsh(L, num_eigs, M, sigma=1e-8)

  M = M.tocoo()
  mass        = np.zeros(H*W, dtype=np.float32)
  mass[M.row] = M.data

  return evecs, evals, mass, L.todense(order="C")

def get_grid_eigs_cached(H, W, num_eigs, image=None):
    # Preferably, load equal
    for file in glob("eigens/*.npz"):
        z = np.load(file)
        if z["h"] == H and z["w"] == W and num_eigs == z["num_eigs"]:
            if image is not None:
                if "image" not in z:
                    continue
                elif not np.array_equal(image.cpu().numpy(), z["image"]):
                    continue # images don't match
            elif "image" in z:
                continue
            print(f'eigens found! equal {file}')
            ret = z["evecs"], z["evals"], z["mass"], None #z["L"]
            return ret

    """
    for file in glob("eigens/*.npz"):
        z = np.load(file)
        if z["h"] == H and z["w"] == W and num_eigs <= z["num_eigs"]:
            print('eigens found!')
            return z["evecs"], z["evals"], z["mass"], z["L"]
    """

    if image is None:
        ret = get_grid_eigs(H, W, num_eigs)
    else:
        ret = get_edgeaware_grid_eigs(H, W, num_eigs, image)
    checksum = float(image.cpu().numpy().sum().item())
    checksum = int(divmod(checksum, 1)[0])
    np.savez(f"eigens/{H}_{W}_{num_eigs}_{checksum}.npz",
        h=H,
        w=W,
        image=image.cpu().numpy(),
        num_eigs=num_eigs,
        evecs=ret[0],
        evals=ret[1],
        mass=ret[2],
        L=ret[3],
    )

    return ret

# 3p
import torch
import torch.nn as nn
# project

class LaplacianDiffusionBlock(nn.Module):
    """
    Applies Laplacian powers/diffusion in the spectral domain like
        f_out = lambda_i ^ k * e ^ (lambda_i t) f_in
    with learned per-channel parameters k and t.

    Inputs:
      - values: (K,C) in the spectral domain
      - evals: (K) eigenvalues
    Outputs:
      - (K,C) transformed values in the spectral domain
    """

    def __init__(self, C_inout, with_power=True, max_time=False):
        super(LaplacianDiffusionBlock, self).__init__()
        self.C_inout = C_inout
        self.with_power = with_power
        self.max_time = max_time

        self.laplacian_power = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)

        if self.with_power:
            nn.init.constant_(self.laplacian_power, 0.0)
        nn.init.constant_(self.diffusion_time, 0.0001)

    def forward(self, x, evals):

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        if self.max_time:
            diffusion_time = self.max_time * torch.sigmoid(self.diffusion_time)
            # diffusion_time = self.diffusion_time.clamp(min=-self.max_time, max=self.max_time)
        else:
            diffusion_time = self.diffusion_time

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * torch.abs(diffusion_time).unsqueeze(0))

        if self.with_power:
            lambda_coefs = torch.pow(evals.unsqueeze(-1), (2.0 * torch.sigmoid(self.laplacian_power) - 1.0).unsqueeze(0))
        else:
            lambda_coefs = torch.ones_like(self.laplacian_power)

        if x.is_complex():
            return NotImplementedError
            #y = ensure_complex(lambda_coefs * diffusion_coefs) * x
        else:
            y = lambda_coefs * diffusion_coefs * x
            # (C,) * (K, C) * (

        return y

