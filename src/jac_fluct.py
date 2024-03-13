from . import commons
import scanpy as sc
from functorch import jvp, vjp, jacrev, jacfwd, vmap
import torch
from einops import rearrange, repeat
import math
import numpy as np


@torch.no_grad()
def calc_jac_f_d_with_batch(z, s, model):
    dec_f = lambda vz, vs: model.dec_z(vz, vs)
    jac_f_d = lambda vz, vs: jacrev(lambda vzz: dec_f(vzz, vs))(vz)
    jac_f_d_mat = vmap(jac_f_d, in_dims=(0, 0))(z, s)
    return jac_f_d_mat

@torch.no_grad()
def calc_jac_f_d(z, model):
    dec_f = lambda vz: model.dec_z(vz)
    jac_f_d = lambda vz: jacrev(lambda vzz: dec_f(vzz))(vz)
    jac_f_d_mat = vmap(jac_f_d, in_dims=(0,))(z)
    return jac_f_d_mat

@torch.no_grad()
def calc_jvp_cov(z, s, vec, model):
    dec_f = lambda vz, vs: model.dec_z(vz, vs)
    v_jac_fz = lambda vz, vs: vjp(lambda vzz: dec_f(vzz, vs), vz)[1](vec)[0]
    cov_vec_f = lambda vz, vs: jvp(lambda vzz: dec_f(vzz, vs), (vz,), (v_jac_fz(vz, vs),))[1]
    cov_vec_mat = vmap(cov_vec_f, in_dims=(0, 0))(z, s)
    return cov_vec_mat
    

@torch.no_grad()
def svd_with_mod(df_d, dscales):
    Ud, Sd, Vd = torch.svd(df_d)
    C = Sd.unsqueeze(-1) * Vd.transpose(-1, -2) * (dscales ** 2).unsqueeze(-2)  @ Vd * Sd.unsqueeze(-2)
    Uc, Sc, Vc = torch.svd(C)
    Utot = Ud @ Uc
    return Utot, Sc


def calc_total_sv_cdiff(adata, model, max_vec, batch_key=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    z = torch.tensor(adata.obsm['X_vicdyf_zl']).to(device).float()
    if batch_key != None:
        s = torch.tensor(np.array(adata.obsm[batch_key])).to(device).float()
        jac_f_d_mat = calc_jac_f_d_with_batch(z, s, model)
    else:
        jac_f_d_mat = calc_jac_f_d(z, model)
    max_vec = torch.tensor(max_vec).float().to(jac_f_d_mat.device).reshape(-1, 1)
    d, qd = model.enc_d(z)
    u, s = svd_with_mod(jac_f_d_mat / max_vec, qd.scale)
    adata.obs['top_jac_fluct'] = s[:, 0].detach().cpu().numpy()
    adata.layers['top_jac_fluct_v'] = u[:, :, 0].detach().cpu().numpy()
    adata.obs['total_jac_fluct'] = torch.linalg.norm(s, dim=-1).detach().cpu().numpy()
    import pdb;pdb.set_trace()
    adata.layers['total_jac_fluct_v'] = torch.linalg.norm(u * s.sqrt(), dim=-1).detach().cpu().numpy()
    return adata


def calc_cdiff_svd_aggr(adata, model, max_vec):
    zl = adata.obsm['X_vicdyf_zl']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tot_jac_f_d_mat = torch.zeros(adata.shape[1], zl.shape[1]).to(device)
    tot_scale_mat = torch.zeros(zl.shape[1]).to(device)
    idx_list = commons.make_idx_list(adata.shape[0], 30)
    z = torch.tensor(adata.obsm['X_vicdyf_zl']).to(device).float()
    d, qd = model.enc_d(z)
    dscales = qd.scale.mean(dim=0)
    for idx in idx_list:
        jac_f_d_mat = calc_jac_f_d(z[idx], model)
        tot_jac_f_d_mat += jac_f_d_mat.sum(dim=0)
    max_vec = torch.tensor(max_vec).to(jac_f_d_mat.device).view(-1, 1)
    tot_jac_f_d_mat /= adata.shape[0]
    u, s = svd_with_mod(tot_jac_f_d_mat / max_vec, dscales)
    return commons.safe_numpy(u), commons.safe_numpy(s)


@torch.no_grad()
def calc_jac_mean_scale(z, d_loc, d_scale, model, max_vec, s=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    z = torch.tensor(z).to(device).float()
    d_loc = torch.tensor(d_loc).to(device).float()
    d_scale = torch.tensor(d_scale).to(device).float()
    max_vec = torch.tensor(max_vec).float().to(device).reshape(-1, 1)
    if s is not None:
        s = torch.tensor(s).to(device).float()
        jac_f_d_mat = calc_jac_f_d_with_batch(z, s, model) / max_vec
    else:
        jac_f_d_mat = calc_jac_f_d(z, model) / max_vec
    jac_loc = (jac_f_d_mat @ d_loc.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
    jac_std = torch.norm(jac_f_d_mat * d_scale.unsqueeze(1), dim=-1).detach().cpu().numpy()
    return jac_loc, jac_std



