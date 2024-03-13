import anndata
from statsmodels.stats.multitest import fdrcorrection
import json
import pdb
from scipy import stats
import math
from os import access
from einops import rearrange, repeat
import enum
from matplotlib import pyplot as plt
import smtplib, ssl
from email.mime.text import MIMEText
import copy
from re import sub
from scipy import sparse
from turtle import distance
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import vjp, jvp
# import mygene
from tkinter import W
import sklearn
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import scanpy as sc
import sys
import numpy as np
import umap
import torch
import scipy
from statsmodels.stats.multitest import multipletests
import functorch


def make_arange_minmax(arr, step_num):
    min_val = np.min(arr)
    max_val = np.max(arr)
    step = (max_val - min_val) / (step_num - 1)
    return np.arange(min_val, max_val + 1.0e-5, step)
    

def make_grids(u_mat, div_num):
    xs = make_arange_minmax(u_mat[:, 0], div_num)
    ys = make_arange_minmax(u_mat[:, 1], div_num)
    grids = np.array([[[x, y] for x in xs] for y in ys]).reshape(-1, 2)
    return grids 


def calc_grid_flow(u_mat, d_embed, div_num):
    grids = make_grids(u_mat, div_num)
    nbrs = NearestNeighbors(n_neighbors=30).fit(u_mat)
    dists, indeces = nbrs.kneighbors(grids)
    sigma = np.median(dists[:, 5])
    kernel = np.exp(- (dists/ sigma)**2)
    grid_flow = (d_embed[indeces] * kernel[..., np.newaxis]).mean(axis=1)
    return grids, grid_flow


def embed_z(z_mat, n_neighbors=30, min_dist=0.3):
    if z_mat.shape[1] != 2:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        z_embed = reducer.fit_transform(z_mat)
    else:
        z_embed = np.array(z_mat)
    return(z_embed)
    
    
def embed_z_adata(adata, n_neighbors=30, min_dist=0.3):
    sc.pp.neighbors(adata, use_rep='X_vicdyf_zl', n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)
    z_embed = adata.obsm['X_umap']
    return z_embed

def colwise_pearsonr(r_c, r_ld):
    val = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    return(val)


def colwise_speamanr(r_c, r_ld):
    val = np.array([scipy.stats.spearmanr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    return(val)


@torch.no_grad()
def calc_corr_df(adata):
    s = torch.tensor(adata.layers['spliced'].toarray().astype(float))
    u = torch.tensor(adata.layers['unspliced'].toarray().astype(float))
    px_z_ld = torch.tensor(adata.layers['lambda'])
    pu_zd_ld = torch.tensor(adata.layers['ulambda'])
    norm_s = s.float() / (s.mean(dim=1, keepdim=True))
    norm_u = u.float() / (u.mean(dim=1, keepdim=True))
    s_corr = colwise_pearsonr((norm_s), px_z_ld)
    u_corr = colwise_pearsonr((norm_u), pu_zd_ld)
    r_c = ((norm_u / (norm_s + 1)))
    r_ld = (pu_zd_ld / (px_z_ld + 1))
    r_corr = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    s_mat = s.float()
    u_mat = u.float()
    r_c = (norm_u / (norm_s + 1))
    s_mean = s_mat.mean(axis=0)
    u_mean = u_mat.mean(axis=0)
    s_var = s_mat.std(axis=0) / s_mean
    u_var = u_mat.std(axis=0) / u_mean
    r_var = r_c.std(axis=0)
    stats_df = pd.DataFrame({'s_mean': s_mean, 's_var': s_var, 'u_mean': u_mean, 'u_var': u_var, 'r_var': r_var, 's_corr': s_corr, 'u_corr': u_corr, 'r_corr': r_corr}, index=adata.var_names)
    return(stats_df)


def update_only_dembed(adata, sigma=0.05, embed_mode='vicdyf_', model_d_coeff=0.01, n_neighbors=30, min_dist=0.1, fix_embed=False):
    d_mat = adata.obsm['X_vicdyf_d']
    dl_mat = adata.obsm['X_vicdyf_dl']
    z_mat = adata.obsm['X_vicdyf_z']
    zl_mat = adata.obsm['X_vicdyf_zl']
    # calculate transition rate
    mean_gene_norm = np.linalg.norm(adata.layers['vicdyf_mean_velocity'], axis=1)
    # embed z    
    if not fix_embed:
        z_embed = embed_z_adata(adata, n_neighbors=n_neighbors, min_dist=min_dist)
        adata.obsm['X_vicdyf_umap'] = z_embed
    else:
        z_embed = adata.obsm['X_vicdyf_umap']
    if zl_mat.shape[1] != 2:
        z_embed, mean_d_embed =  calc_d_embed_grad(zl_mat, dl_mat, z_embed, nn=n_neighbors)
        re_z_embed, stoc_d_embed =  calc_d_embed_grad(zl_mat, d_mat, z_embed, nn=n_neighbors)
    else:
        stoc_d_embed = d_mat
        mean_d_embed = dl_mat
    adata.obsm['X_' + embed_mode + 'umap'] = z_embed
    adata.obsm['X_' + embed_mode + 'sdumap'] = np.array(stoc_d_embed) * mean_gene_norm.reshape(-1, 1) / np.linalg.norm(stoc_d_embed, axis=1, keepdims=True)
    adata.obsm['X_' + embed_mode + 'mdumap'] = np.array(mean_d_embed) * mean_gene_norm.reshape(-1, 1) / np.linalg.norm(mean_d_embed, axis=1, keepdims=True)
    return(adata)


def embed_tr_mat(z_embed, tr_mat, gene_norm):
    z_embed = torch.tensor(z_embed)
    cell_num = z_embed.shape[0]
    gene_num = z_embed.shape[1]
    zz_diff_mat = z_embed.view(1, cell_num, gene_num) - z_embed.view(cell_num, 1, gene_num)
    norm_fix_mat = (zz_diff_mat == 0).type(torch.DoubleTensor)
    zz_diff_mat /= torch.norm(zz_diff_mat, dim=2, p=2, keepdim=True) + norm_fix_mat
    d_embed = torch.sum(tr_mat.view(cell_num, cell_num, 1) * zz_diff_mat, dim=1).view(cell_num, 2)
    d_norm = np.linalg.norm(d_embed, axis=1)
    d_embed *= (gene_norm.reshape(-1, 1) / d_norm.reshape(-1, 1))
    return(d_embed)
    
@torch.no_grad()
def estimate_cond_dynamics(adata, cond, model, n=1):
    orig_device = model.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    c = torch.zeros_like(torch.tensor(adata.obsm['condition'].values)).float().to(device)
    c[:, adata.obsm['condition'].columns == cond] = 1.0
    s = torch.tensor(adata.obsm['batch'].values).float().to(device)
    z = torch.tensor(adata.obsm['X_vicdyf_zl']).to(device)
    model.enc_d.norm_input = False
    d, qd = model.enc_d(torch.cat([z, c], dim=1))
    if n > 1:
        d = model.d_coeff * qd.sample_n(n)
    else:
        d = model.d_coeff * qd.loc
    pxd_zd_ld = model.dec_z(z + d, s)
    pxmd_zd_ld = model.dec_z(z - d, s)
    diff_px_zd_ld = pxd_zd_ld - pxmd_zd_ld    
    model.to(orig_device)
    return diff_px_zd_ld.detach().cpu().numpy(), d.detach().cpu().numpy()

def parse_gmt(gmt_fname):
    lines = open(gmt_fname, mode='r').readlines()
    gmt_dict = {
        line.split('\t')[0]: line.split('\t')[2:]
        for line in lines
    }
    return gmt_dict



def intersect_top_degs(key, gmt_dict, deg_df, topn=10):
    gene_list = gmt_dict[key]
    deg_df = deg_df.sort_values('scores', ascending=False)
    top_group_genes = deg_df.index[deg_df.index.str.upper().isin(gene_list)][:topn]
    return top_group_genes


def plot_vicdyf_umap(adata, gene_list, layer_name=None, use_raw=False, ax=None, cmap=None, **kwargs):
    sc.pl.scatter(adata, color=gene_list, use_raw=use_raw, layers=layer_name, basis='vicdyf_umap', ax=ax, sort_order=False,  color_map=cmap, **kwargs)

def extract_two_condition_adata(adata, cond1, cond2, l_ratio=0.3, u_ratio=0.7, nn=30):
    # extract cond1 and cond2  cellstate
    adata = adata[adata.obsm['condition'][[cond1, cond2]].sum(axis=1) > 0]
    # calculate cond1 ratio
    cond1_ratio = basic.calculate_neighbor_ratio(adata.obsm['X_vicdyf_zl'], adata.obsm['condition'][cond1].values, nn=nn)
    adata.obs['cond1_ratio'] = cond1_ratio
    cond2_ratio = basic.calculate_neighbor_ratio(adata.obsm['X_vicdyf_zl'], adata.obsm['condition'][cond2].values, nn=nn)
    adata.obs['cond2_ratio'] = cond2_ratio
    # extract middle cond1 ratio
    l1_ratio = l_ratio * np.mean(cond1_ratio)
    l2_ratio = l_ratio * np.mean(cond2_ratio)
    adata = adata[adata.obs.query('cond1_ratio > @l1_ratio and cond2_ratio > @l2_ratio').index]
    return adata

def estimate_two_cond_dynamics(adata, cond1, cond2, lit_envdyn):
    sub_adata = adata
    for cond in [cond1, cond2]:
        cond_vel, cond_d = estimate_cond_dynamics(sub_adata, cond, lit_envdyn)
        sub_adata.obsm[f'cond_d_{cond}'] = cond_d
        sub_adata.layers[f'cond_vel_{cond}'] = cond_vel
    sub_adata.layers['cond_vel_diff'] = sub_adata.layers[f'cond_vel_{cond2}'] -  sub_adata.layers[f'cond_vel_{cond1}']
    return sub_adata


def make_sample_one_hot_mat(adata, sample_key='sample', stored_obsm_key='batch'):
    sidxs = np.sort(adata.obs[sample_key].unique())
    sidx_num = sidxs.shape[0]
    b = np.array([
        (sidxs == sidx).astype(int)
        for sidx in adata.obs[sample_key]]).astype(float)
    cond_df = pd.DataFrame(b, index=adata.obs_names, columns=sidxs.astype(str))
    adata.obsm[stored_obsm_key] = cond_df
    b = torch.tensor(b).float()
    return b, adata

def make_multi_hot(dim, indexs):
    vec = np.zeros(dim)
    vec[indexs] = 1
    return vec

def mask_neighbors(z, nn=15):
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(z.cpu().detach().numpy())
    distances, indices = nbrs.kneighbors(z.cpu().detach().numpy())
    mask = np.array([make_multi_hot(z.size()[0], idxs) for idxs in indices])
    return(mask)

def calc_self_neighbors(z, nn):
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(z.cpu().detach().numpy())
    distances, indices = nbrs.kneighbors(z.cpu().detach().numpy())
    return distances, indices

def calc_tr_mat_nn(z, d, sigma, nn=30):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    distances, indices = calc_self_neighbors(z, nn=nn)
    zz_diff_mat = z[indices[:, 1:]] - z.unsqueeze(1)
    zz_diff_mat /= torch.norm(zz_diff_mat, dim=2, p=2, keepdim=True) 
    d = d.unsqueeze(1)
    d /= torch.norm(d, dim=2, p=2, keepdim=True)
    cos_sim = torch.sum(d * zz_diff_mat, dim=2)
    tr_mat = torch.exp(cos_sim / sigma)
    tr_mat /= torch.sum(tr_mat, axis=1, keepdim=True)
    tr_mat = (tr_mat - 1/cell_num)
    return(tr_mat, indices[:, 1:])



def calc_tr_mat(z, od, sigma, nn=30):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    mask_mat = mask_neighbors(z, nn=nn)
    zz_diff_mat = z.view(1, cell_num, gene_num) - z.view(cell_num, 1, gene_num)
    norm_fix_mat = (zz_diff_mat == 0).type(torch.DoubleTensor)
    zz_diff_mat /= torch.norm(zz_diff_mat, dim=2, p=2, keepdim=True) + norm_fix_mat
    d = od.view(cell_num, 1, gene_num)
    d = d / torch.norm(d, dim=2, p=2, keepdim=True)
    cos_sim = torch.sum(d * zz_diff_mat, dim=2)
    cos_sim = cos_sim * mask_mat
    tr_mat = torch.exp(cos_sim / sigma)
    tr_mat /= torch.sum(tr_mat, axis=1).view(-1, 1)
    tr_mat = (tr_mat - 1/cell_num) * mask_mat
    return(tr_mat)

  
def calc_dembed(z_embed, tr_mat, indices):
	dest_embed = (tr_mat.unsqueeze(2) * z_embed[indices]).sum(dim=1)
	dembed = dest_embed - z_embed
	return dembed


def calc_int_dembed(z_embed, z, d, sigma, nn=30):
    dembed =  calc_d_embed_grad(z, d, z_embed, nn=nn)[1]
    return dembed

@torch.no_grad()    
def update_dembed_full(adata, model, ds, sigma=0.05, embed_mode='vicdyf_', nn=30, mdist=0.1, dz_var_prop=0.05, no_tr=False, fix_embed=False, skip_norm=False, only_vae=False):
    z, qz, d, qd, px_z_ld, pu_zd_ld, gene_vel, mean_gene_vel, batch_std_mat = model.extract_info(ds[:])
    zl = qz.loc
    dl = qd.loc
    sll, ull = model.encode_size(ds[:])
    adata.obs['sl'] = sll.numpy()
    adata.obs['ul'] = ull.numpy()
    adata.layers['lambda'] = px_z_ld.numpy()
    adata.obsm['X_vicdyf_z'] = z.numpy()
    adata.obsm['X_vicdyf_zl'] = zl.numpy()
    if not only_vae:
        adata.layers['ulambda'] = pu_zd_ld.numpy()
        adata.layers['vicdyf_velocity'] = gene_vel.numpy()
        adata.layers['vicdyf_mean_velocity'] = mean_gene_vel.numpy()
        adata.layers['vicdyf_fluctuation'] = batch_std_mat.numpy()
        adata.obsm['X_vicdyf_d'] = d.numpy()
        adata.obsm['X_vicdyf_dl'] = dl.numpy()
        adata.obs['vicdyf_fluctuation'] = adata.layers['vicdyf_fluctuation'].mean(axis=1)
        adata.obs['vicdyf_mean_velocity'] = np.abs(adata.layers['vicdyf_mean_velocity']).mean(axis=1)
        adata.obs['vicdyf_velocity'] = np.abs(adata.layers['vicdyf_velocity']).mean(axis=1)
        if not skip_norm:
            for measure in ['fluctuation', 'mean_velocity', 'velocity']:
                adata.layers[f'norm_vicdyf_{measure}'] = adata.layers[f'vicdyf_{measure}'] / np.mean(adata.layers[f'lambda'], axis=0)
        # calculate transition rate
        adata = update_only_dembed(adata, fix_embed=fix_embed, n_neighbors=nn, min_dist=mdist)
    else:
        z_embed = embed_z_adata(adata, n_neighbors=nn, min_dist=mdist)
        adata.obsm['X_vicdyf_umap'] = z_embed
    adata.obsm['X_umap'] = adata.obsm['X_vicdyf_umap']
    return(adata)



@torch.no_grad()    
def update_latent(adata, model, ds, sigma=0.05, embed_mode='vicdyf_', nn=30, mdist=0.1, dz_var_prop=0.05, scale_z=False):
    z, qz = model.encode_z(ds[:])
    zl = qz.loc
    adata.obsm['X_vicdyf_z'] = z.numpy()
    adata.obsm['X_vicdyf_zl'] = zl.numpy()
    # embed z
    z_embed = embed_z_adata(adata, n_neighbors=nn, min_dist=mdist)
    adata.obsm['X_vicdyf_umap'] = z_embed
    return(adata)


def separate_dataset(ds, test_ratio = 0.1, val_ratio = 0.05):
    total_size = len(ds)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.dataset.random_split(ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    return(train_ds, val_ds, test_ds)


def get_knn_indeces(z, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z)
    distances, indices = nbrs.kneighbors(z)
    return indices


def get_original_barcodes(cells):
    original_barcodes = pd.Series(cells).str.extract('([ACGT]{16})').iloc[:, 0].values
    return original_barcodes

def integrate_kb_cellranger(kb_adata, cellranger_adata):
    kb_adata.obs_names = get_original_barcodes(kb_adata.obs_names)
    cellranger_adata.obs_names = get_original_barcodes(cellranger_adata.obs_names)
    common_cells = np.intersect1d(kb_adata.obs_names, cellranger_adata.obs_names)
    common_genes = np.intersect1d(kb_adata.var_names, cellranger_adata.var_names)
    kb_adata = kb_adata[common_cells, common_genes]
    cellranger_adata = cellranger_adata[common_cells, common_genes]
    cellranger_adata.layers['spliced'] = kb_adata.layers['spliced']
    cellranger_adata.layers['unspliced'] = kb_adata.layers['unspliced']
    return cellranger_adata


def set_ens2symbol(kb_adata):
    if 'ENS' == kb_adata.var_names[0][:3]:
        mg = mygene.MyGeneInfo()
        ens = np.array([orig_id.split('.')[0] for orig_id in kb_adata.var_names])
        kb_adata.var_names = ens
        kb_adata.var_names_make_unique()
        ens = kb_adata.var_names
        info_df = pd.DataFrame(mg.querymany(ens, scopes='ensembl.gene'))
        info_df = info_df[['query', 'symbol']].dropna().drop_duplicates()
        info_df = info_df.groupby('query').head(1)
        info_df.index = info_df['query'].values
        kb_adata = kb_adata[:, info_df['query'].values]
        kb_adata.var_names = info_df['symbol']
        kb_adata.var_names_make_unique()
    return kb_adata

def make_same_batch_knn(adata, k):
    z = adata.obsm['X_vicdyf_zl']
    knn_indices = get_knn_indeces(z, k)
    self_batch_mat = np.repeat(np.array(adata.obs['sample'].values[knn_indices[:, :1]]).reshape((-1, 1)), k, axis=1)
    batch_knn_mat = adata.obs['sample'].values[knn_indices] == self_batch_mat
    return batch_knn_mat

def norm_z_score(x):
    nx = x / np.mean(np.array(x), axis=1, keepdims=True)
    nx = scipy.stats.zscore(nx)
    return nx


def make_neighbor_mean_exp(x, k):
    nx = norm_z_score(x)
    pca = sklearn.decomposition.PCA(n_components=30).fit_transform(nx)
    knn_indeces = get_knn_indeces(pca, k)
    mean_nx = np.mean(np.array(nx[knn_indeces]), axis=1)
    return mean_nx


def make_mean_diff_df(v1, v2, key):
    mean_diff_df = pd.DataFrame({
        f'{key}_mean': (v1 + v2) / 2,
        f'{key}_diff': (v1 - v2)
        })
    return mean_diff_df


def calc_cond2_ratio(adata, cond1, cond2):
    cond2_ratio = basic.calculate_neighbor_ratio(adata.obsm['X_vicdyf_zl'], adata.obsm['condition'][cond2].values)
    return cond2_ratio

def calc_neighbor_ratio(adata, cond, cond_key):
    cond_ratio = basic.calculate_neighbor_ratio(adata.obsm['X_vicdyf_zl'], (adata.obs[cond_key].values == cond).astype(int))
    return cond_ratio

@torch.no_grad()
def make_cond_diff_adata(adata, model, cond1, cond2, only_pos=False, u_ratio=0.7, l_ratio=0.3):
    sub_adata = adata[adata.obs.condition.isin([cond1, cond2])]
    sub_adata.uns['cond1'] = cond1
    sub_adata.uns['cond2'] = cond2
    sub_adata = adata[adata.obs.condition.isin([cond1, cond2])]
    cond2_ratio = basic.calculate_neighbor_ratio(sub_adata.obsm['X_vicdyf_zl'], sub_adata.obsm['condition'][cond2].values)
    sub_adata.obs['cond2_ratio'] = cond2_ratio
    model.eval()
    sub_adata = sub_adata[np.logical_and(sub_adata.obs.cond2_ratio > l_ratio, sub_adata.obs.cond2_ratio < u_ratio)]
    sub_adata.layers['cond1_dyn'], sub_adata.obsm['cond1_d'] = estimate_cond_dynamics(sub_adata, cond1, model)
    sub_adata.layers['cond2_dyn'] , sub_adata.obsm['cond2_d']= estimate_cond_dynamics(sub_adata, cond2, model)
    sub_adata.layers['diff_dyn'] = sub_adata.layers['cond2_dyn'] - sub_adata.layers['cond1_dyn']
    sub_adata.obs['cdiff'] = f'{cond1}_{cond2}'
    if only_pos:
        sub_adata.layers['diff_dyn'][sub_adata.layers['diff_dyn'] < 0] = 0
    sub_adata.layers['norm_diff_dyn'] = sub_adata.layers['diff_dyn'] / adata.layers['lambda'].max(axis=0)
    # sub_adata.layers['norm_diff_dyn'] = (sub_adata.layers['diff_dyn'] - sub_adata.layers['diff_dyn'].mean(axis=0))  / sub_adata.layers['lambda'].max(axis=0)
    sub_adata.obs['cdiff_strength'] = np.linalg.norm(sub_adata.layers['norm_diff_dyn'], axis=1)
    return sub_adata


def random_x_hot(dim, num):
    if num > dim:
        raise ValueError('dim must be greater than num')
    x_hot = np.zeros(dim)
    pos_idxs = np.random.choice(np.arange(dim), size=num, replace=False)
    x_hot[pos_idxs] = 1
    return x_hot


def score_ligac(ligac, pligac, cell_num, lig_num, p_num):
    lig_rank = (ligac.values.reshape((cell_num, ligac.shape[1], 1)) < pligac.values.reshape((cell_num, 1, p_num))).sum(axis=2)
    lig_ratio = (lig_rank + 1) / (p_num + 1)
    ligac_score = - pd.Series(np.log(lig_ratio).min(axis=0), index=ligac.columns)
    return ligac_score

def calc_ct_ligac(sub_adata, norm_lt_df, common_genes):
    ligac = (sub_adata[:, common_genes].layers['norm_diff_dyn'] @ norm_lt_df)
    t_ligac = ligac.assign(cdiff=sub_adata.obs.cdiff.values, celltype=sub_adata.obs.celltype.values).groupby(['celltype', 'cdiff']).mean()
    return t_ligac

def calc_t_ligac(sub_adata, norm_lt_df, common_genes):
    ligac = (sub_adata[:, common_genes].layers['norm_diff_dyn'] @ norm_lt_df).abs()
    t_ligac = ligac.assign(cdiff=sub_adata.obs.cdiff.values).groupby('cdiff').mean()
    return t_ligac


def make_sparse_plt(common_genes, pos_num, p_num, p):
    rows = np.random.choice(np.arange(common_genes.shape[0]), pos_num * p_num, p=p)
    cols = np.repeat(np.arange(p_num), pos_num)
    vals = np.ones_like(cols)
    plt_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(common_genes.shape[0], p_num))
    return plt_mat

def make_plt_df(common_genes, pos_num, p_num):
    plt_df = pd.DataFrame(np.column_stack([random_x_hot(common_genes.shape[0], pos_num) for _ in range(p_num)]), index=common_genes)
    return plt_df


def make_norm_lt_plt_from_adata(adata, lt_df, p_num=1000):
    common_genes = np.intersect1d(lt_df.index.astype(str), adata.var_names)
    lt_df = lt_df.loc[common_genes, :]
    norm_lt_df = make_norm_lt_df(lt_df, q=0.99)
    pos_num = norm_lt_df.sum(axis=0).iloc[0]
    target_props = norm_lt_df.sum(axis=1).values
    target_props = target_props / target_props.sum()
    plt_mat = make_sparse_plt(norm_lt_df.index, pos_num, p_num, p=target_props)
    adata = adata[:, common_genes]
    return norm_lt_df, plt_mat, adata


def make_non_zero_mat(mat):
    mat[mat < 0] = 0
    return mat

def target_dembed(z, target_z, d, embed, target_embed, mv_gene_norm):
    int_z = np.concatenate([z, target_z], axis=0)
    int_d = np.concatenate([d, np.zeros_like(target_z)], axis=0)
    int_embed = np.concatenate([embed, target_embed], axis=0)
    int_embed, int_d_embed = calc_d_embed_grad(int_z, int_d, int_embed)
    mv_gene_norm = np.concatenate([mv_gene_norm, np.zeros(target_embed.shape[0])])
    md_umap = mv_gene_norm.reshape((-1, 1)) * int_d_embed / np.linalg.norm(int_d_embed, axis=1, keepdims=True)
    return md_umap[:z.shape[0]]


def neighbor_u(qz, nn_z, nn_u, z_sigma):
    p = jax.nn.softmax((- (nn_z - qz)**2 / (z_sigma **2)).sum(axis=1))
    neighbor_u = (p.reshape((-1, 1)) * nn_u).sum(axis=0)
    return neighbor_u

@torch.no_grad()
def calc_d_embed_grad(z_mat, d_mat, u_mat, nn=30): 
    nbrs = NearestNeighbors(n_neighbors=nn).fit(z_mat)
    distances, indeces = nbrs.kneighbors(z_mat)
    z_sigmas = distances[:, int(nn/10)] 
    nn_z_mat = z_mat[indeces]
    nn_u_mat = u_mat[indeces]
    f_z_embed = vmap(neighbor_u, in_axes=(0, 0, 0, 0))
    z_embed = f_z_embed(z_mat, nn_z_mat, nn_u_mat, z_sigmas)
    jacob_nnu = lambda z, d, nn_z, nn_u, z_sigma: jvp(lambda qz: neighbor_u(qz, nn_z, nn_u, z_sigma), (z, ), (d, ))[1]
    f_d_embed = vmap(jacob_nnu, in_axes=(0, 0, 0, 0, 0))
    d_embed = f_d_embed(z_mat, d_mat, nn_z_mat, nn_u_mat, z_sigmas)
    return np.array(z_embed), np.array(d_embed)


def calc_ligac_significance(condiff_dyn_adata, lt_df):
    sub_adata = condiff_dyn_adata
    sub_adata.var_names = sub_adata.var_names.str.upper()
    common_ligands = np.intersect1d(lt_df.columns, sub_adata.var_names)
    norm_lt_df = pd.DataFrame((lt_df.values > lt_df.quantile(0.99, axis=0).values).astype(int), index=lt_df.index, columns=lt_df.columns).loc[:, common_ligands]
    pos_num = norm_lt_df.sum(axis=0).max()
    norm_lt_df = norm_lt_df.loc[:, norm_lt_df.sum(axis=0) == pos_num]
    ligac = (sub_adata[:, common_genes].layers['norm_diff_dyn'] @ norm_lt_df).abs()
    t_ligac = commons.calc_ct_ligac(sub_adata, norm_lt_df, common_genes)
    p_num = 10000
    # t_pligac = calc_ct_ligac(sub_adata, make_plt_df(common_genes, pos_num, p_num))
    lt_pligac_list = [
        commons.calc_ct_ligac(sub_adata, commons.make_plt_df(common_genes, pos_num, p_num), common_genes)
        for _ in range(10)]
    t_pligac = pd.concat(lt_pligac_list, axis=1)
    int_total_ligac = pd.concat([t_ligac, t_pligac], axis=1).rank(axis=1).max(axis=0)
    total_ligac = int_total_ligac[:ligac.shape[1]]
    total_pligac = int_total_ligac[ligac.shape[1]:]
    total_ligac = total_ligac[common_ligands].sort_values(ascending=False)
    ligand_pvals = total_ligac.apply(lambda x: (total_pligac > x).mean())
    ligand_fdrs = pd.Series(multipletests(ligand_pvals, method='fdr_bh')[1], index=ligand_pvals.index)
    total_df = pd.DataFrame({
        'ligac': total_ligac,
        'pval': ligand_pvals,
        'qval': ligand_fdrs,
        'ligand': total_ligac.index
    })
    return total_df

def calculate_ligac(sub_adata, lt_df):
    sub_adata.var_names.values
    common_genes = select_common_lt_targets(sub_adata, lt_df)
    norm_lt_df = make_norm_lt_df(lt_df.loc[common_genes])
    sparse_lt_mat = sparse.csr_matrix(norm_lt_df.values)
    ligands = norm_lt_df.columns 
    ligac_df = make_ligac_df(sub_adata, sparse_lt_mat, common_genes, ligands)
    sub_adata.obsm['ligac'] = ligac_df
    return sub_adata


def make_lig_targets(lt_df, q=0.99):
    lig_targets = {
        ligand: list(lt_df.index[lt_df[ligand] > lt_df[ligand].quantile(q)])
        for ligand in lt_df.columns}
    return lig_targets


def make_norm_lt_df(lt_df, q=0.99):
    norm_lt_df = pd.DataFrame((lt_df.values > lt_df.quantile(q, axis=0).values).astype(int), index=lt_df.index, columns=lt_df.columns)
    pos_num = norm_lt_df.sum(axis=0).max()
    norm_lt_df = norm_lt_df.loc[:, norm_lt_df.sum(axis=0) == pos_num]
    return norm_lt_df

def make_ligac_df(sub_adata, sparse_lt_mat, common_genes, ligands, only_pos=False, only_neg=False):
    target_mat = copy.deepcopy(sub_adata.layers['norm_diff_dyn'])
    if only_pos:
        target_mat[target_mat < 0] = 0
    elif only_neg:
        target_mat[target_mat > 0] = 0
    ligac_df = pd.DataFrame((sub_adata[:, common_genes].layers['norm_diff_dyn'] @ sparse_lt_mat), columns=ligands, index=sub_adata.obs_names)
    return ligac_df

def make_t_ligac_df(sub_adata, sparse_lt_mat, common_genes, ligands):
    ligac_df = make_ligac_df(sub_adata, sparse_lt_mat, common_genes, ligands)
    ligac_df['cellid'] = ligac_df.index
    t_ligac_df = ligac_df.melt(id_vars=['cellid'], value_vars=ligands, var_name='ligand', value_name='ligac')
    return t_ligac_df

def select_common_lt_targets(sub_adata, lt_df, q=0.99):
    genes = sub_adata.var_names
    common_genes = np.intersect1d(lt_df.index.values.astype(str), genes)
    lt_df = lt_df.loc[common_genes]
    lt_df = lt_df.loc[(lt_df.values > lt_df.quantile(q, axis=0).values).astype(int).sum(axis=1) < lt_df.shape[1] * 0.5]
    sub_adata = sub_adata[:, lt_df.index]
    return sub_adata, lt_df

def evaluate_ligac_sig(int_t_ligac_df, t_ligac_df, t_pligac_df):
    int_t_ligac_df['cum_type'] = int_t_ligac_df.type.cumsum()
    int_t_ligac_df['rev_cum_type'] = t_ligac_df.shape[0] - int_t_ligac_df.cum_type + 1
    int_t_ligac_df['type1_cum_prop'] = int_t_ligac_df.cum_type / t_ligac_df.shape[0]
    int_t_ligac_df['type0_cum_prop'] = (np.arange(int_t_ligac_df.shape[0]) + 1 - int_t_ligac_df.cum_type) / t_pligac_df.shape[0]
    int_t_ligac_df['type1_rev_cum_prop'] = int_t_ligac_df.rev_cum_type / t_ligac_df.shape[0]
    int_t_ligac_df['type0_rev_cum_prop'] = (np.arange(int_t_ligac_df.shape[0], 0, -1) - int_t_ligac_df.rev_cum_type) / t_pligac_df.shape[0]
    int_t_ligac_df['FDR_neg'] = int_t_ligac_df.type0_cum_prop / (int_t_ligac_df.type1_cum_prop)
    int_t_ligac_df['FDR_pos'] = int_t_ligac_df.type0_rev_cum_prop / (int_t_ligac_df.type1_rev_cum_prop)
    return int_t_ligac_df



def calculate_ligac_significance(sub_adata, lt_df, p_num=10000, only_pos=False, only_neg=False):
    sub_adata.var_names = sub_adata.var_names.str.upper()
    common_genes = select_common_lt_targets(sub_adata, lt_df)
    norm_lt_df = make_norm_lt_df(lt_df.loc[common_genes])
    sparse_lt_mat = sparse.csr_matrix(norm_lt_df.values)
    ligands = norm_lt_df.columns 
    ligac_df = make_ligac_df(sub_adata, sparse_lt_mat, common_genes, ligands, only_pos=only_pos, only_neg=only_neg)
    # sub_adata.varm['ligac'] = t_ligac_df
    gene_ps = norm_lt_df.sum(axis=1).values
    gene_ps = gene_ps / gene_ps.sum()
    sparse_plt_mat = make_sparse_plt(common_genes, norm_lt_df.sum(axis=0).values[0], p_num, gene_ps)
    pligac_df = make_ligac_df(sub_adata, sparse_plt_mat, common_genes, np.arange(sparse_plt_mat.shape[1]), only_pos=only_pos, only_neg=only_neg)
    ligacs = pd.Series(np.linalg.norm(ligac_df.values, ord=2, axis=0), index=ligac_df.columns)
    pligacs = pd.Series(np.linalg.norm(pligac_df.values, ord=2, axis=0), index=pligac_df.columns)
    pvals = pd.Series([(ligac < pligacs).sum() / pligacs.shape[0] for ligac in ligacs], index=ligacs.index)
    ligand_fdrs = pd.Series(multipletests(pvals, method='fdr_bh')[1], index=pvals.index)
    return ligand_fdrs


def calculate_sc_ligac_sig(sub_adata, lt_df):
    sub_adata.var_names = sub_adata.var_names.str.upper()
    common_genes = select_common_lt_targets(sub_adata, lt_df)
    norm_lt_df = make_norm_lt_df(lt_df.loc[common_genes])
    sparse_lt_mat = sparse.csr_matrix(norm_lt_df.values)
    ligands = norm_lt_df.columns 
    t_ligac_df = make_t_ligac_df(sub_adata, sparse_lt_mat, common_genes, ligands)
    t_ligac_df['type'] = 1
    gene_ps = norm_lt_df.sum(axis=1).values
    gene_ps = gene_ps / gene_ps.sum()
    sparse_plt_mat = make_sparse_plt(common_genes, norm_lt_df.sum(axis=0).values[0], norm_lt_df.shape[1], gene_ps)
    t_pligac_df = make_t_ligac_df(sub_adata, sparse_plt_mat, common_genes, ligands)
    t_pligac_df['type'] = 0
    int_t_ligac_df = pd.concat([t_ligac_df, t_pligac_df], axis=0).sort_values('ligac', axis=0)
    int_t_ligac_df = evaluate_ligac_sig(int_t_ligac_df, t_ligac_df, t_pligac_df)
    return int_t_ligac_df

def ingets_integration(q_adata, r_adata):
    genes = np.intersect1d(q_adata.var_names, r_adata.var_names)
    q_adata = q_adata[:, genes]
    r_adata = r_adata[:, genes]
    sc.pp.pca(r_adata)
    sc.pp.neighbors(r_adata)
    r_adata.obsm['X_umap'] = r_adata.obsm['X_vicdyf_umap']
    sc.tl.ingest(q_adata, r_adata, 'leiden', embedding_method=('umap', 'pca'))
    return q_adata, r_adata

def transform_exp(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata)
    return adata

def extract_cluster_adata(adata, cluster_label, cluster):
    sub_adata = adata[adata.obs[cluster_label] == cluster]
    return sub_adata

def extract_top_genes(vec, genes, k):
    scores = pd.Series(vec, index=genes).sort_values()
    bottom_genes = scores.iloc[:k].index
    top_genes = scores.iloc[-k:].index
    return bottom_genes, top_genes

def extract_top_mean_genes(adata, layer, k=5):
    means = adata.layers[layer].mean(axis=0)
    return extract_top_genes(means, adata.var_names, k)


def extract_top_norm_genes(adata, layer, k=5):
    norms = np.linalg.norm(adata.layers[layer], axis=0)
    return extract_top_genes(norms, adata.var_names, k)[1]


def determine_int_adata(adata):
    deviation = np.abs((adata.X - adata.X.astype(int))).sum()
    try:
        if deviation > 0:
            raise ValueError('adata.X must be integer')
    except ValueError as e:
        print(e)
        raise


def gene_selection(adata, n_top_genes=4000, exp_prop=0.1):
    determine_int_adata(adata)
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_genes(adata, min_cells=int(adata.shape[0] * exp_prop))
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata

def calc_ligac_significance(condiff_dyn_adata, lt_df):
    sub_adata = condiff_dyn_adata
    sub_adata.var_names = sub_adata.var_names.str.upper()
    common_ligands = np.intersect1d(lt_df.columns, sub_adata.var_names)
    norm_lt_df = pd.DataFrame((lt_df.values > lt_df.quantile(0.99, axis=0).values).astype(int), index=lt_df.index, columns=lt_df.columns).loc[:, common_ligands]
    pos_num = norm_lt_df.sum(axis=0).max()
    norm_lt_df = norm_lt_df.loc[:, norm_lt_df.sum(axis=0) == pos_num]
    ligac = (sub_adata[:, common_genes].layers['norm_diff_dyn'] @ norm_lt_df).abs()
    t_ligac = commons.calc_ct_ligac(sub_adata, norm_lt_df, common_genes)
    p_num = 10000
    # t_pligac = calc_ct_ligac(sub_adata, make_plt_df(common_genes, pos_num, p_num))
    lt_pligac_list = [
        commons.calc_ct_ligac(sub_adata, commons.make_plt_df(common_genes, pos_num, p_num), common_genes)
        for _ in range(10)]
    t_pligac = pd.concat(lt_pligac_list, axis=1)
    int_total_ligac = pd.concat([t_ligac, t_pligac], axis=1).rank(axis=1).max(axis=0)
    total_ligac = int_total_ligac[:ligac.shape[1]]
    total_pligac = int_total_ligac[ligac.shape[1]:]
    total_ligac = total_ligac[common_ligands].sort_values(ascending=False)
    ligand_pvals = total_ligac.apply(lambda x: (total_pligac > x).mean())
    ligand_fdrs = pd.Series(multipletests(ligand_pvals, method='fdr_bh')[1], index=ligand_pvals.index)
    total_df = pd.DataFrame({
        'ligac': total_ligac,
        'pval': ligand_pvals,
        'qval': ligand_fdrs,
        'ligand': total_ligac.index
    })
    return total_df



def change_length(v_mat, len_vec):
    v_mat = len_vec.reshape(-1, 1) * v_mat / np.linalg.norm(v_mat, axis=1, keepdims=True)
    return v_mat

def select_grid_values(u_mat, grid_num, values):
    u1, u2 = u_mat[:, 0], u_mat[:, 1]
    u1_res = (u1.max() - u1.min()) / grid_num
    u2_res = (u2.max() - u2.min()) / grid_num
    grid_df = pd.DataFrame({'du1': (u1 / u1_res).astype(int), 'du2': (u2 / u2_res).astype(int), 'value': values, 'num_idx': np.arange(u1.shape[0])})
    return grid_df.sort_values('value', ascending=False).groupby(['du1', 'du2']).head(n=1)

def visuazlie_condiff_dynamics(adata, cdiff_adata, cond1, cond2, plot_fname, plot_num=30, plot_top_prop=0.8, dembed_nn=100, same_size=False, grid_divs=20):
    conds = [cond1, cond2]
    t_adata = adata
    # t_adata = cdiff_adata
    cond2_ratio = basic.calculate_neighbor_ratio(t_adata.obsm['X_vicdyf_zl'], (t_adata.obs['condition'] == cond2).values, nn=100)
    t_adata.obs['cond2_ratio'] = cond2_ratio
    ot_adata = t_adata[~t_adata.obs_names.isin(cdiff_adata.obs_names)]
    exp_mat = np.concatenate([cdiff_adata.obsm['X_vicdyf_zl'], ot_adata.obsm['X_vicdyf_zl']], axis=0)
    embed = np.concatenate([cdiff_adata.obsm['X_vicdyf_umap'], ot_adata.obsm['X_vicdyf_umap']], axis=0)
    pseudo_mat = np.zeros(ot_adata.obsm['X_vicdyf_zl'].shape)
    cond1_d = np.concatenate([cdiff_adata.obsm[f'cond_d_{cond1}'], pseudo_mat], axis=0)
    cond2_d = np.concatenate([cdiff_adata.obsm[f'cond_d_{cond2}'], pseudo_mat], axis=0)
    cdiff_adata.obs[f'{cond1} vel'] = np.linalg.norm(cdiff_adata.layers[f'cond_vel_{cond1}'], axis=1)
    cdiff_adata.obs[f'{cond2} vel'] = np.linalg.norm(cdiff_adata.layers[f'cond_vel_{cond2}'], axis=1)
    cond1_vels = np.concatenate([np.linalg.norm(cdiff_adata.layers[f'cond_vel_{cond1}'], axis=1), np.ones(ot_adata.shape[0])], axis=0)
    cond2_vels = np.concatenate([np.linalg.norm(cdiff_adata.layers[f'cond_vel_{cond2}'], axis=1), np.ones(ot_adata.shape[0])], axis=0)
    if same_size:
        cond1_vels = np.ones_like(cond1_vels)
        cond2_vels = np.ones_like(cond2_vels)
    cond1_dembed = change_length(calc_int_dembed(embed, exp_mat, cond1_d, sigma=0.05, nn=dembed_nn), cond1_vels)
    cond2_dembed = change_length(calc_int_dembed(embed, exp_mat, cond2_d, sigma=0.05, nn=dembed_nn), cond2_vels)
    fig, axes = plt.subplots(1, 1, figsize=(12 * 1, 10 * 1))
    tot_embed = t_adata.obsm['X_vicdyf_umap']
    cm = plt.cm.get_cmap('copper')
    cnum = cdiff_adata.shape[0]
    cax = axes.scatter(tot_embed[:, 0], tot_embed[:, 1], s=10, c=t_adata.obs.cond2_ratio, cmap=cm)
    sembed = embed[:cnum] 
    plt.colorbar(cax, label=f'{cond2} ratio')
    cdiff_adata.layers['norm_cond_vel_diff'] = cdiff_adata.layers['cond_vel_diff'] / cdiff_adata.layers['lambda'].max(axis=0)
    cdiff_adata.obs['total_dyn_diff'] = np.linalg.norm(cdiff_adata.layers['norm_cond_vel_diff'], axis=1)
    total_dyn_diffs = cdiff_adata.obs['total_dyn_diff'].values
    grid_df = select_grid_values(embed[:cnum], grid_divs, total_dyn_diffs)
    diff_thresh = np.quantile(total_dyn_diffs, plot_top_prop)
    idx = grid_df.query('value > @diff_thresh').num_idx.values
    for dembed, color, label in zip([cond1_dembed, cond2_dembed], ['teal', 'coral'], [cond1, cond2]):
        axes.quiver(embed[:cnum][idx, 0], embed[:cnum][idx, 1], dembed[:cnum][idx, 0], dembed[:cnum][idx, 1], color=color, label=label, alpha=1)
    plt.legend()
    plt.savefig(plot_fname);plt.close('all')


def get_common_cells(adata1, adata2):
    common_cells = np.intersect1d(adata1.obs_names, adata2.obs_names)
    return common_cells

def safe_to_array(mat):
    if type(mat) == np.ndarray:
        return mat
    else:
        return mat.toarray()


def layer_transfer(adata1, adata2, layer):
    adata2.layers[layer] = np.full_like(safe_to_array(adata2.X), np.nan)
    common_cells = get_common_cells(adata1, adata2)
    common_cells = adata2.obs_names[adata2.obs_names.isin(common_cells)]
    adata2.layers[layer][adata2.obs_names.isin(common_cells)] = adata1[common_cells].layers[layer]
    return adata2

def cond_diff_transfer(cdiff_adata, adata, cond1, cond2):
    for layer in [f'cond_vel_{cond1}', f'cond_vel_{cond2}', 'cond_vel_diff', 'norm_cond_vel_diff']:
        adata = layer_transfer(cdiff_adata, adata, layer)
    return adata

def visualize_layers(adata, genes, layers):
    fig, axes = plt.subplots(len(layers), len(genes), figsize=(6 * len(genes), 5 * len(layers)))
    for i, layer in enumerate(layers):
        for j, gene in enumerate(genes):
            sc.pl.umap(adata, color=gene, layer=layer, ax=axes[i, j]) 
    return fig, axes


def visualize_condiff_marker_vel(adata, genes, cond1, cond2):
    fig, axes = plt.subplots(4, len(genes), figsize=(6 * len(genes), 5 * 4))
    for j, gene in enumerate(genes):
        cond_vels = np.concatenate([adata[:, gene].layers[f'cond_vel_{cond1}'], adata[:, gene].layers[f'cond_vel_{cond2}']])
        abs_max = max(np.abs(np.nanmax(cond_vels)), np.abs(np.nanmin(cond_vels)))
        sc.pl.umap(adata, color=gene, layer='lambda', ax=axes[0, j]) 
        sc.pl.umap(adata, color=gene, layer='cond_vel_diff', ax=axes[1, j], color_map='coolwarm', vcenter=0) 
        for i, cond in enumerate([cond1, cond2]):
            try:
                sc.pl.umap(adata, color=gene, layer=f'cond_vel_{cond}', ax=axes[i + 2, j], color_map='coolwarm', vmin=-abs_max, vcenter=0, vmax=abs_max, sort_order=True) 
            except:
                import pdb;pdb.set_trace()
    return fig, axes



def quantify_cond_response(est_adata, model):
    print('test')



@torch.no_grad()
def cond_diff_x(model, z, c, s):
    encd = model.enc_d
    enc_dmu = lambda vz, vc: encd.h2mu(encd.seq_nn(encd.x2h(torch.cat([vz, vc], dim=-1))))
    dl = enc_dmu(z, c)
    dec_f = lambda vz, vs: model.dec_z(vz, vs)
    dec_jvp = lambda vz, vd, vs: functorch.jvp(lambda vzz: dec_f(vzz, vs), (vz, ), (vd, ))[1]
    diff_px_zd_ld = functorch.vmap(dec_jvp, in_dims=(0, 0, 0))(z, dl, s)
    return diff_px_zd_ld


def calc_jacc_sum(adata, model, condition_key='coloc', batch_key='batch'):
    z = torch.tensor(adata.obsm['X_vicdyf_zl'])
    d = torch.tensor(adata.obsm['X_vicdyf_dl'])
    s = torch.tensor(adata.obsm[batch_key].values)
    c = torch.tensor(adata.obsm[condition_key]).float()
    encd = model.enc_d
    enc_dmu = lambda z: encd.h2mu(encd.seq_nn(encd.x2h(z)))
    dec_f = lambda vz, vs: model.dec_z(vz, vs)
    diff_dec_f = lambda vz, vd, vs: functorch.jvp(lambda vzz: dec_f(vzz, vs), (vz, ), (vd, ))[1]
    fenc_d = lambda vz, vc: enc_dmu(torch.cat([vz, vc], dim=-1))
    jac_v_c = lambda vz, vc, vs: functorch.jacrev(lambda vcc: diff_dec_f(vz, fenc_d(vz, vcc), vs))(vc)
    gene_num = adata.shape[1]
    c_num = c.size()[-1]
    jacc_c = torch.zeros(gene_num, c_num)
    for i in range(z.size()[0]):
        vz, vd, vc, vs = z[i], d[i], c[i], s[i]
        jacc_c += jac_v_c(vz, vc, vs).detach()
    return jacc_c

def get_tops(scores, topn=5, ascending=False):
    sorted_scores = scores.sort_values(ascending=ascending)
    return sorted_scores.iloc[:topn].index

def add_text(xs, ys, labels, ax):
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y))

def calc_nn_diff_x(nn_z, nn_c, nn_s, z, c, s, model, nn, max_vec):
    nn_diff_x = cond_diff_x(model, nn_z.reshape(-1, z.size()[-1]), nn_c.reshape(-1, c.size()[-1]), nn_s.reshape(-1, s.size()[-1])) / max_vec
    nn_diff_x = rearrange(nn_diff_x, '(b nn) d -> b nn d', nn=nn) 
    nn_diff_x = nn_diff_x.std(dim=1)
    return nn_diff_x

def evaluate_covariate_diff(est_adata, model, cond_key, nn=30, batch_key='batch'):
    z = torch.tensor(est_adata.obsm['X_vicdyf_zl'])
    s = torch.tensor(est_adata.obsm[batch_key].values)
    c = torch.tensor(est_adata.obsm[cond_key]).float()
    nbrs = NearestNeighbors(n_neighbors=nn).fit(z)
    distances, indeces = nbrs.kneighbors(z)
    nn_c = c[indeces]
    nn_z = repeat(z, 'b d -> b nn d', nn=nn)
    nn_s = repeat(s, 'b d -> b nn d', nn=nn)
    max_vec = torch.tensor(est_adata.layers['lambda'].max(axis=0))
    idx_list = make_idx_list(est_adata.shape[0])
    nn_diff_x = torch.cat([
       calc_nn_diff_x(nn_z[idx], nn_c[idx], nn_s[idx], z, c, s, model, nn, max_vec)
       for idx in idx_list], dim=0)
    nn_diff_norm = nn_diff_x.norm(dim=-1)
    est_adata.obs['nn_diff_std'] = nn_diff_norm.detach().numpy()
    est_adata.layers['nn_diff'] = nn_diff_x.detach().numpy()
    return est_adata

def make_top_diff_adata(est_adata, topq=0.9, res=0.5):
    top_diff_adata = est_adata[est_adata.obs.nn_diff_std > est_adata.obs.nn_diff_std.quantile(topq)]
    top_diff_adata.X = top_diff_adata.layers['nn_diff']
    sc.tl.pca(top_diff_adata)
    sc.pp.neighbors(top_diff_adata)
    sc.tl.leiden(top_diff_adata, resolution=res)
    return top_diff_adata

def top_var_condyn(cluster_diff_adata, est_adata, model, condition_key):
    max_vec = torch.tensor(est_adata.layers['lambda'].max(axis=0))
    jacx_c_sum = calc_jacc_sum(cluster_diff_adata, model, condition_key=condition_key) / max_vec.view(-1, 1)
    sq_jacx = jacx_c_sum.transpose(0, 1) @ jacx_c_sum
    ld, u = torch.eig(sq_jacx.detach(), eigenvectors=True)
    top_vec = u[:, 0].numpy()
    top_diffs = (jacx_c_sum @ u[:, 0]).detach().numpy()
    return top_vec, top_diffs

def make_idx_list(tot_num, unit_size=300):
    div_num = math.ceil(tot_num / unit_size)
    idx_list = np.array_split(np.arange(tot_num), div_num)
    return idx_list


@torch.no_grad()
def conduct_combined_svd(A, B):
    Ua, Sa, Va = torch.svd(A)
    Ub, Sb, Vb = torch.svd(B)
    C = Sa.unsqueeze(-1) * Va.transpose(-1, -2) @ Ub * Sb.unsqueeze(-2)
    Uc, Sc, Vc = torch.svd(C)
    Utot = Ua @ Uc
    Vtot = Vb @ Vc
    return Utot, Sc, Vtot



@torch.no_grad()
def calc_jac(adata, model, cidx=None, condition_key='coloc', batch_key='batch', calc_dvdd=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.tensor(adata.obsm['X_vicdyf_zl']).to(device).float()
    d = torch.tensor(adata.obsm['X_vicdyf_dl']).to(device).float()
    s = torch.tensor(np.array(adata.obsm[batch_key])).to(device).float()
    c = torch.tensor(np.array(adata.obsm[condition_key])).float().to(device)
    model.to(device)
    encd = model.enc_d
    enc_dmu = lambda z: encd.h2mu(encd.seq_nn(encd.x2h(z)))
    if calc_dvdd:
        dec_f = lambda vz, vs: model.dec_z(vz, vs)
        diff_dec_f = lambda vz, vd, vs: functorch.jvp(lambda vzz: dec_f(vzz, vs), (vz, ), (vd, ))[1]
        jac_v_d = lambda vz, vd, vs: functorch.jacrev(lambda vdd: diff_dec_f(vz, vdd, vs))(vd)
        jac_v_d_mat = functorch.vmap(jac_v_d, in_dims=(0, 0, 0))(z, d, s)
    else:
        jac_v_d_mat = None
    if cidx is None:
        fenc_d = lambda vz, vc: enc_dmu(torch.cat([vz, vc], dim=-1))
        jac_d_c = lambda vz, vc: functorch.jacrev(lambda vcc: fenc_d(vz, vcc))(vc)
        jac_d_c_mat = functorch.vmap(jac_d_c, in_dims=(0, 0))(z, c)
    else:
        cbegin, cend = cidx
        vbc, vtc, vac = c[:, :cbegin], c[:, cbegin:cend], c[:, cend:]
        fenc_d = lambda vz, vbc, vac, vtc: enc_dmu(torch.cat([vz, vbc, vtc, vac], dim=-1))
        jac_d_c = lambda vz, vbc, vac, vtc: functorch.jacrev(lambda vvtc: fenc_d(vz, vbc, vac, vvtc))(vtc)
        jac_d_c_mat = functorch.vmap(jac_d_c, in_dims=(0, 0, 0, 0))(z, vbc, vac, vtc)
    return jac_v_d_mat, jac_d_c_mat


def safe_numpy(m):
    return m.detach().cpu().numpy()


def trancate_ext(vals, q=0.99):
    vals[vals > np.quantile(vals, q)] = np.quantile(vals, q)
    vals[vals < np.quantile(vals, 1 - q)] = np.quantile(vals, 1 - q)
    return vals


def calc_total_sv_cdiff(adata, model, max_vec, condition_key='coloc', batch_key='batch', cidx=None):
    jac_v_d_mat, jac_d_c_mat = calc_jac(adata, model, cidx, condition_key, batch_key)
    max_vec = torch.tensor(max_vec).float().to(jac_v_d_mat.device)
    jac_v_d_mat = jac_v_d_mat / max_vec
    if 'high_qual' in adata.var.columns:
        u, s, v = conduct_combined_svd(jac_v_d_mat[:, adata.var.high_qual], jac_d_c_mat)
        adata.layers['top_cdiff_v'] = np.zeros(adata.shape)
        adata.layers['total_cdiff_v'] = np.zeros(adata.shape)
        adata.layers['top_cdiff_v'][:, adata.var.high_qual] = u[:, :, 0].detach().cpu().numpy()
        adata.layers['total_cdiff_v'][:, adata.var.high_qual] = torch.linalg.norm(u, dim=-1).detach().cpu().numpy()
    else:
        u, s, v = conduct_combined_svd(jac_v_d_mat, jac_d_c_mat)
        adata.layers['top_cdiff_v'] = u[:, :, 0].detach().cpu().numpy()
        adata.layers['total_cdiff_v'] = torch.linalg.norm(u, dim=-1).detach().cpu().numpy()
    adata.obs['top_cdiff_size'] = s[:, 0].detach().cpu().numpy()
    adata.obsm['top_cdiff_cond'] = v[:, :, 0].detach().cpu().numpy()
    adata.obs['total_cdiff_size'] = torch.linalg.norm(s, dim=-1).detach().cpu().numpy()
    adata.obsm['total_cdiff_cond'] = torch.linalg.norm(v, dim=-1).detach().cpu().numpy()
    return adata


def calc_cdiff_svd_aggr(adata, model, max_vec, cidx=None, condition_key='coloc', batch_key='batch'):
    max_vec = max_vec.reshape(-1, 1)
    cond_mat = np.array(adata.obsm[condition_key])
    if cidx != None:
        cond_mat = cond_mat[:, cidx[0]:cidx[1]]
    zl = adata.obsm['X_vicdyf_zl']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tot_jac_v_d_mat = torch.zeros(adata.shape[1], zl.shape[1]).to(device)
    tot_jac_d_c_mat = torch.zeros(zl.shape[1], cond_mat.shape[1]).to(device)
    idx_list = make_idx_list(adata.shape[0], 5)
    for idx in idx_list:
        jac_v_d_mat, jac_d_c_mat = calc_jac(adata[idx], model, cidx, condition_key, batch_key)
        tot_jac_v_d_mat += jac_v_d_mat.sum(dim=0)
        tot_jac_d_c_mat += jac_d_c_mat.sum(dim=0)
    max_vec = torch.tensor(max_vec).to(jac_v_d_mat.device)
    u, s, v = conduct_combined_svd(tot_jac_v_d_mat / max_vec, tot_jac_d_c_mat)
    return safe_numpy(u), safe_numpy(s), safe_numpy(v)


def extract_deg_df(adata, cluster):
    keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
    deg_df = pd.DataFrame({
        key: adata.uns['rank_genes_groups'][key][cluster]
        for key in keys
    }, index=adata.uns['rank_genes_groups']['names'][cluster])
    return deg_df


def get_top_degs(adata, cluster, min_logfc=1, min_pvals_adj=0.1):
    deg_df = extract_deg_df(adata, cluster)
    top_genes = deg_df.query('pvals_adj < 0.1 and logfoldchanges > @min_logfc').index
    return top_genes



def generate_cont_table_batch(top_genes, all_genes, gene_sets):
    S = np.array([
        np.isin(all_genes, gene_set).astype(int)
        for gene_set in gene_sets.values()
    ])
    t = np.isin(all_genes, top_genes).astype(int)
    tps = S @ t
    fns = t.sum() - tps
    fps = S.sum(axis=1) - tps
    tns = (all_genes.shape[0] - t.sum()) - fps
    cont_tables = np.array([[tps, fns], [fps, tns]]).transpose((2, 0, 1))
    return cont_tables

def gene_set_enrichment(top_genes, all_genes, gene_sets):
    cont_tables = generate_cont_table_batch(top_genes, all_genes, gene_sets)
    pvals = pd.Series([stats.fisher_exact(tb, alternative='greater')[1] for tb in cont_tables], index=gene_sets.keys())
    qvals = pd.Series(multipletests(pvals.values, method='fdr_bh')[1], index=gene_sets.keys())
    pval_df = pd.DataFrame({'pval': pvals, 'qval': qvals}, index=pvals.index).sort_values('pval')
    return pval_df


def make_cont_table(targets, alls, geneset):
    tp_genes = np.intersect1d(targets, geneset)
    tp = tp_genes.shape[0]
    fp = targets.shape[0] - tp
    fn = np.intersect1d(geneset, alls).shape[0] - tp
    tn = alls.shape[0] - fn
    tb = np.array([[tp, fp], [fn, tn]])
    return tb, tp_genes


def fisher_geneset(targets, alls, genset):
    tb, tp_genes = make_cont_table(targets, alls, genset)
    try:
        odds, pval = stats.fisher_exact(tb, alternative='greater')
    except:
        import pdb;pdb.set_trace()
    return pval, tp_genes

def parse_gmt(gmt_fname):
    gene_sets = {}
    with open(gmt_fname) as f:
        for l in f.read().splitlines():
            vals = l.split('\t')
            set_name = vals[0]
            genes = vals[2:]
            gene_sets[set_name] = genes
    return gene_sets


def def_dev():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def to_devtensor(arr, dev=None):
    if dev is None:
        dev = def_dev()
    return torch.tensor(arr).to(dev).float()


def cuda_svd(mat):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mat = torch.tensor(mat).to(device)
    res = torch.svd(mat)
    u, s, v = [safe_numpy(mat) for mat in res]
    return u, s, v
    

def calc_fluct_cov(adata):
    norm_vec = adata.layers['lambda'].mean(axis=0)
    diff_mat = (adata.layers['vicdyf_velocity'] - adata.layers['vicdyf_mean_velocity']) / norm_vec
    cov_mat = diff_mat.transpose() @ diff_mat
    return cov_mat / adata.shape[0]


def calc_stochastic_vel(adata, model, ds):
    zl = torch.tensor(adata.obsm['X_vicdyf_zl'])
    d = torch.tensor(adata.obsm['X_vicdyf_d'])
    batch = ds[:]
    gene_vel = model.calculate_diff_x_grad(zl, d, batch)
    return gene_vel

def safe_load_json(fname):
    try:
        cond = json.load(open(fname, mode='r'))
    except:
        cond = {}
    return cond


def calc_pq_vals_emp_bg(vals, backs):
    cdf_vals = np.array([np.mean(val > backs) for val in vals])
    pvals = np.row_stack([cdf_vals, 1 - cdf_vals]).min(axis=0) * 2
    qvals = np.array([float(q) for q in fdrcorrection(pvals)[1]])
    return pvals, qvals

def calc_pq_vals_gauss_bg(vals, backs):
    cdf_vals = stats.norm.cdf(vals, loc=backs.mean(), scale=backs.std())
    pvals = np.row_stack([cdf_vals, 1 - cdf_vals]).min(axis=0) * 2
    qvals = np.array([float(q) for q in fdrcorrection(pvals)[1]])
    return pvals, qvals


def calc_pq_vals_ligac(lt_df, cdiff_vals):
    import pdb;pdb.set_trace()
    pvals = np.array([stats.ttest_ind(cdiff_vals[lt_df[ligand] == 1], cdiff_vals[lt_df[ligand] == 1]) for ligand in lt_df.columns])
    qvals = np.array([float(q  for q in fdrcorrection(pvals)[1])])
    return pvals, qvals


def norm_mat(mat):
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def cossim(mat1, mat2):
    mat1 = norm_mat(mat1)
    mat2 = norm_mat(mat2)
    cossim = (mat1 * mat2).sum(axis=1)
    return cossim


def calculate_average_matrix(adata: anndata.AnnData, mat_key: str, category_key: str) -> dict:
    """
    Calculate average matrix in obs of an Anndata object based on category of a specific column in obs of the anndata.

    :param adata: AnnData object to use
    :type adata: anndata.AnnData
    :param category_key: the key of the category to use for grouping cells
    :param mat_key: the key of the matrix in obsm
    :type category_key: str
    :return: dictionary where keys are categories of observed variable and values are numpy arrays of means
    :rtype: dict
    """
    # Get unique categories
    categories = np.unique(adata.obs[category_key])
    
    # Initialize dictionary to store results
    result_dict = {}

    if mat_key in adata.obsm.keys():
        mat = adata.obsm[mat_key]
    else:
        mat = adata.layers[mat_key]
    # Calculate mean expression for each gene in each category
    for cat in categories:
        cat_cells = adata.obs[category_key] == cat
        cat_mean = np.mean(mat[cat_cells], axis=0)
        result_dict[cat] = cat_mean
    
    return result_dict

def take_difference2avgs(adata, mat_key, category_key, norm=False):
    # Get unique categories
    categories = np.unique(adata.obs[category_key])
    avg_dict = calculate_average_matrix(adata, mat_key, category_key)
    diff_dict = {
        category: avg_dict[category] - adata.obsm[mat_key]
        for category in categories
    }
    if norm:
        diff_dict = {
            category: diff_dict[category] / np.linalg.norm(diff_dict[category], axis=1, keepdims=True)
            for category in categories
        }
    return diff_dict

# calculate cosine similaryty between each row of two matrix.import numpy as np

def cosine_similarity(matrix_1, matrix_2):
    # Calculate dot product
    dot_product = np.einsum('ij, ij -> i', matrix_1, matrix_2)
    # Calculate magnitudes of each row in both matrices
    matrix_1_magnitudes = np.sqrt(np.sum(np.square(matrix_1), axis=1))
    matrix_2_magnitudes = np.sqrt(np.sum(np.square(matrix_2), axis=1))

    # Calculate denominator of cosine similarity formula
    denominator = matrix_1_magnitudes * matrix_2_magnitudes

    # Divide dot product by denominator element-wise, ignoring division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine_similarities = np.true_divide(dot_product, denominator)
        cosine_similarities[denominator == 0] = 0

    # Return cosine similarities as a dictionary
    return cosine_similarities

def align_df(corresp_df, df_list,  key_list):
    # subset included corresp_df entry
    corresp_df = corresp_df.loc[
        np.logical_and.reduce(
            [
                corresp_df[key].isin(df.columns) 
                    for df, key in zip(df_list, key_list)
            ]
        )
    ]
    res_df_list = []
    for df, key in zip(df_list, key_list):
        df = df[corresp_df[key]]
        df.columns = corresp_df.index
        res_df_list.append(df)
    return corresp_df, res_df_list

def calculate_similarity_celltypes(est_adata, cell_types, obs_key):
    est_adata.layers['scaled_lambda'] = stats.zscore(est_adata.layers['lambda'], axis=0)
    scale_exp_dict = {
        celltype: est_adata[est_adata.obs[obs_key] == celltype].layers['scaled_lambda'].mean(axis=0)
        for celltype in cell_types
    }
    for celltype in cell_types:
        est_adata.obs[f'similarity_{celltype}'] = est_adata.layers['scaled_lambda'] @ scale_exp_dict[celltype]
    return est_adata

def load_motif_df():
    motif_df = pd.read_csv('db/meta/TFFM_table.csv', index_col=0, header=0)
    motif_df['raw_motif'] = motif_df.matrix_base_id
    motif_df['motif'] = motif_df.matrix_base_id + '.' + motif_df.matrix_version.astype(str)
    motif_df.index = motif_df.motif
    motif_df['tf'] = motif_df.name
    return motif_df


def load_norm_lt_df(adata):
    lt_df = pd.read_csv('db/meta/lt_df_2022.csv', index_col=0)
    common_genes = np.intersect1d(lt_df.index.astype(str), adata.var_names)
    lt_df = lt_df.loc[common_genes, :]
    norm_lt_df = make_norm_lt_df(lt_df, q=0.99)
    common_genes = np.intersect1d(lt_df.index.astype(str), adata.var_names)
    return norm_lt_df, common_genes


def col_motif2tf(motif_act_df):
    motif_df = pd.read_table('db/jaspar/jaspar_motif_df.tsv', header=None, names=['motif_id', 'tf'], index_col=0)
    motif_act_df = motif_act_df.loc[:, motif_act_df.columns.isin(motif_df.index)]
    motif_act_df.columns = motif_df.loc[motif_act_df.columns, 'tf'].values
    return motif_act_df

def make_unique(orig_str_list):
    str_list = []
    uq_str_list = []
    for s in orig_str_list:
        if s in str_list:
            sub_idxs = (s == np.array(str_list)).sum()
            mod_s = s + '_' + str(sub_idxs)
        else:
            mod_s = s
        str_list.append(s)
        uq_str_list.append(mod_s)
    return uq_str_list


def exclude_small_clusters(adata, cluster_key, min_num=10):
    cluster_counts = adata.obs[cluster_key].value_counts()
    small_clusters = cluster_counts[cluster_counts < min_num].index
    adata = adata[~adata.obs[cluster_key].isin(small_clusters)]
    return adata




def get_top_lfc_genes(adata, cluster, n_genes=100, max_pval=0.01):
    deg_df = sc.get.rank_genes_groups_df(adata, cluster).query('pvals_adj < @max_pval and logfoldchanges > 0')
    top_genes = deg_df.sort_values('logfoldchanges', ascending=False).names[:n_genes].values
    return top_genes


def make_obs_consistent(adata1, adata2):
    common_cells = np.intersect1d(adata1.obs_names, adata2.obs_names)
    return adata1[common_cells], adata2[common_cells]


def separate_dataset(ds, test_ratio = 0.05, val_ratio = 0.1):
    total_size = len(ds)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.dataset.random_split(ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    return(train_ds, val_ds, test_ds)



def batch_in_device(batch, device):
    batch = [tensor.to(device) for tensor in batch]
    return batch

def calculate_neighbor_ratio(X, val_vec, nn=30):
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    val_mat = val_vec[indices]
    val_ratio_vec = val_mat.mean(axis=1)
    return (val_ratio_vec)
