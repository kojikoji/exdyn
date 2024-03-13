from torch.utils.data import DataLoader, TensorDataset
from . import visualization
from torch.utils.data import DataLoader
from . import dataset
import torch
from scipy import stats
import scanpy as sc
from . import commons, workflow, jac_fluct
import pandas as pd
import numpy as np

def make_diff_mat(cluster_adata, tot_adata, max_vec=None):
    if max_vec is None:
        max_vec = tot_adata.layers['lambda'].max(axis=0)
    norm_ld_mat = tot_adata.layers['lambda'] / max_vec
    mean_lds = (cluster_adata.layers['lambda'] / max_vec).mean(axis=0)
    diff_mat = (norm_ld_mat - mean_lds) / np.linalg.norm(norm_ld_mat - mean_lds, axis=1, keepdims=True)
    return diff_mat


def calculate_cdiff_target_scores(cluster_adata, tot_adata, cdiff_key='norm_cond_vel_diff'):
    max_vec = tot_adata.layers['lambda'].max(axis=0)
    cdiff_vec = cluster_adata.layers[cdiff_key].mean(axis=0) 
    cdiff_vec = cdiff_vec / np.linalg.norm(cdiff_vec)
    diff_mat = make_diff_mat(cluster_adata, tot_adata)
    cdiff_target_scores = pd.Series(diff_mat @ cdiff_vec, index=tot_adata.obs_names)
    return cdiff_target_scores
    

def make_deg_cdiff_df(source_cells, jac_adata, cluster, target_scores, gene_scores, q=0.8):
    top_diff_cells = target_scores.index[target_scores > target_scores.quantile(q)]
    jac_adata.obs['diff_pop'] = 'None'
    jac_adata.obs['diff_pop'][source_cells] = 'Source'
    jac_adata.obs['diff_pop'][top_diff_cells] = 'Target'
    sc.tl.rank_genes_groups(jac_adata, 'diff_pop', groups=['Target'], reference='Source', method='wilcoxon')
    deg_df = commons.extract_deg_df(jac_adata, 'Target')
    deg_df['score'] = gene_scores[deg_df.index]
    return deg_df


def make_specifc_cond_tensor(adata, cond, condition_key='condition'):
    c = torch.zeros_like(torch.tensor(adata.obsm[condition_key].values)).float()
    c[:, adata.obsm[condition_key].columns == str(cond)] = 1.0
    return c


@torch.no_grad()
def estimate_cond_dynamics(adata, cond, model, condition_key='condition', batch_key='sample', cl_key='sample', cond_in_obsm=False, onehot_key=None, model_type='cvicdyf'):
    orig_device = model.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    s, u, snorm_mat, unorm_mat, b, t, adata = workflow.make_datasets(adata, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
    if onehot_key is None:
        one_t = make_specifc_cond_tensor(adata, cond, condition_key=condition_key).to(device)
    else:
        one_t = make_specifc_cond_tensor(adata, cond, condition_key=onehot_key).to(device)
        t = t.to(device)
        one_t = torch.cat([one_t, t[:, one_t.size()[1]:]], dim=1)
    if model_type == 'cvicdyf':
        ds = dataset.CvicDyfDataSet(s, u, snorm_mat, unorm_mat, b, one_t)
    elif model_type == 'cvicdyfmk':
        c, adata = commons.make_sample_one_hot_mat(adata, sample_key=cl_key)
        ds = TensorDataset(s, u, snorm_mat, unorm_mat, b, one_t, c)
    loader = DataLoader(ds, batch_size=128)
    gene_vel_list = []
    gene_vscale_list = []
    d_loc_list = []
    d_scale_list = []
    for batch in loader:
        batch = commons.batch_in_device(batch, device)
        z, qz, d, qd, px_z_ld, pu_zd_ld, rgene_vel, gene_vel, gene_vel_std = model.mean_forward_batch(batch)
        gene_vel_list.append(gene_vel.cpu().numpy())
        gene_vscale_list.append(gene_vel_std.cpu().numpy())
        d_loc_list.append(qd.loc.cpu().numpy())
        d_scale_list.append(qd.scale.cpu().numpy())
    gene_vel = np.concatenate(gene_vel_list, axis=0)
    gene_vscale = np.concatenate(gene_vscale_list, axis=0)
    d_loc = np.concatenate(d_loc_list, axis=0)
    d_scale = np.concatenate(d_scale_list, axis=0)
    model.to(orig_device)
    return gene_vel, gene_vscale, d_loc, d_scale


@torch.no_grad()
def estimate_stochastic_condiff(adata, cond1, cond2, model, condition_key='condition', batch_key='sample', cond_in_obsm=False):
    orig_device = model.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    b = torch.tensor(adata.obsm['batch'].values).float()
    t = torch.tensor(adata.obsm['condition'].values).float()
    s = torch.tensor(adata.layers['spliced'].toarray())
    u = torch.tensor(adata.layers['unspliced'].toarray())
    snorm_mat = s
    unorm_mat = u
    t1 = make_specifc_cond_tensor(adata, cond1).to(device)
    t2 = make_specifc_cond_tensor(adata, cond2).to(device)
    batch1 = (s.to(device), u.to(device), snorm_mat.to(device), unorm_mat.to(device), b.to(device), t1.to(device))
    batch2 = (s.to(device), u.to(device), snorm_mat.to(device), unorm_mat.to(device), b.to(device), t2.to(device))
    z, qz = model.encode_z(batch1)
    d1, qd1 = model.encode_d(z, batch1)
    d2, qd2 = model.encode_d(z, batch2)
    v1 = model.calculate_diff_x_grad(z, d1, batch1)
    v2 = model.calculate_diff_x_grad(z, d2, batch1)
    diff_v = v2 - v1
    model.to(orig_device)
    return diff_v.detach().cpu().numpy()


def calculate_cdiff_bf(adata, cond1, cond2, model, n=10, condition_key='condition', batch_key='sample', cond_in_obsm=False, cellwise=False):
    if cellwise:
        aggregate = np.stack
    else:
        aggregate = np.concatenate
    diff_v = aggregate([
        estimate_stochastic_condiff(adata, cond1, cond2, model, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
        for _ in range(n)], axis=0)
    eps = 1.0e-16
    p = (diff_v > 0).mean(axis=0) + 1.0e-16
    bfs = np.log2((p + eps) / (1 - p + eps))
    if not cellwise:
        bfs = pd.Series(bfs, index=adata.var_names)
    return bfs


def estimate_multi_cond_dynamics(adata, cond_list, lit_envdyn, condition_key='condition', batch_key='sample', cond_in_obsm=False, onehot_key=None, cl_key='sample', model_type='cvicdyf', embed_dyn=True):
    for cond in cond_list:
        cond_vel, cond_vscale, cond_d, connd_dscale = estimate_cond_dynamics(adata, cond, lit_envdyn, condition_key, batch_key, cl_key=cl_key, cond_in_obsm=cond_in_obsm, onehot_key=onehot_key, model_type=model_type)
        adata.obsm[f'cond_d_{cond}'] = cond_d
        adata.obsm[f'cond_dscale_{cond}'] = connd_dscale
        adata.layers[f'cond_vel_{cond}'] = cond_vel
        adata.layers[f'cond_vscale_{cond}'] = cond_vscale
        if embed_dyn:
            cond_dumap = commons.calc_int_dembed(adata.obsm['X_vicdyf_umap'], adata.obsm['X_vicdyf_zl'], cond_d, 1)
            adata.obsm[f'cond_dumap_{cond}'] = cond_dumap
    mean_vec = np.mean(adata.layers['lambda'], axis=0)
    adata = norm_multicond(adata, cond_list, mean_vec)
    return adata

def estimate_two_cond_dynamics(adata, cond1, cond2, lit_envdyn, **kwargs):
    sub_adata = estimate_multi_cond_dynamics(adata, [cond1, cond2], lit_envdyn, **kwargs)
    sub_adata.layers['cond_vel_diff'] = sub_adata.layers[f'cond_vel_{cond2}'] -  sub_adata.layers[f'cond_vel_{cond1}']
    max_vec = np.max(sub_adata.layers['lambda'], axis=0)
    layer = 'cond_vel_diff'
    adata.layers[f'norm_{layer}'] = adata.layers[layer] / max_vec
    sub_adata.obsm['cond_d_diff'] = sub_adata.obsm[f'cond_d_{cond2}'] - sub_adata.obsm[f'cond_d_{cond1}']
    sub_adata.obsm['cond_dumap_diff'] = sub_adata.obsm[f'cond_dumap_{cond2}'] - sub_adata.obsm[f'cond_dumap_{cond1}']
    return sub_adata


def norm_multicond(adata, cond_list, norm_vec):
    layers = [f'cond_vel_{cond}' for cond in cond_list]
    for layer in layers:
        adata.layers[f'norm_{layer}'] = adata.layers[layer] / norm_vec
    return adata

def norm_condiff(adata, cond1, cond2, norm_vec):
    layers = [f'cond_vel_{cond}' for cond in [cond1, cond2]] + ['cond_vel_diff']
    for layer in layers:
        adata.layers[f'norm_{layer}'] = adata.layers[layer] / norm_vec
    return adata

def condiff_clustering(adata, q, res):
    top_adata = adata[adata.obs.total_condiff > adata.obs.total_condiff.quantile(q)]
    sc.pp.neighbors(top_adata, use_rep='X_vicdyf_zl')
    sc.tl.leiden(top_adata, key_added='cdiff_cluster', resolution=res)
    return top_adata


def visualize_population_dyndiff(adata, cells, conds, ax, scale=5, width=0.0125):
    color_dict = {conds[0]: '#1F72AA', conds[1]: '#EF7D21'}
    top_adata = adata[cells]
    u_mean = top_adata.obsm['X_umap'].mean(axis=0)
    du_mean_dict = {}
    for cond in conds:
        du_mean_dict[cond] = top_adata.obsm[f'cond_dumap_{cond}'].mean(axis=0)
    for cond in conds:
        du_mean = du_mean_dict[cond]
        color = color_dict[cond]
        ax.quiver(np.array(u_mean[0]), np.array(u_mean[1]), np.array(du_mean[0]), np.array(du_mean[1]), color=color, label=cond, scale=scale, width=width, alpha=0.5)


def calc_cdiff_prob_layer(est_adata, model, cond1, cond2, unit_size=5):
    max_vec = est_adata.layers['lambda'].max(axis=0)
    idx_list = commons.make_idx_list(est_adata.shape[0], unit_size=unit_size)
    conds = [cond1, cond2]
    for cond in conds:
        jac_loc_list = []
        jac_std_list = []
        for idxs in idx_list:
            sub_adata = est_adata[idxs]
            jac_loc, jac_std = jac_fluct.calc_jac_mean_scale(sub_adata.obsm['X_vicdyf_zl'], sub_adata.obsm[f'cond_d_{cond}'], sub_adata.obsm[f'cond_dscale_{cond}'], model, max_vec, s=sub_adata.obsm['batch'].values)
            jac_loc_list.append(jac_loc)
            jac_std_list.append(jac_std)
        int_jac_loc = np.concatenate(jac_loc_list, axis=0)
        int_jac_std = np.concatenate(jac_std_list, axis=0)
        est_adata.layers[f'{cond}_jac_loc'] = int_jac_loc
        est_adata.layers[f'{cond}_jac_std'] = int_jac_std
    est_adata.layers['cdiff_jac_loc'] = est_adata.layers[f'{cond2}_jac_loc'] - est_adata.layers[f'{cond1}_jac_loc']
    est_adata.layers['cdiff_jac_std'] = np.sqrt(est_adata.layers[f'{cond2}_jac_std']**2 + est_adata.layers[f'{cond1}_jac_std'] **2)
    est_adata.layers['cdiff_prob'] = 1 - stats.norm.cdf(0, loc=est_adata.layers['cdiff_jac_loc'], scale=est_adata.layers['cdiff_jac_std'])
    return est_adata

def decomp_dd_dc(est_adata, model, condition_key, batch_key, cidx=None):
    _, dd_dc = commons.calc_jac(est_adata, model, condition_key=condition_key, batch_key=batch_key, calc_dvdd=False, cidx=cidx)
    res = torch.svd(dd_dc.view(-1, dd_dc.size()[-1]))
    u, s, v = [commons.safe_numpy(mat) for mat in res]
    u = u.reshape(est_adata.shape[0], -1, v.shape[1])
    return u, s, v


def make_embed_dd_dc(est_adata, top_modes, u, s, v):
    z_mat = est_adata.obsm['X_vicdyf_zl']
    u_mat = est_adata.obsm['X_vicdyf_umap']
    est_adata.uns[f'dd_dc_s'] = s[:top_modes]
    est_adata.uns[f'dd_dc_v'] = v
    est_adata.obs['dd_dc_strength'] = np.linalg.norm((u * s).reshape(est_adata.shape[0], -1), axis=1)
    for i in range(top_modes):
        est_adata.obsm[f'dd_dc_u{i}'] = u[:, :, i]
        d_mat = est_adata.obsm[f'dd_dc_u{i}']
        est_adata.obsm[f'dd_dc_embed{i}'] = commons.calc_d_embed_grad(z_mat, d_mat, u_mat, nn=30)[1]
        est_adata.obs[f'dd_dc_u_st{i}'] = np.linalg.norm(d_mat, axis=1)
    return est_adata

def make_dd_dc_adata(est_adata, model, condition_key, batch_key, cidx=None, top_modes=10):
    u, s, v = decomp_dd_dc(est_adata, model, condition_key, batch_key, cidx=cidx)
    est_adata = make_embed_dd_dc(est_adata, top_modes, u, s, v)
    return est_adata

def calculate_lineage_bias_scores(adata, category_key, diff_mat_key, mat_key='X_vicdyf_zl'):
    # calculate average profile for each categories
    diff_dict = commons.take_difference2avgs(adata, mat_key, category_key)
    # cossim dict
    cossim_dict = {
        category: commons.cosine_similarity(diff_dict[category], adata.obsm[diff_mat_key])
        for category in diff_dict.keys()
    }
    return pd.DataFrame(cossim_dict, index=adata.obs_names)