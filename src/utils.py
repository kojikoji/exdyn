import os
import sys
import pandas as pd
import numpy as np
import umap
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
import torch.distributions as dist
import dataset, envdyn
import importlib
import torch
import torch
from sklearn.neighbors import NearestNeighbors



def fix_key(key):
    val_list = key.split('.')
    if len(val_list) == 1:
        return(val_list[0])
    else:
        val_list.insert(1, 'module')
        return('.'.join(val_list))

def load_old_existing_model(envdyn_exp, model_param_path):
    state_dict = torch.load(model_param_path, map_location=envdyn_exp.device)    
    state_dict = {
        fix_key(key): state_dict[key]
        for key in state_dict.keys()}
    envdyn_exp.model.load_state_dict(state_dict)
    torch.no_grad()
    envdyn_exp.model.eval()
    return(envdyn_exp)
    

def colwise_speamanr(r_c, r_ld):
    val = np.array([scipy.stats.spearmanr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    return(val)


def colwise_pearsonr(r_c, r_ld):
    val = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    return(val)


def calc_corr_df(envdyn_exp, ds=None):
    if ds != None:
        size = len(ds)
        loader = torch.utils.data.DataLoader(dataset=ds, batch_size=size)        
        s, u, snorm_mat, unorm_mat = next(iter(loader))
    else:
        s = envdyn_exp.edm.test_s
        u = envdyn_exp.edm.test_u
        norm_mat = envdyn_exp.edm.test_norm_mat.to(envdyn_exp.device)
    s = s.to(envdyn_exp.device)
    u = u.to(envdyn_exp.device)
    x = s
    z, d, qz, qd, px_z_ld, pu_zd_ld, sl, qsl, ul, qul, dyn_loss = envdyn_exp.model(x, u)
    norm_s = x.float() / (x.mean(dim=1, keepdim=True))
    norm_u = u.float() / (u.mean(dim=1, keepdim=True))
    s_corr = colwise_pearsonr(norm_s.cpu().detach().numpy(), px_z_ld.cpu().detach().numpy())
    u_corr = colwise_pearsonr(norm_u.cpu().detach().numpy(), pu_zd_ld.cpu().detach().numpy())
    r_c = (norm_u / (norm_s + 1)).cpu().detach().numpy()
    r_ld = (pu_zd_ld / (px_z_ld + 1)).cpu().detach().numpy()
    r_corr = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    s_mat = x.float().cpu().detach().numpy()
    u_mat = u.float().cpu().detach().numpy()
    r_c = (norm_u / (norm_s + 1)).cpu().detach().numpy()
    s_mean = np.mean(s_mat, axis=0)
    u_mean = np.mean(u_mat, axis=0)
    s_var = np.std(s_mat, axis=0) / s_mean
    u_var = np.std(u_mat, axis=0) / u_mean
    r_var = np.std(r_c, axis=0)
    stats_df = pd.DataFrame({'s_mean': s_mean, 's_var': s_var, 'u_mean': u_mean, 'u_var': u_var, 'r_var': r_var, 's_corr': s_corr, 'u_corr': u_corr, 'r_corr': r_corr})
    return(stats_df)


def calc_valid_corr_df(envdyn_exp):
    s = envdyn_exp.edm.validation_s
    u = envdyn_exp.edm.validation_u
    norm_mat = envdyn_exp.edm.validation_norm_mat.to(envdyn_exp.device)
    s = s.to(envdyn_exp.device)
    u = u.to(envdyn_exp.device)
    x = s
    z, d, qz, qd, px_z_ld, pu_zd_ld, sl, qsl, ul, qul = envdyn_exp.model(x, u)
    norm_s = x.float() / norm_mat
    norm_u = u.float() / norm_mat
    s_corr = colwise_pearsonr(norm_s.cpu().detach().numpy(), px_z_ld.cpu().detach().numpy())
    u_corr = colwise_pearsonr(norm_u.cpu().detach().numpy(), pu_zd_ld.cpu().detach().numpy())
    r_c = (norm_u / (norm_s + 1)).cpu().detach().numpy()
    r_ld = (pu_zd_ld / (px_z_ld + 1)).cpu().detach().numpy()
    r_corr = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    s_mat = x.float().cpu().detach().numpy()
    u_mat = u.float().cpu().detach().numpy()
    r_c = (norm_u / (norm_s + 1)).cpu().detach().numpy()
    s_mean = np.mean(s_mat, axis=0)
    u_mean = np.mean(u_mat, axis=0)
    s_var = np.std(s_mat, axis=0) / s_mean
    u_var = np.std(u_mat, axis=0) / u_mean
    r_var = np.std(r_c, axis=0)
    stats_df = pd.DataFrame({'s_mean': s_mean, 's_var': s_var, 'u_mean': u_mean, 'u_var': u_var, 'r_var': r_var, 's_corr': s_corr, 'u_corr': u_corr, 'r_corr': r_corr})
    return(stats_df)


def plot_correlations(envdyn_exp, fig_dir):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    s = envdyn_exp.edm.test_s
    u = envdyn_exp.edm.test_u
    s = s.to(envdyn_exp.device)
    u = u.to(envdyn_exp.device)
    x = s
    norm_mat = torch.sum(x, dim=1).view(-1, 1) * torch.sum(x, dim=0).view(1, -1)
    norm_mat = norm_mat / torch.mean(norm_mat)
    z, d, qz, qd, px_z_ld, pu_zd_ld = envdyn_exp.model(x, u)
    norm_s = x.float() / norm_mat
    norm_u = u.float() / norm_mat
    s_corr = colwise_speamanr(norm_s.cpu().detach().numpy(), px_z_ld.cpu().detach().numpy())
    u_corr = colwise_speamanr(norm_u.cpu().detach().numpy(), pu_zd_ld.cpu().detach().numpy())
    r_c = (norm_u / (norm_s + 1)).cpu().detach().numpy()
    r_ld = (pu_zd_ld / (px_z_ld + 1)).cpu().detach().numpy()
    r_corr = np.array([scipy.stats.spearmanr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    fig_path = fig_dir + 's_corr.png'
    fig, axes = plt.subplots(1, 1, figsize=(6, 5 * 1))
    axes.hist(s_corr)
    plt.savefig(fig_path)
    plt.close()
    fig_path = fig_dir + 'u_corr.png'
    fig, axes = plt.subplots(1, 1, figsize=(6, 5 * 1))
    axes.hist(u_corr)
    plt.savefig(fig_path)
    plt.close()
    fig_path = fig_dir + 'r_corr.png'
    fig, axes = plt.subplots(1, 1, figsize=(6, 5 * 1))
    axes.hist(r_corr)
    plt.savefig(fig_path)
    plt.close()
    s_mat = x.float().cpu().detach().numpy()
    u_mat = u.float().cpu().detach().numpy()
    norm_s = x.float() / norm_mat
    norm_u = u.float() / norm_mat
    r_c = (norm_u / (norm_s + 1)).cpu().detach().numpy()

    s_mean = np.mean(s_mat, axis=0)
    u_mean = np.mean(u_mat, axis=0)
    s_var = np.std(s_mat, axis=0) / s_mean
    u_var = np.std(u_mat, axis=0) / u_mean
    r_var = np.std(r_c, axis=0)
    stats_df = pd.DataFrame({'s_mean': s_mean, 's_var': s_var, 'u_mean': u_mean, 'u_var': u_var, 'r_var': r_var, 's_corr': s_corr, 'u_corr': u_corr, 'r_corr': r_corr})
    fig_path = fig_dir + 's_mean_var_corr.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x='s_mean', y='s_var', hue='s_corr', data=stats_df)
    plt.savefig(fig_path);plt.close('all')

    fig_path = fig_dir + 'u_mean_var_corr.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x='u_mean', y='u_var', hue='u_corr', data=stats_df)
    plt.savefig(fig_path);plt.close('all')

    fig_path = fig_dir + 'r_mean_var_corr.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x='s_mean', y='r_var', hue='r_corr', data=stats_df)
    plt.savefig(fig_path);plt.close('all')


def embed_zd(z_mat, d_mat, gene_norm, n_neighbors=30, min_dist=0.3):
    zd_mat = z_mat + d_mat
    int_z = np.concatenate([z_mat, zd_mat])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(int_z)
    z_embed = embedding[:z_mat.shape[0]]
    zd_embed = embedding[z_mat.shape[0]:]
    d_embed = (zd_embed - z_embed)
    d_embed_norm = np.linalg.norm(d_embed, axis=1)
    d_norm = np.linalg.norm(d_mat, axis=1)
    d_embed = (gene_norm / d_embed_norm).reshape((-1, 1)) * d_embed
    return(z_embed, d_embed, zd_embed)
    

def embed_z(z_mat, n_neighbors=30, min_dist=0.3):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    z_embed = reducer.fit_transform(z_mat)
    return(z_embed)
    


def plot_embed(envdyn_exp, fig_dir, n_neighbors=30, min_dist=0.3, d_coeff=30, cluster_lab='end', plot_num=500):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = envdyn_exp.edm.s.to(envdyn_exp.device)
    u = envdyn_exp.edm.u.to(envdyn_exp.device)
    end_celltypes = envdyn_exp.edm.meta_df[cluster_lab].astype(str)
    z, d, qz, qd, px_z_ld, pu_zd_ld = envdyn_exp.model(x, u)
    zl = qz.loc
    d, qd = envdyn_exp.model.enc_d(zl)
    d = envdyn_exp.model.d_coeff * qd.rsample()
    model_d_coeff = envdyn_exp.model.d_coeff
    px_z_ld = envdyn_exp.model.dec_z(zl)
    # calc gene velocity
    pxd_zd_ld = envdyn_exp.model.dec_z(z + d)
    gene_vel_mat = (pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    gene_norm = np.linalg.norm(gene_vel_mat, axis=1)
    # calc gene sd
    d_sd = (model_d_coeff * qd.scale)
    pxvp_zd_ld = envdyn_exp.model.dec_z(z + d_sd)
    pxvm_zd_ld = envdyn_exp.model.dec_z(z - d_sd)
    gene_vel_mat = (pxvp_zd_ld - pxvm_zd_ld).cpu().detach().numpy()
    gene_sd = np.linalg.norm(gene_vel_mat, axis=1)
    # matrix 
    d_mat = d.cpu().detach().numpy() * d_coeff
    z_mat = z.cpu().detach().numpy()
    zl_mat = qz.loc.cpu().detach().numpy()
    dl_mat = qd.loc.cpu().detach().numpy() * model_d_coeff * d_coeff
    # index for plot
    plot_idx = np.random.choice(np.arange(zl_mat.shape[0]), plot_num, replace=True)

    z_embed, d_embed, zd_embed = embed_zd(zl_mat, d_mat, gene_norm, n_neighbors=n_neighbors, min_dist=min_dist)
    z_embed, d_embed, zd_embed = z_embed[plot_idx], d_embed[plot_idx], zd_embed[plot_idx]
    embed_fname = fig_dir + 'embedding_stochastic.png'
    fig, axes = plt.subplots(1, 1, figsize=(7, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=end_celltypes[plot_idx])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.quiver(z_embed[:, 0], z_embed[:, 1], d_embed[:, 0], d_embed[:, 1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.4, hspace=0.8)
    plt.savefig(embed_fname);plt.close('all')

    embed_bf_fname = fig_dir + 'embedding_before_after.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    plt.scatter(x=z_embed[:, 0], y=z_embed[:, 1])
    plt.scatter(x=zd_embed[:, 0], y=zd_embed[:, 1], marker='x')
    plt.savefig(embed_bf_fname);plt.close('all')

    embed_fname = fig_dir + 'embedding_speed.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=gene_norm[plot_idx])
    plt.savefig(embed_fname);plt.close('all')

    embed_fname = fig_dir + 'embedding_fluctuation.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=gene_sd[plot_idx])
    plt.savefig(embed_fname);plt.close('all')    
    
    zl_embed, dl_embed, zdl_embed = embed_zd(zl_mat, dl_mat, gene_norm, n_neighbors=n_neighbors, min_dist=min_dist)
    zl_embed, dl_embed, zdl_embed = zl_embed[plot_idx], dl_embed[plot_idx], zdl_embed[plot_idx]
    vembed_fname = fig_dir + 'embedding_mean.png'
    fig, axes = plt.subplots(1, 1, figsize=(7, 5 * 1))
    sns.scatterplot(x=zl_embed[:, 0], y=zl_embed[:, 1], hue=end_celltypes[plot_idx])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.quiver(zl_embed[:, 0], zl_embed[:, 1], dl_embed[:, 0], dl_embed[:, 1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.4, hspace=0.8)
    plt.savefig(vembed_fname);plt.close('all')

    vembed_bf_fname = fig_dir + 'embedding_mean_before_after.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    plt.scatter(x=zl_embed[:, 0], y=zl_embed[:, 1])
    plt.scatter(x=zdl_embed[:, 0], y=zdl_embed[:, 1], marker='x')
    plt.savefig(vembed_bf_fname);plt.close('all')



def plot_latent_space(envdyn_exp, dim1, dim2, fig_name, cluster_lab='end'):
    data = envdyn_exp.edm.s
    x = envdyn_exp.edm.s.to(envdyn_exp.device)
    u = envdyn_exp.edm.u.to(envdyn_exp.device)
    end_celltypes = envdyn_exp.edm.meta_df[cluster_lab].astype(str)
    z, d, qz, qd, px_z_ld, pu_zd_ld = envdyn_exp.model(x)
    z_mat = z.cpu().detach().numpy()
    dl_mat = qd.loc.cpu().detach().numpy()
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_mat[:, dim1], y=z_mat[:, dim2], hue=end_celltypes)
    plt.quiver(z_mat[:, dim1], z_mat[:, dim2], dl_mat[:, dim1], dl_mat[:, dim2])
    plt.savefig(fig_name);plt.close('all')
    

def mask_neighbors(z, nn=15):
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(z.cpu().detach().numpy())
    distances, indices = nbrs.kneighbors(z.cpu().detach().numpy())
    mask = torch.zeros(z.size()[0], z.size()[0])
    mask[indices] = 1
    return(mask)


def transition_dest(z_mat, d_mat, nn=30):
    cell_num = z_mat.shape[0]
    z = torch.tensor(z_mat)
    d = torch.tensor(d_mat)
    nbrs = NearestNeighbors(n_neighbors=nn + 1).fit(z_mat)
    distances, indices =  nbrs.kneighbors(z_mat)
    sources = np.repeat(np.arange(cell_num), nn)
    targets = indices[:, 1:].reshape(-1)
    idx = np.vstack([sources, targets])
    source_z = torch.sparse_coo_tensor(idx, z[sources], (cell_num, cell_num, 10))
    target_z = torch.sparse_coo_tensor(idx, z[targets], (cell_num, cell_num, 10))
    source_d = torch.sparse_coo_tensor(idx, d[sources], (cell_num, cell_num, 10))
    exp_d = target_z - source_z
    torch.sparse.sum(exp_d**2, dim=2).sqrt()
    norm = (torch.sparse.sum(exp_d**2, dim=2).sqrt() * torch.sparse.sum(source_d**2, dim=2).sqrt()).to_dense()
    raw_sim = torch.sparse.sum((source_d * exp_d), dim=2).to_dense()
    raw_sim[raw_sim!=0] /= norm[raw_sim!=0]
    return(raw_sim)



def calc_tr_mat(z, d, sigma, nn=30):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    mask_mat = mask_neighbors(z, nn=nn)
    zz_diff_mat = z.view(1, cell_num, gene_num) - z.view(cell_num, 1, gene_num)
    norm_fix_mat = (zz_diff_mat == 0).type(torch.DoubleTensor)
    zz_diff_mat /= torch.norm(zz_diff_mat, dim=2, p=2, keepdim=True) + norm_fix_mat
    d = d.view(cell_num, 1, gene_num)
    d /= torch.norm(d, dim=2, p=2, keepdim=True)
    cos_sim = torch.sum(d * zz_diff_mat, dim=2)
    cos_sim = cos_sim * mask_mat
    tr_mat = torch.exp(cos_sim / sigma)
    tr_mat /= torch.sum(tr_mat, axis=1).view(-1, 1)
    tr_mat = (tr_mat - 1/cell_num)
    return(tr_mat)

# def calc_tr_mat(z, d, sigma, nn=30):
#     cell_num = z.shape[0]
#     gene_num = z.shape[1]
#     cos_sim = transition_dest(z, d, nn=nn)
#     tr_mat = torch.exp(cos_sim / sigma)
#     tr_mat /= torch.sum(tr_mat, axis=1).view(-1, 1)
#     tr_mat = (tr_mat - 1/cell_num)
#     return(tr_mat)

def calc_short_tr_mat(z, d, sigma, nn_prop=0.1, d_coeff=10):
    from sklearn.neighbors import NearestNeighbors
    nn = int(nn_prop * z.shape[0])
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(z)
    zz_distances, zz_indices = nbrs.kneighbors(z)
    norm_d = d / torch.norm(d, dim=1, p=2, keepdim=True)
    d_length = torch.tensor(zz_distances[:, nn -1]).view(-1, 1)
    # corr_d = d_length * norm_d
    corr_d = d * d_coeff
    zd = z + corr_d
    zdz_distances, zdz_indices = nbrs.kneighbors(zd)
    # exclude min zdz_distances > 0.1
    exclude_idx = (torch.tensor(np.min(zdz_distances, axis=1)).view(-1, 1) > d_length ).view(-1)
    r = torch.mean(torch.tensor(zdz_distances), dim=1)
    raw_prob = torch.exp(-(torch.tensor(zdz_distances) / torch.tensor(r).view(-1, 1)).pow(2))
    prob = raw_prob / torch.sum(raw_prob, dim=1, keepdim=True)
    # prob[exclude_idx] = 0
    tr_mat = torch.zeros(z.shape[0], z.shape[0])
    for i, (p, idx) in enumerate(zip(prob, zdz_indices)):
        tr_mat[i, idx] = p.float()
    return(tr_mat)

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
    
def calc_gene_sd(z, qd, dcoeff, model, num=50):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    zd_batch = z.view(1, cell_num, gene_num) + dcoeff * qd.sample((num,))
    px_ld_batch = model.dec_z(zd_batch)
    batch_std_mat = torch.std(px_ld_batch, dim=0)
    gene_sd = torch.norm(batch_std_mat, dim=1).cpu().detach().numpy()
    return(gene_sd, batch_std_mat)
    

def calc_gene_mean_sd(z, qd, dcoeff, model, num=50):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    zd_batch = z.view(1, cell_num, gene_num) + dcoeff * qd.sample((num,))
    px_ld_batch = model.dec_z(zd_batch) - model.dec_z(z).view(1, cell_num, -1)
    batch_std_mat = torch.std(px_ld_batch, dim=0)
    batch_mean_mat = torch.mean(px_ld_batch, dim=0)
    gene_sd = torch.norm(batch_std_mat, dim=1).cpu().detach().numpy()
    gene_mean = torch.norm(batch_mean_mat, dim=1).cpu().detach().numpy()
    return(gene_mean, gene_sd, batch_mean_mat, batch_std_mat)
    

def plot_trans_embed(envdyn_exp, fig_dir, data_dir, n_neighbors=15, min_dist=0.01, cluster_lab='end', plot_num=500, sigma=0.05, embed_gene_vel=False):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = envdyn_exp.edm.s.to(envdyn_exp.device)
    u = envdyn_exp.edm.u.to(envdyn_exp.device)
    end_celltypes = envdyn_exp.edm.meta_df[cluster_lab].astype(str)
    z, d, qz, qd, px_z_ld, pu_zd_ld = envdyn_exp.model(x)
    zl = qz.loc
    qd_mu, qd_logvar = envdyn_exp.model.enc_d(zl)
    qd = dist.Normal(qd_mu, envdyn_exp.model.softplus(qd_logvar))
    d = envdyn_exp.model.d_coeff * qd.rsample()
    px_z_ld = envdyn_exp.model.dec_z(zl)
    model_d_coeff = envdyn_exp.model.d_coeff
    dl = qd.loc
    dsd = qd.scale
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    model_d_coeff = envdyn_exp.model.d_coeff
    print('Extract info')
    # calc gene velocity
    pxd_zd_ld = envdyn_exp.model.dec_z(zl + d)
    gene_vel = (pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    mean_pxd_zd_ld = envdyn_exp.model.dec_z(zl + dl * model_d_coeff)
    mean_gene_vel = (mean_pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    gene_norm = np.linalg.norm(gene_vel, axis=1)
    print('before sd2')
    gene_sd2, batch_std_mat = calc_gene_sd(z, qd, model_d_coeff, envdyn_exp.model)
    batch_std_mat = batch_std_mat.cpu().detach().numpy()
    print('after sd2')
    # matrix 
    d_mat = d.cpu().detach().numpy()
    z_mat = z.cpu().detach().numpy()
    zl_mat = qz.loc.cpu().detach().numpy()
    dl_mat = qd.loc.cpu().detach().numpy() * model_d_coeff
    ld_mat = px_z_ld.cpu().detach().numpy()
    print('make mat')
    # calculate transition rate
    stoc_tr_mat = calc_tr_mat(zl.cpu().detach(), d.cpu().detach() / model_d_coeff, sigma)
    mean_tr_mat = calc_tr_mat(zl.cpu().detach(), dl.cpu().detach(), sigma)
    print('calc transition')
    # embed z
    z_embed = embed_z(zl_mat, n_neighbors=n_neighbors, min_dist=min_dist)
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, gene_norm)
    mean_d_embed =embed_tr_mat(z_embed, mean_tr_mat, gene_norm)
    print('calc embed')
    # # save matrix 
    for mat_name in ['z_mat', 'd_mat', 'zl_mat', 'dl_mat', 'z_embed', 'stoc_tr_mat', 'mean_tr_mat', 'stoc_d_embed', 'mean_d_embed', 'gene_vel', 'mean_gene_vel', 'batch_std_mat', 'ld_mat']:
        fname = data_dir + mat_name
        np.savetxt(fname, eval(mat_name))
    print('save objs')
    # index for plot
    plot_idx = np.random.choice(np.arange(zl_mat.shape[0]), plot_num)
    
    # plot embedding 
    embed_fname = fig_dir + 'embedding_stochastic.png'
    fig, axes = plt.subplots(1, 1, figsize=(7, 5 * 1))
    sns.scatterplot(x=z_embed[plot_idx, 0], y=z_embed[plot_idx, 1], hue=end_celltypes[plot_idx])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.quiver(z_embed[plot_idx, 0], z_embed[plot_idx, 1], stoc_d_embed[plot_idx, 0], stoc_d_embed[plot_idx, 1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.4, hspace=0.8)
    plt.savefig(embed_fname);plt.close('all')

    vembed_fname = fig_dir + 'embedding_mean.png'
    fig, axes = plt.subplots(1, 1, figsize=(7, 5 * 1))
    sns.scatterplot(x=z_embed[plot_idx, 0], y=z_embed[plot_idx, 1], hue=end_celltypes[plot_idx])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.quiver(z_embed[plot_idx, 0], z_embed[plot_idx, 1], mean_d_embed[plot_idx, 0], mean_d_embed[plot_idx, 1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.4, hspace=0.8)
    plt.savefig(vembed_fname);plt.close('all')

    # embedding speedn an fluctuation
    embed_fname = fig_dir + 'embedding_speed.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=gene_norm[:])
    plt.savefig(embed_fname);plt.close('all')

    embed_fname = fig_dir + 'embedding_fluctuation.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=gene_sd2[:])
    plt.savefig(embed_fname);plt.close('all')    

    embed_fname = fig_dir + 'embedding_fluctuation2.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=gene_sd2[:])
    plt.savefig(embed_fname);plt.close('all')    

    embed_fname = fig_dir + 'embedding_full.png'
    fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
    sns.scatterplot(x=z_embed[:, 0], y=z_embed[:, 1], hue=end_celltypes[:])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.4, hspace=0.8)
    plt.savefig(embed_fname);plt.close('all')

    for thresh in [1, 2, 3, 4, 5]:
        select_idx = gene_sd2 > thresh
        embed_fname = fig_dir + 'embedding_fluctuation2_' + str(thresh) + '.png'
        fig, axes = plt.subplots(1, 1, figsize=(5, 5 * 1))
        plt.scatter(z_embed[:, 0], z_embed[:, 1], c='gray')
        sns.scatterplot(x=z_embed[select_idx, 0], y=z_embed[select_idx, 1], hue=gene_sd2[select_idx])
        plt.savefig(embed_fname);plt.close('all')    
    print('plot fig')


def plot_gene_exp(z_embed, ld, gene_idx_vec, gene_name_vec, fig_name):
    fig, axes = plt.subplots(len(gene_idx_vec), 1, figsize=(5, 5 * len(gene_idx)))
    for i, gene_idx in enumerate(gene_idx_vec):
        axes[i].scatter(z_embed[:, 0], z_embed[:, 1], c=ld[:, i])
        axes[i].set_title(gene_name_vec[gene_idx])
    plt.savefig(fig_name);plt.close('all')



def update_dembed(adata, envdyn_exp, sigma=0.05, embed_mode='vicdyf_'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(adata.layers['spliced'].toarray()).to(envdyn_exp.device)
    u = torch.tensor(adata.layers['unspliced'].toarray()).to(envdyn_exp.device)
    z, d, qz, qd, px_z_ld, pu_zd_ld = envdyn_exp.model(x)
    # make zl centered embedding
    zl = qz.loc
    qd_mu, qd_logvar = envdyn_exp.model.enc_d(zl)
    qd = dist.Normal(qd_mu, envdyn_exp.model.softplus(qd_logvar))
    d = envdyn_exp.model.d_coeff * qd.rsample()
    px_z_ld = envdyn_exp.model.dec_z(zl)
    model_d_coeff = envdyn_exp.model.d_coeff
    dl = qd.loc
    dsd = qd.scale
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    model_d_coeff = envdyn_exp.model.d_coeff
    print('Extract info')
    # calc gene velocity
    pxd_zd_ld = envdyn_exp.model.dec_z(zl + d)
    gene_vel = (pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    gene_norm = np.linalg.norm(gene_vel, axis=1)
    # mean_gene_norm, gene_sd2, mean_gene_vel, batch_std_mat = calc_gene_mean_sd(z, qd, model_d_coeff, envdyn_exp.model)
    # batch_std_mat = batch_std_mat.cpu().detach().numpy()
    # mean_gene_vel = mean_gene_vel.cpu().detach().numpy()
    adata.layers['lambda'] = px_z_ld.cpu().detach().numpy()
    adata.layers['vicdyf_velocity'] = gene_vel
    # adata.layers['vicdyf_mean_velocity'] = mean_gene_vel
    # adata.layers['vicdyf_fluctuation'] = batch_std_mat
    adata.layers['corrected_vicdyf_fluctuation'] = (adata.layers['vicdyf_fluctuation'] - np.abs(adata.layers['vicdyf_mean_velocity'])).clip(min=0)
    # matrix 
    d_mat = d.cpu().detach().numpy()
    z_mat = z.cpu().detach().numpy()
    zl_mat = qz.loc.cpu().detach().numpy()
    dl_mat = qd.loc.cpu().detach().numpy() * model_d_coeff
    ld_mat = px_z_ld.cpu().detach().numpy()
    print('make mat')
    # calculate transition rate
    stoc_tr_mat = calc_tr_mat(zl.cpu().detach(), d.cpu().detach() / model_d_coeff, sigma)
    mean_tr_mat = calc_tr_mat(zl.cpu().detach(), dl.cpu().detach(), sigma)
    mean_gene_norm = np.linalg.norm(adata.layers['vicdyf_velocity'], axis=1)
    print('calc transition')
    # embed z
    z_embed = adata.obsm['X_' + embed_mode + 'umap']
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, gene_norm)
    mean_d_embed =embed_tr_mat(z_embed, mean_tr_mat, mean_gene_norm)
    adata.obsm['X_vicdyf_d'] = d_mat
    adata.obsm['X_vicdyf_dl'] = dl_mat
    adata.obsm['X_' + embed_mode + 'sdumap'] = stoc_d_embed.cpu().detach().numpy()
    adata.obsm['X_' + embed_mode + 'mdumap'] = mean_d_embed.cpu().detach().numpy()
    adata.obsm['mean_tr_mat'] = stoc_d_embed.cpu().detach().numpy()
    return(adata)


def update_only_dembed(adata, sigma=0.05, embed_mode='vicdyf_', model_d_coeff=0.01, n_neighbors=15, min_dist=0.01):
    d_mat = torch.tensor(adata.obsm['X_vicdyf_d'])
    dl_mat = torch.tensor(adata.obsm['X_vicdyf_dl'])
    z_mat = torch.tensor(adata.obsm['X_vicdyf_z'])
    zl_mat = torch.tensor(adata.obsm['X_vicdyf_zl'])
    # calculate transition rate
    stoc_tr_mat = calc_tr_mat(zl_mat, d_mat, sigma)
    mean_tr_mat = calc_tr_mat(zl_mat, dl_mat, sigma)
    mean_gene_norm = np.linalg.norm(adata.layers['vicdyf_mean_velocity'], axis=1)
    gene_norm = np.linalg.norm(adata.layers['vicdyf_velocity'], axis=1)
    print('calc transition')
    # embed z    
    z_embed = embed_z(zl_mat, n_neighbors=n_neighbors, min_dist=min_dist)
    adata.obsm['X_' + embed_mode + 'umap'] = z_embed
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, gene_norm)
    mean_d_embed =embed_tr_mat(z_embed, mean_tr_mat, mean_gene_norm)
    adata.obsm['X_' + embed_mode + 'sdumap'] = stoc_d_embed.cpu().detach().numpy()
    adata.obsm['X_' + embed_mode + 'mdumap'] = mean_d_embed.cpu().detach().numpy()
    adata.obsm['mean_tr_mat'] = stoc_d_embed.cpu().detach().numpy()
    return(adata)


def update_dembed_zver(adata, envdyn_exp, sigma=0.05, n_neighbors=15, min_dist=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(adata.layers['spliced'].toarray())
    u = torch.tensor(adata.layers['unspliced'].toarray())
    z, d, qz, qd, px_z_ld, pu_zd_ld = envdyn_exp.model(x)
    zl = qz.loc
    qd_mu, qd_logvar = envdyn_exp.model.enc_d(z)
    qd = dist.Normal(qd_mu, envdyn_exp.model.softplus(qd_logvar))
    d = envdyn_exp.model.d_coeff * qd.rsample()
    px_z_ld = envdyn_exp.model.dec_z(z)
    model_d_coeff = envdyn_exp.model.d_coeff
    dl = qd.loc
    dsd = qd.scale
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    model_d_coeff = envdyn_exp.model.d_coeff
    print('Extract info')
    # calc gene velocity
    pxd_zd_ld = envdyn_exp.model.dec_z(z + d)
    gene_vel = (pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    adata.layers['vicdyf_velocity'] = gene_vel
    gene_norm = np.linalg.norm(gene_vel, axis=1)
    mean_pxd_zd_ld = envdyn_exp.model.dec_z(z + dl * model_d_coeff)
    mean_gene_vel = (mean_pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    adata.layers['vicdyf_mean_velocity'] = mean_gene_vel
    mean_gene_norm = np.linalg.norm(mean_gene_vel, axis=1)
    # matrix 
    d_mat = d.cpu().detach().numpy()
    z_mat = z.cpu().detach().numpy()
    zl_mat = qz.loc.cpu().detach().numpy()
    dl_mat = qd.loc.cpu().detach().numpy() * model_d_coeff
    ld_mat = px_z_ld.cpu().detach().numpy()
    print('make mat')
    # calculate transition rate
    stoc_tr_mat = calc_tr_mat(zl_mat, d_mat, sigma)
    mean_tr_mat = calc_tr_mat(zl_mat, dl_mat, sigma)
    print('calc transition')
    # embed z
    z_embed = embed_z(zl_mat, n_neighbors=n_neighbors, min_dist=min_dist)
    adata.obsm['X_vicdyf_umap'] = z_embed
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, gene_norm)
    mean_d_embed =embed_tr_mat(z_embed, mean_tr_mat, mean_gene_norm)
    adata.obsm['X_vicdyf_z'] = z_mat
    adata.obsm['X_vicdyf_zl'] = zl_mat
    adata.obsm['X_vicdyf_d'] = d_mat
    adata.obsm['X_vicdyf_dl'] = dl_mat
    adata.obsm['X_vicdyf_sdumap'] = stoc_d_embed.cpu().detach().numpy()
    adata.obsm['X_vicdyf_mdumap'] = mean_d_embed.cpu().detach().numpy()
    adata.obsm['mean_tr_mat'] = stoc_d_embed.cpu().detach().numpy()
    return(adata)


def cluster_vel(adata, cluster_lab, clusters, nn=30, res=0.8):
    import scvelo as scv
    import scanpy as sc
    for cluster in clusters:
        select_peak_mask = adata.obs[cluster_lab] == cluster
        cluster_idx = np.where(select_peak_mask)[0]
        peakN_adata = adata[cluster_idx]
        # clustering v
        peakN_adata.obsm['X_vicdyf_norm_d'] = peakN_adata.obsm['X_vicdyf_d'] / np.linalg.norm(peakN_adata.obsm['X_vicdyf_d'], axis=1).reshape((-1, 1))
        sc.pp.neighbors(peakN_adata, use_rep='X_vicdyf_norm_d', n_neighbors=nn, key_added='v_nn')
        sc.tl.leiden(peakN_adata, neighbors_key='v_nn', key_added='v_leiden', resolution=res)
        adata.obs[cluster_lab + str(cluster) + '_vcluster'] = 'None'
        adata.obs[cluster_lab + str(cluster) + '_vcluster'][cluster_idx] = peakN_adata.obs['v_leiden']
        v_clusters = pd.unique(peakN_adata.obs['v_leiden'])
        for v_cluster in v_clusters:
            v_cluster_idx = np.where(adata.obs[cluster_lab + str(cluster) + '_vcluster'] == v_cluster)[0]
            peakN_vN_adata = adata[v_cluster_idx]
            pN_vN_mean_vel_vec = np.mean(peakN_vN_adata.layers['vicdyf_mean_velocity'], axis=0)
            pN_vN_stoc_vel_vec = np.mean(peakN_vN_adata.layers['vicdyf_velocity'], axis=0)
            pN_vN_dstoc_vel_vec = np.mean(peakN_vN_adata.layers['vicdyf_velocity'] - peakN_vN_adata.layers['vicdyf_mean_velocity'], axis=0)
            mean_label = 'p' + cluster + '_v' + v_cluster + '_meanv'
            adata.var[mean_label] = pN_vN_mean_vel_vec
            stoc_label = 'p' + cluster + '_v' + v_cluster + '_stocv'
            adata.var[stoc_label] = pN_vN_stoc_vel_vec
            dstoc_label = 'p' + cluster + '_v' + v_cluster + '_dstocv'
            adata.var[dstoc_label] = pN_vN_dstoc_vel_vec
    return(adata)

def visualize_velocity_cluster_wise(adata, cluster_lab, fig_name):
    import scvelo as scv
    import scanpy as sc
    clusters = pd.unique(adata.obs[cluster_lab])
    clusters = np.sort(clusters[clusters != 'None'])
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 5 * 1))
    for i, cluster in enumerate(clusters):
        select_peak_mask = adata.obs[cluster_lab] == cluster
        cluster_idx = np.where(select_peak_mask)[0]
        peakN_adata = adata[cluster_idx]
        scv.pl.velocity_embedding(peakN_adata, X=peakN_adata.obsm['X_vicdyf_umap'], V=peakN_adata.obsm['X_vicdyf_sdumap'], density=1.0, show=False, arrow_length=0.01, arrow_size=10, legend_loc='right_margin', ax=axes[i])
        axes[i].set_title(cluster)
    plt.savefig(fig_name);plt.close('all')
    return(0)

def extract_v_sig_idx(adata, peak, v1, v2, peak_lab='peak_cluster'):
    from statsmodels.stats import multitest
    from scipy import stats
    dstoc_mat = adata.layers['vicdyf_velocity'] - adata.layers['vicdyf_mean_velocity']
    peakv1_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v1)[0]
    peakv2_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v2)[0]
    pvalue = np.array([
        stats.ks_2samp(dstoc_mat[peakv1_idx, g], dstoc_mat[peakv2_idx, g])[1]
        for g in np.arange(dstoc_mat.shape[1])])
    fdr = multitest.fdrcorrection(pvalue)[1]
    return(fdr)

def extract_v_other_sig_idx(adata, peak, v1, peak_lab='peak_cluster'):
    from statsmodels.stats import multitest
    from scipy import stats
    dstoc_mat = adata.layers['vicdyf_velocity'] - adata.layers['vicdyf_mean_velocity']
    peakv1_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v1)[0]
    peakv2_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] != v1)[0]
    pvalue = np.array([
        stats.ks_2samp(dstoc_mat[peakv1_idx, g], dstoc_mat[peakv2_idx, g])[1]
        for g in np.arange(dstoc_mat.shape[1])])
    fdr = multitest.fdrcorrection(pvalue)[1]
    return(fdr)

 # test v0 v3 in peak 1
def plot_vcomp(adata, peak, v1, v2, fig_dir, qthresh=0.1):
    from scipy import stats
    fig, axes = plt.subplots(1, 1, figsize=(5 * 1, 5 * 1))
    deg_adata = deg_peak_v(adata, peak, v1, v2)
    order_adata = adata[:, deg_adata.var_names]
    v1_vec = np.array(order_adata.var['p' + peak + '_v' + v1 + '_dstocv'].values)
    v2_vec = np.array(order_adata.var['p' + peak + '_v' + v2 + '_dstocv'].values)
    mv1_vec = np.array(order_adata.var['p' + peak + '_v' + v1 + '_meanv'].values)
    mv2_vec = np.array(order_adata.var['p' + peak + '_v' + v2 + '_meanv'].values)
    diff_val = v1_vec - v2_vec
    avg_val = mv1_vec + mv2_vec
    plt.scatter(avg_val, diff_val, s=0.1, c='gray')
    # deg plot
    sig_idx = deg_adata.uns['rank_genes_groups']['pvals_adj']['p' + peak + '_v' + v1] < qthresh
    plt.scatter(avg_val[sig_idx], diff_val[sig_idx], s=0.1, c='red')
    # marker plot
    markers = ['IL7R', 'ITGAL']
    for marker in markers:
        select_idx = order_adata.var_names == marker
        ranking = str(int(stats.rankdata(-diff_val)[select_idx][0]))
        rranking = str(int(stats.rankdata(diff_val)[select_idx][0]))
        plt.text(avg_val[select_idx], diff_val[select_idx], s=marker + ' top' + ranking + ' bottom' + rranking, c='red', fontsize=5)
    diff_v_figname = fig_dir + 'p' + peak + '_v' + v1 + v2 +'_comp_2021_01_20.pdf'
    plt.savefig(diff_v_figname);plt.close('all')
    return(0)

def plot_dstoc_vcluster(adata, peak, v1, figname, qthresh=0.1, peak_lab='peak_cluster'):
    from scipy import stats
    fig, axes = plt.subplots(1, 1, figsize=(5 * 1, 5 * 1))
    # deg_adata = deg_peak_v(adata, peak, v1, v2)
    # order_adata = adata[:, deg_adata.var_names]
    peak_adata = adata[adata.obs['peak_cluster'] == peak]
    mean_vmat = peak_adata.layers['vicdyf_mean_velocity']
    dstoc_vmat = peak_adata.layers['vicdyf_velocity'] - peak_adata.layers['vicdyf_mean_velocity']
    vcluster_lab = peak_lab + peak + '_vcluster'
    v1_idx = peak_adata.obs[vcluster_lab] == v1
    other_idx = peak_adata.obs[vcluster_lab] != v1
    v1_vec = np.mean(dstoc_vmat[v1_idx], axis=0)
    v2_vec = np.mean(dstoc_vmat[other_idx], axis=0)
    mv1_vec = np.mean(mean_vmat[v1_idx], axis=0)
    mv2_vec = np.mean(mean_vmat[other_idx], axis=0)
    diff_val = v1_vec - v2_vec
    avg_val = mv1_vec - mv2_vec
    plt.scatter(avg_val, diff_val, s=0.1, c='gray')
    # deg plot
    sig_idx = extract_v_other_sig_idx(peak_adata, peak, v1) < 0.001
    print(sum(sig_idx))
    plt.scatter(avg_val[sig_idx], diff_val[sig_idx], s=0.1, c='red')
    # marker plot
    markers = ['IL7R', 'ITGAL']
    for marker in markers:
        select_idx = peak_adata.var_names == marker
        ranking = str(int(stats.rankdata(-diff_val)[select_idx][0]))
        rranking = str(int(stats.rankdata(diff_val)[select_idx][0]))
        plt.text(avg_val[select_idx], diff_val[select_idx], s=marker + ' top' + ranking + ' bottom' + rranking, c='black', fontsize=5)
    plt.savefig(figname);plt.close('all')
    return(0)

def plot_dstoc_vcluster_txt(adata, peak, v1, figname, qthresh=0.1, peak_lab='peak_cluster'):
    from scipy import stats
    fig, axes = plt.subplots(1, 1, figsize=(5 * 1, 5 * 1))
    # deg_adata = deg_peak_v(adata, peak, v1, v2)
    # order_adata = adata[:, deg_adata.var_names]
    peak_adata = adata[adata.obs['peak_cluster'] == peak]
    mean_vmat = peak_adata.layers['vicdyf_mean_velocity']
    dstoc_vmat = peak_adata.layers['vicdyf_velocity'] - peak_adata.layers['vicdyf_mean_velocity']
    vcluster_lab = peak_lab + peak + '_vcluster'
    v1_idx = peak_adata.obs[vcluster_lab] == v1
    other_idx = peak_adata.obs[vcluster_lab] != v1
    v1_vec = np.mean(dstoc_vmat[v1_idx], axis=0)
    v2_vec = np.mean(dstoc_vmat[other_idx], axis=0)
    mv1_vec = np.mean(mean_vmat[v1_idx], axis=0)
    mv2_vec = np.mean(mean_vmat[other_idx], axis=0)
    diff_val = v1_vec - v2_vec
    avg_val = mv1_vec - mv2_vec
    plt.scatter(avg_val, diff_val, s=0.1, c='gray')
    for gidx in np.arange(peak_adata.shape[1]):
        plt.text(avg_val[gidx], diff_val[gidx], s=peak_adata.var_names.values[gidx], c='gray', fontsize=1)
    # deg plot
    sig_idx = np.where(extract_v_other_sig_idx(peak_adata, peak, v1) < 0.001)[0]
    for gidx in sig_idx:
        plt.text(avg_val[gidx], diff_val[gidx], s=peak_adata.var_names.values[gidx], c='red', fontsize=1)    
    # marker plot
    markers = ['IL7R', 'ITGAL']
    for marker in markers:
        select_idx = peak_adata.var_names == marker
        plt.scatter(avg_val[select_idx], diff_val[select_idx], s=1, c='blue')
    plt.savefig(figname);plt.close('all')
    return(0)

def plot_vcomp_txt(adata, peak, v1, v2, fig_dir, qthresh=0.1):
    from scipy import stats
    fig, axes = plt.subplots(1, 1, figsize=(5 * 1, 5 * 1))
    deg_adata = deg_peak_v(adata, peak, v1, v2)
    order_adata = adata[:, deg_adata.var_names]
    v1_vec = np.array(order_adata.var['p' + peak + '_v' + v1 + '_dstocv'].values)
    v2_vec = np.array(order_adata.var['p' + peak + '_v' + v2 + '_dstocv'].values)
    diff_val = v1_vec - v2_vec
    avg_val = v1_vec + v2_vec
    plt.scatter(avg_val, diff_val, s=0.01, c='gray')
    for gidx in np.arange(order_adata.shape[1]):
        plt.text(avg_val[gidx], diff_val[gidx], s=order_adata.var_names.values[gidx], c='gray', fontsize=1)
    # deg plot
    sig_idx = np.where(deg_adata.uns['rank_genes_groups']['pvals_adj']['p' + peak + '_v' + v1] < qthresh)[0]
    for gidx in sig_idx:
        plt.text(avg_val[gidx], diff_val[gidx], s=order_adata.var_names.values[gidx], c='red', fontsize=1)    
    # marker plot
    markers = ['IL7R', 'ITGAL']
    for marker in markers:
        select_idx = order_adata.var_names == marker
        plt.scatter(avg_val[select_idx], diff_val[select_idx], s=1)
    plt.savefig(figname);plt.close('all')
    return(0)

def make_vcomp(adata, peak, v1, v2, peak_lab='peak_cluster'):
    from scipy import stats
    import scanpy as sc
    v1_vec = np.array(adata.var['p' + peak + '_v' + v1 + '_dstocv'].values)
    v2_vec = np.array(adata.var['p' + peak + '_v' + v2 + '_dstocv'].values)
    diff_val = v1_vec - v2_vec
    avg_val = v1_vec + v2_vec
    peakv1_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v1)[0]
    peakv2_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v2)[0]
    

def calc_tr_prob(adata, peak, v, peak_lab='peak_cluster'):
    tr_mat = adata.obsm['tr_mat']
    peakv_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v)[0]
    v_tr_probs = np.mean(tr_mat[peakv_idx], axis=0)
    return(v_tr_probs)


def extract_topn_trans_cell(adata, peak, v, top_n=100):
    tr_prob = calc_tr_prob(adata, peak, v)
    thresh = -np.sort(-tr_prob)[top_n]
    select_adata = adata[tr_prob > thresh]
    select_adata.obs['vannot'] = 'p' + peak + '_v' + v
    return(select_adata)

def deg_peak(adata, peak, v1, v2):
    import anndata
    import scanpy as sc
    top_n = int(adata.shape[0] * 0.05)
    v1_adata = extract_topn_trans_cell(adata, peak, v1, top_n=top_n)
    v2_adata = extract_topn_trans_cell(adata, peak, v2, top_n=top_n)
    common_cells = np.intersect1d(v1_adata.obs_names, v2_adata.obs_names)
    deg_adata = anndata.concat([
        v1_adata, v2_adata])
    deg_adata = deg_adata[
        deg_adata.obs_names.isin(common_cells) == False]
    sc.tl.rank_genes_groups(deg_adata, 'vannot', method='wilcoxon')
    deg_adata = deg_adata[:, deg_adata.uns['rank_genes_groups']['names']['p' + peak + '_v' + v1]]
    return(deg_adata)    


def extract_topn_trans_cell(adata, peak, v, top_n=100):
    tr_prob = calc_tr_prob(adata, peak, v)
    thresh = -np.sort(-tr_prob)[top_n]
    select_adata = adata[tr_prob > thresh]
    select_adata.obs['vannot'] = 'p' + peak + '_v' + v
    return(select_adata)

def deg_peak_v(adata, peak, v1, v2):
    import anndata
    import scanpy as sc
    top_n = int(adata.shape[0] * 0.05)
    v1_adata = extract_topn_trans_cell(adata, peak, v1, top_n=top_n)
    v2_adata = extract_topn_trans_cell(adata, peak, v2, top_n=top_n)
    common_cells = np.intersect1d(v1_adata.obs_names, v2_adata.obs_names)
    deg_adata = anndata.concat([
        v1_adata, v2_adata])
    deg_adata = deg_adata[
        deg_adata.obs_names.isin(common_cells) == False]
    sc.tl.rank_genes_groups(deg_adata, 'vannot', method='wilcoxon')
    deg_adata = deg_adata[:, deg_adata.uns['rank_genes_groups']['names']['p' + peak + '_v' + v1]]
    return(deg_adata)
    
def construct_deg_table(adata, peak, v1, v2, peak_lab='peak_cluster'):
    from statsmodels.stats import multitest
    from scipy import stats
    vcluster_lab = peak_lab + peak + '_vcluster'
    tr_mat = adata.obsm['tr_mat']
    peakv1_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v1)[0]
    peakv2_idx = np.where(adata.obs[peak_lab + peak + '_vcluster'] == v2)[0]
    v1_tr_probs = np.mean(tr_mat[peakv1_idx], axis=0)
    v2_tr_probs = np.mean(tr_mat[peakv2_idx], axis=0)
    
    dstoc_mat = adata.layers['vicdyf_velocity'] - adata.layers['vicdyf_mean_velocity']
    pvalue = np.array([
        stats.ks_2samp(dstoc_mat[peakv1_idx, g], dstoc_mat[peakv2_idx, g])[1]
        for g in np.arange(dstoc_mat.shape[1])])
    fdr = multitest.fdrcorrection(pvalue)[1]
    return(fdr)


def topn(v, n):
    return(-np.sort(-v)[n])
    
def outlier_correction(v, o_prop=0.01):
    o_n = int(v.shape[0] * o_prop)
    corr_v = np.copy(v)
    topn_v = topn(v, o_n)
    bottomn_v = -topn(-v, o_n)
    corr_v[corr_v > topn_v] = topn_v
    corr_v[corr_v < bottomn_v] = bottomn_v
    return(corr_v)

def one_hot(i, n):
    v = np.zeros(n)
    v[i] = 1.0
    return(v)

def reg_norm_tensor(t, dim=1):
    norm_t = torch.norm(t, dim=1, keepdim=True)
    norm_t[norm_t == 0] = 1
    reg_t = t / norm_t
    return(reg_t)


@torch.no_grad()
def calc_poisson_prob(ld, norm_mat, obs):
    p_z = dist.Poisson(ld * norm_mat + 1.0e-16)
    l = p_z.log_prob(obs)
    return(l)
        


@torch.no_grad()
def calc_lu_diff(x, u, norm_mat, model):
    z, d, qz, qd, px_z_ld, pu_zd_ld, sl, qsl, ul, qul = model(x, u)
    d = qd.loc * model.d_coeff
    pxd_zd_ld = model.dec_z(z + d)
    pxmd_zd_ld = model.dec_z(z - d)
    diff_px_zd_ld = pxd_zd_ld - pxmd_zd_ld
    gamma = model.sigmoid(model.loggamma) * model.dt
    beta = model.sigmoid(model.logbeta) * model.dt
    pu_zdl_ld = model.softplus(diff_px_zd_ld + px_z_ld * gamma) / beta
    lu_d = calc_poisson_prob(pu_zd_ld, norm_mat, u)
    lu_dl = calc_poisson_prob(pu_zdl_ld, norm_mat, u)
    return(lu_d - lu_dl)

@torch.no_grad()    
def update_dembed_full(adata, envdyn_exp, sigma=0.05, embed_mode='vicdyf_', nn=30, mdist=0.1, dz_var_prop=0.05, no_tr=False, fix_embed=False):
    x = torch.tensor(adata.layers['spliced'].toarray().astype(float)).float()
    u = torch.tensor(adata.layers['unspliced'].toarray().astype(float)).float()
    s = x
    norm_mat = torch.sum(s, dim=1).view(-1, 1) * torch.sum(s, dim=0).view(1, -1)
    norm_mat = torch.mean(s) * norm_mat / torch.mean(norm_mat)
    envdyn_exp.device = torch.device('cpu')
    envdyn_exp.model = envdyn_exp.model.to(envdyn_exp.device)
    z, d, qz, qd, px_z_ld, pu_zd_ld, sl, qsl, ul, qul, dyn_loss = envdyn_exp.model(in_x, u)
    # lu_diff = calc_lu_diff(in_x, u, norm_mat, envdyn_exp.model)
    # make zl centered embedding
    zl = qz.loc
    d, qd = envdyn_exp.model.enc_d(zl)
    d = envdyn_exp.model.d_coeff * qd.rsample()
    px_z_ld = envdyn_exp.model.dec_z(zl)
    model_d_coeff = envdyn_exp.model.d_coeff
    dl = qd.loc * model_d_coeff
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    print('Extract info')
    # calc gene velocity
    pxd_zd_ld = envdyn_exp.model.dec_z(zl + d)
    gene_vel = (pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    gene_norm = np.linalg.norm(gene_vel, axis=1)
    mean_gene_norm, gene_sd2, mean_gene_vel, batch_std_mat = calc_gene_mean_sd(zl, qd, model_d_coeff, envdyn_exp.model)
    dmean_gene_norm, dgene_sd2, dmean_gene_vel, dbatch_std_mat = calc_gene_mean_sd(zl + d, qd, model_d_coeff, envdyn_exp.model)
    dmean_gene_norm, dgene_sd2, dmean_gene_vel, dbatch_std_mat = dmean_gene_norm - mean_gene_norm, dgene_sd2 - gene_sd2, dmean_gene_vel - mean_gene_vel, dbatch_std_mat - batch_std_mat
    batch_std_mat = batch_std_mat.cpu().detach().numpy()
    mean_gene_vel = mean_gene_vel.cpu().detach().numpy()
    dbatch_std_mat = dbatch_std_mat.cpu().detach().numpy()
    dmean_gene_vel = dmean_gene_vel.cpu().detach().numpy()
    adata.layers['lambda'] = px_z_ld.cpu().detach().numpy()
    adata.layers['vicdyf_velocity'] = gene_vel
    adata.layers['vicdyf_mean_velocity'] = mean_gene_vel
    adata.layers['vicdyf_fluctuation'] = batch_std_mat
    # adata.layers['lu_diff'] = lu_diff.cpu().detach().numpy()
    # adata.obs['lu_diff'] = np.sum(adata.layers['lu_diff'], axis=1)
    # # matrix 
    d_mat = np.copy(d.cpu().detach().numpy())
    z_mat = np.copy(z.cpu().detach().numpy())
    zl_mat = np.copy(zl.cpu().detach().numpy())
    dl_mat = np.copy(dl.cpu().detach().numpy())
    ld_mat = np.copy(px_z_ld.cpu().detach().numpy())
    print('make mat')
    adata.obsm['X_vicdyf_z'] = z_mat
    adata.obsm['X_vicdyf_zl'] = zl_mat
    adata.obsm['X_vicdyf_d'] = d_mat
    adata.obsm['X_vicdyf_dl'] = dl_mat
    adata.obs['vicdyf_fluctuation'] = np.mean(adata.layers['vicdyf_fluctuation'], axis=1)
    adata.obs['vicdyf_velocity'] = np.mean(np.abs(adata.layers['vicdyf_velocity']), axis=1)
    adata.obs['vicdyf_mean_velocity'] = np.mean(np.abs(adata.layers['vicdyf_mean_velocity']), axis=1)
    # calculate transition rate
    stoc_tr_mat = calc_tr_mat(zl.cpu().detach(), d.cpu().detach(), sigma)
    mean_tr_mat = calc_tr_mat(zl.cpu().detach(), dl.cpu().detach(), sigma)
    mean_gene_norm = np.linalg.norm(adata.layers['vicdyf_velocity'], axis=1)
    print('calc transition')
    # embed z
    if not fix_embed:
        z_embed = embed_z(zl_mat, n_neighbors=nn, min_dist=mdist)
        adata.obsm['X_vicdyf_umap'] = z_embed
    else:
        z_embed = adata.obsm['X_vicdyf_umap']
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, gene_norm)
    mean_d_embed = embed_tr_mat(z_embed, mean_tr_mat, mean_gene_norm)
    adata.obsm['mean_tr_mat'] = mean_tr_mat.detach().numpy()
    adata.obsm['stoc_tr_mat'] = stoc_tr_mat.detach().numpy()
    adata.obsm['X_' + embed_mode + 'sdumap'] = stoc_d_embed.cpu().detach().numpy()
    adata.obsm['X_' + embed_mode + 'mdumap'] = mean_d_embed.cpu().detach().numpy()
    adata.obs['sl'] = qsl.loc.cpu().detach().numpy()
    adata.obs['ul'] = qul.loc.cpu().detach().numpy()
    return(adata)



@torch.no_grad()    
def update_latent(adata, envdyn_exp, sigma=0.05, embed_mode='vicdyf_', nn=30, mdist=0.1, dz_var_prop=0.05):
    x = torch.tensor(adata.layers['spliced'].toarray().astype(float)).float()
    u = torch.tensor(adata.layers['unspliced'].toarray().astype(float)).float()
    s = x
    norm_mat = torch.sum(s, dim=1).view(-1, 1) * torch.sum(s, dim=0).view(1, -1)
    norm_mat = torch.mean(s) * norm_mat / torch.mean(norm_mat)
    envdyn_exp.device = torch.device('cpu')
    envdyn_exp.model = envdyn_exp.model.to(envdyn_exp.device)
    z, d, qz, qd, px_z_ld, pu_zd_ld, sl, qsl, ul, qul, dyn_loss = envdyn_exp.model(in_x, u)
    # lu_diff = calc_lu_diff(in_x, u, norm_mat, envdyn_exp.model)
    # make zl centered embedding
    zl = qz.loc
    z_mat = np.copy(z.cpu().detach().numpy())
    zl_mat = np.copy(zl.cpu().detach().numpy())
    adata.obsm['X_vicdyf_z'] = z_mat
    adata.obsm['X_vicdyf_zl'] = zl_mat
    # embed z
    z_embed = embed_z(zl_mat, n_neighbors=nn, min_dist=mdist)
    adata.obsm['X_vicdyf_umap'] = z_embed
    return(adata)

