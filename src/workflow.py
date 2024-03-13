from re import I
from matplotlib import cm
from threading import enumerate
from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import scvelo as scv
import joblib
import os
import anndata
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import sys
from . import commons
import umap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colors
import importlib
import torch
import json
import pytorch_lightning as pl
from . import dataset
from . import modules
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

class tqdm(_tqdm):
    """
    Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from flickering
    """

class FixTqdmProgress(ProgressBar):
    def init_sanity_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


    def init_train_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=80,
            file=sys.stdout,
            smoothing=0,
        )
        return bar


    def init_predict_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=80,
            file=sys.stdout,
            smoothing=0,
        )
        return bar


    def init_validation_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            ncols=80,
            file=sys.stdout,
        )
        return bar


    def init_test_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=80,
            file=sys.stdout,
        )
        return bar


def make_norm_mat(s, ratio=1.0):
    snorm_mat = torch.mean(s, dim=1, keepdim=True) * torch.mean(s, dim=0, keepdim=True)
    # snorm_mat = torch.mean(s, dim=0, keepdim=True) * torch.ones_like(s)
    snorm_mat = ratio * torch.mean(s) * snorm_mat / torch.mean(snorm_mat)
    return snorm_mat

def make_datasets(adata, condition_key='condition', batch_key='sample', cond_in_obsm=False):
    if type(adata.layers['spliced']) == np.ndarray:
        s = torch.tensor(adata.layers['spliced'].astype(int)).float()
        u = torch.tensor(adata.layers['unspliced'].astype(int)).float()
    else:
        s = torch.tensor(adata.layers['spliced'].toarray().astype(int)).float()
        u = torch.tensor(adata.layers['unspliced'].toarray().astype(int)).float()
    u[:, ~adata.var.dynamics_genes] = 0
    snorm_mat = make_norm_mat(s)
    unorm_mat = make_norm_mat(u)
    adata.obs['batch'] = adata.obs[batch_key]
    b, adata = commons.make_sample_one_hot_mat(adata, sample_key=batch_key)
    if cond_in_obsm:
        t = torch.tensor(np.array(adata.obsm[condition_key])).float()
    elif type(condition_key) is list:
        cond_mat_list = []
        for cond in condition_key:
            t, adata = commons.make_sample_one_hot_mat(adata, sample_key=cond, stored_obsm_key=cond)
            cond_mat_list.append(t)
        t = torch.cat(cond_mat_list, dim=-1)
        adata.obsm[condition_key] = t.numpy()
    else:
        t, adata = commons.make_sample_one_hot_mat(adata, sample_key=condition_key, stored_obsm_key=condition_key)
    return s, u, snorm_mat, unorm_mat, b, t, adata


def optimize_vdicdyf(train_ds, val_ds, model_params, checkpoint_dirname, module, epoch=1000, lr=0.003, patience=30, batch_size=128, two_step=False, dyn_mode=True, only_vae=False, monitor_loss='elbo'):
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=6, pin_memory=True)
    lit_envdyn = module(**model_params)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dirname, monitor=monitor_loss)
    bar = FixTqdmProgress()
    trainer = pl.Trainer(max_epochs=epoch, gpus=1, callbacks=[bar, EarlyStopping(monitor=monitor_loss, patience=patience), checkpoint_callback])
    if two_step:
        lit_envdyn.vae_mode()
        trainer.fit(lit_envdyn, train_loader, val_loader)
        lit_envdyn = module.load_from_checkpoint(**model_params, checkpoint_path=checkpoint_callback.best_model_path)
        lit_envdyn.configure_optimizers()
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dirname)
        # lit_envdyn.dyn_mode()
        trainer = pl.Trainer(max_epochs=epoch, gpus=1, callbacks=[bar, EarlyStopping(monitor=monitor_loss, patience=patience), checkpoint_callback])
        if dyn_mode:
            lit_envdyn.dyn_mode()
    elif only_vae:
        lit_envdyn.vae_mode()
    trainer.fit(lit_envdyn, train_loader, val_loader)
    lit_envdyn = module.load_from_checkpoint(**model_params, checkpoint_path=checkpoint_callback.best_model_path)
    lit_envdyn.eval()
    return lit_envdyn


def postprocess_civcdyf(adata, model, use_highly_variable=False):
    adata = adata[np.random.permutation(adata.obs_names)]
    if use_highly_variable:
        adata = adata[:, adata.var.highly_variable]
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata)
    ds = dataset.CvicDyfDataSet(s, u, snorm_mat, unorm_mat, b, t)
    adata = commons.update_latent(adata, model, ds)
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata = anndata.concat([
        commons.update_dembed_full(
            adata[idx], model, ds[idx], sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1, fix_embed=True)
        for idx in idx_list], merge='unique')
    # adata = commons.update_dembed_full(adata, model, ds, sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1)
    return adata


def modify_model_params(model_params, adata, condition_key='condition', batch_key='sample', cond_in_obsm=True, cl_key=None):
    if not 'highly_variable' in adata.var.columns:
        adata.var['highly_variable'] = True
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata, batch_key=batch_key, condition_key=condition_key, cond_in_obsm=cond_in_obsm)
    model_params['x_dim'] = s.size()[1]
    model_params['batch_num'] = b.size()[1]
    model_params['c_dim'] = t.size()[1]
    if not cl_key is None:
        c, adata = commons.make_sample_one_hot_mat(adata, sample_key=cl_key, stored_obsm_key=cl_key)
        model_params['class_num'] = c.size()[1]
    return model_params

def conduct_cvicdyf_inference(adata, model_params, checkpoint_dirname, epoch=1000, lr=0.003, patience=100, batch_size=128, use_highly_variable=False, two_step=False, dyn_mode=True, module=modules.Cvicdyf, only_vae=False, condition_key='condition', batch_key='sample_id', class_key='sample_id', cond_in_obsm=False, non_opt=False, lit_envdyn=None):
    if use_highly_variable:
        adata = adata[:, adata.var.highly_variable]
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
    if class_key == 'sample_id':
        class_key = batch_key
    c, adata = commons.make_sample_one_hot_mat(adata, sample_key=class_key, stored_obsm_key=class_key)
    model_params['x_dim'] = s.size()[1]
    model_params['batch_num'] = b.size()[1]
    model_params['c_dim'] = t.size()[1]
    model_params['class_num'] = c.size()[1]
    if module == modules.Cvicdyf:
        ds = dataset.CvicDyfDataSet(s, u, snorm_mat, unorm_mat, b, t)
    else:
        ds = TensorDataset(s, u, snorm_mat, unorm_mat, b, t, c) 
    train_ds, val_ds, test_ds = commons.separate_dataset(ds)
    if not non_opt:
        lit_envdyn = optimize_vdicdyf(train_ds, val_ds, model_params, checkpoint_dirname, module, epoch=epoch, lr=lr, patience=patience, batch_size=batch_size, two_step=two_step, dyn_mode=True, only_vae=only_vae)
    model = lit_envdyn
    adata = commons.update_latent(adata, model, ds)
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata = anndata.concat([
        commons.update_dembed_full(
            adata[idx], model, ds[idx], sigma=0.1, embed_mode='vicdyf_', nn=100, mdist=0.1, fix_embed=True)
        for idx in idx_list], merge='unique')
    return adata, lit_envdyn



def conduct_vicdyf_inference(adata, model_params, checkpoint_dirname, epoch=1000, lr=0.003, patience=100, batch_size=128, two_step=False, only_vae=False, module=modules.Cvicdyf, use_highly_variable=False):
    if use_highly_variable:
        adata = adata[:, adata.var.highly_variable]
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata)
    model_params['x_dim'] = s.size()[1]
    ds = dataset.EnvDynDataSet(s, u, snorm_mat, unorm_mat)
    train_ds, val_ds, test_ds = commons.separate_dataset(ds)
    lit_envdyn = optimize_vdicdyf(train_ds, val_ds, model_params, checkpoint_dirname, module, epoch=epoch, lr=lr, patience=patience, batch_size=batch_size, two_step=two_step, dyn_mode=True, only_vae=only_vae)
    envdyn_exp = envdyn.ExpMock(lit_envdyn)
    model = envdyn_exp.model
    adata = commons.update_latent(adata, model, ds)
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata = anndata.concat([
        commons.update_dembed_full(
            adata[idx], model, ds[idx], sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1, fix_embed=True)
        for idx in idx_list], merge='unique')
    return adata, lit_envdyn


def make_sub_set_cells_with_map(int_adata, map_adata, cell_set, cluster_label):
    norm_p = map_adata.obsm['map2sp'] /  map_adata.obsm['map2sp'].sum(axis=1, keepdims=True)
    coloc_mat = norm_p @ norm_p.T
    coloc_df = pd.DataFrame(np.log(coloc_mat + 1.0e-16 ) - np.log(1.0 / norm_p.shape[1]), index=map_adata.obs_names, columns=map_adata.obs_names)
    common_cells = np.intersect1d(int_adata.obs_names, map_adata[map_adata.obs[cluster_label].isin(cell_set)].obs_names)
    eadata = int_adata[common_cells]
    mcoloc_df = coloc_df.loc[eadata.obs_names, ~coloc_df.columns.isin(eadata.obs_names)]
    eadata.obsm['coloc'] = mcoloc_df.values
    eadata.uns['coloc_partners'] = list(mcoloc_df.columns)
    return eadata


def extract_top_corr_genes(adata, corr_thresh=0.6):
    adata.X = commons.safe_to_array(adata.layers['spliced'])
    counts = np.array(adata.layers['unspliced'].sum(axis=0)).reshape(-1)
    total_exps = pd.Series(counts, index=adata.var_names).sort_values(ascending=False)
    top_exp_genes = total_exps[total_exps > adata.shape[0] * 0.01].index
    adata = scv.pp.moments(adata, n_neighbors=min(int(adata.shape[0] * 0.1), 100), copy=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    corrs = pd.Series(scv.utils.vcorrcoef(adata.layers['Ms'], adata.layers['Mu'], axis=0), index=adata.var_names)
    top_corr_genes = corrs.index[corrs > corr_thresh]
    top_corr_genes = np.intersect1d(top_corr_genes, top_exp_genes)
    return top_corr_genes, adata.var_names

def clustering_on_vicdyf(est_adata, resolution=0.3, n_neighbors=30):
    sc.pp.neighbors(est_adata, n_neighbors=n_neighbors, use_rep='X_vicdyf_zl')
    sc.tl.leiden(est_adata, resolution=resolution)
    return est_adata


def add_stochastic_vel(adata, model_params, opt_param_path, module=modules.Cvicdyf, condition_key='condition', batch_key='sample', cond_in_obsm=False, non_opt=False, lit_envdyn=None):
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
    model_params['x_dim'] = s.size()[1]
    model_params['batch_num'] = b.size()[1]
    model_params['c_dim'] = t.size()[1]
    ds = dataset.CvicDyfDataSet(s, u, snorm_mat, unorm_mat, b, t)
    train_ds, val_ds, test_ds = commons.separate_dataset(ds)
    lit_envdyn = module(**model_params)
    lit_envdyn.load_state_dict(torch.load(opt_param_path))
    envdyn_exp = envdyn.ExpMock(lit_envdyn)
    model = envdyn_exp.model
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata.layers['vicdyf_velocity'] = np.concatenate([
        commons.safe_numpy(commons.calc_stochastic_vel(
            adata[idx], model, ds[idx]))
        for idx in idx_list], axis=0)
    adata.layers['norm_vicdyf_velocity'] = adata.layers['vicdyf_velocity'] / adata.layers['lambda'].mean(axis=0)
    return adata, lit_envdyn


def load_pretrained_model(opt_path, model_params, est_adata, condition_key=None, batch_key=None, module=modules.Cvicdyf, pl_ckpt=False):
    model_params = modify_model_params(model_params, est_adata, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=True)
    if pl_ckpt:
        model = module.load_from_checkpoint(**model_params, checkpoint_path=opt_path)
    else:
        model = module(**model_params)
        model.load_state_dict(torch.load(opt_path))
    return model


def conduct_jacobian_analysis(adata, model, top_modes=5):
    _, dd_dc = commons.calc_jac(adata, model, condition_key='coloc', batch_key='batch', calc_dvdd=False)
    res = torch.svd(dd_dc.view(-1, dd_dc.size()[-1]))
    u, s, v = [commons.safe_numpy(mat) for mat in res]
    u = u.reshape(adata.shape[0], -1, v.shape[1])
    for i in range(top_modes):
        adata.obsm[f'dd_dc{i}'] = u[:, :, i]
    adata.uns['dd_dc_s'] = s
    adata.uns['dd_dc_v'] = v[:, :top_modes]
    z_mat = adata.obsm['X_vicdyf_zl']
    u_mat = adata.obsm['X_vicdyf_umap']
    for i in range(top_modes):
        d_mat = adata.obsm[f'dd_dc{i}']
        adata.obsm[f'dd_dc_embed{i}'] = commons.calc_d_embed_grad(z_mat, d_mat, u_mat, nn=30)[1]
        adata.obs[f'dd_dc_val{i}'] = np.linalg.norm(d_mat, axis=1)
    return adata

def analyze_flux_into_clusters(adata, cluster, i, sign, cluster_key):
    clusters = adata.obs[cluster_key].unique()
    cluster_vec_dict = {
        cluster: (adata.obs[cluster_key] == cluster).astype(int).values.reshape(-1, 1)
        for cluster in clusters
    }
    d_mat = adata.obsm[f'dd_dc{i}']
    z_mat = adata.obsm['X_vicdyf_zl']
    dp_dt = commons.calc_d_embed_grad(z_mat, d_mat, cluster_vec_dict[cluster])[1]
    adata.obs[f'mode{i}_flux_into_{cluster}'] = sign * dp_dt.reshape(-1)
    return adata

def calc_covdiff(adata, model, mode, sign, condition_key, batch_key, sep_num=5):
    sub_idxs = commons.make_idx_list(adata.shape[0], sep_num)
    dv_dd = torch.cat(
        [commons.calc_jac(adata[idx], model, condition_key=condition_key, batch_key=batch_key)[0]
        for idx in sub_idxs], dim=0)
    dv_dd = commons.safe_numpy(dv_dd)
    dd_dc = adata.obsm[f'dd_dc{mode}']
    adata.layers[f'cov_diff_{mode * sign}'] = sign * (dv_dd @ dd_dc[:, :, np.newaxis]).reshape(dv_dd.shape[0], dv_dd.shape[1]) / np.mean(adata.layers['lambda'], axis=0)
    return adata

def analyze_ligand_activity(condiff_vec, lt_df, q=0.99):
    ligands = lt_df.columns
    common_genes = np.intersect1d(lt_df.index.astype(str), condiff_vec.index)
    lt_df = lt_df.loc[common_genes, ligands]
    norm_lt_df = commons.make_norm_lt_df(lt_df, q=0.99)
    common_genes = np.intersect1d(norm_lt_df.index, condiff_vec.index)
    ligacs = pd.Series(condiff_vec[common_genes].values @ norm_lt_df.loc[common_genes].values, index=norm_lt_df.columns)
    return ligacs


def recover_model(est_adata, opt_id, opt_path):
    opt_conds = json.load(open('snake_meta/optimization_conds.json'))[opt_id]
    model_params = json.load(open('snake_meta/model_params.json'))[opt_conds.get('model_id', '221111')]
    batch_key = opt_conds.get('batch_key', 'sample_id')
    model_params = modify_model_params(model_params, est_adata, condition_key='tot_cond', batch_key=batch_key, cond_in_obsm=True)
    model = modules.Cvicdyf(**model_params)
    model.load_state_dict(torch.load(opt_path))
    return model