from sklearn.neighbors import NearestNeighbors
from matplotlib import colors, cm
from matplotlib import pyplot as plt
from . import commons
import numpy as np
import networkx as nx
import scvelo as scv

## based on https://stackoverflow.com/questions/14938541/how-to-improve-the-label-placement-in-scatter-plot
def repel_labels(ax, xs, ys, labs, k=0.01):
    G = nx.DiGraph()
    data_ids = [f'data_{lab}' for lab in labs]
    for lab, data_id in zip(labs, data_ids):
        G.add_node(lab)
        G.add_node(data_id)
        G.add_edge(lab, data_id)
    init_pos_dict = {
        key: (x, y)
        for key, x, y in zip(
            np.concatenate([labs, data_ids]),
            np.tile(xs, 2), np.tile(ys, 2))
    }
    mod_pos_dict = nx.spring_layout(G, pos=init_pos_dict, fixed=data_ids, k=k, scale=None)
    arrowprops = dict(arrowstyle='->', shrinkA=0, shrinkB=0, connectionstyle='arc3')
    for lab, data_id in G.edges():
        ax.annotate(lab, xy=mod_pos_dict[data_id], xytext=mod_pos_dict[lab], xycoords='data', textcoords='data', arrowprops=arrowprops, fontsize=12)
    
def annotated_scatters(ax, xs, ys, sub_idxs, labs=None, k=0.01):
    ax.scatter(xs, ys, s=1, c='gray')
    if sub_idxs.shape[0] > 0:
        ax.scatter(xs[sub_idxs], ys[sub_idxs], s=10, c='red')
        if not labs is None:
            sub_labs = labs[sub_idxs]
        else:
            sub_labs = sub_idxs
        repel_labels(ax, xs[sub_idxs], ys[sub_idxs], sub_labs, k=k)


def annotated_bars(ax, labels, values):
    xs = np.arange(len(values))
    ax.bar(xs, values)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right')

def annoted_hbars(ax, labels, values):
    ys = np.arange(len(labels))
    ax.barh(ys, values)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

def grid_dyn_visualize(ddiff_embed, u_mat, ax):
    ddiff_embed /= 5 * np.linalg.norm(ddiff_embed, axis=1).mean()
    grids, grid_flow = commons.calc_grid_flow(u_mat, ddiff_embed, 15)
    ax.scatter(u_mat[:, 0], u_mat[:, 1], s=1, c='lightgray')
    ax.quiver(grids[:, 0], grids[:, 1], grid_flow[:, 0], grid_flow[:, 1], color='black', scale=1.5)

def diff_visualize(ddiff_embed, u_mat, ax):
    ddiff_embed /= np.linalg.norm(ddiff_embed, axis=1).max()
    grids, grid_flow = commons.calc_grid_flow(u_mat, ddiff_embed, 15)
    ax.scatter(u_mat[:, 0], u_mat[:, 1], s=1, c='lightgray')
    ax.quiver(grids[:, 0], grids[:, 1], grid_flow[:, 0], grid_flow[:, 1], color='red', scale=1.5)
    ax.quiver(grids[:, 0], grids[:, 1], -grid_flow[:, 0], -grid_flow[:, 1], color='blue', scale=1.5)

def diff_visualize_scale(ddiff_embed, u_mat, scales, ax):
    scales = scales / np.abs(scales).max()
    ddiff_embed *= (scales / np.linalg.norm(ddiff_embed, axis=1)).reshape(-1, 1)
    grids, grid_flow = commons.calc_grid_flow(u_mat, ddiff_embed, 15)
    grids, grid_scales = commons.calc_grid_flow(u_mat, scales.reshape(-1, 1), 15)
    grid_scales = grid_scales.reshape(-1)
    max_val = np.abs(grid_scales).max()
    norm = colors.Normalize(vmax=max_val, vmin=-max_val)
    # norm.autoscale(grid_scales)
    cmap = cm.bwr
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.scatter(u_mat[:, 0], u_mat[:, 1], s=1, c='lightgray')
    ax.quiver(grids[:, 0], grids[:, 1], grid_flow[:, 0], grid_flow[:, 1], scale=1.5, color=cmap(norm(grid_scales)))
    # ax.colorbar(sm)
    return sm

def plot_grouped_box(df, group_col, ax=None):
    import seaborn as sns

    # Assuming df is the given data frame and groupby_col is the column based on which we want to group
    sns.set_style('whitegrid')
    sns.boxplot(data=df.reset_index().drop(columns=[groupby_col]), x='index', hue=groupby_col, palette="Set3") # drop the grouping column and set x-axis as index

    # Assuming the data is stored in a dataframe called 'df' with columns 'group_col', 'col1', 'col2', ...
    # 'group_col' is the column used for grouping in the boxplot

    # Create a list of columns to plot
    columns_to_plot = df.columns.tolist()
    columns_to_plot.remove(group_col)

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    df.boxplot(column=columns_to_plot, by=group_col, ax=ax)


def get_inter_cluster_edges(est_adata, obs_key, rep, nn=15):
    z = est_adata.obsm[rep]
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(z)
    distances, indices = nbrs.kneighbors(z)
    source = np.repeat(np.arange(est_adata.shape[0]), nn)
    target = indices.reshape(-1)
    cl_vec = est_adata.obs[obs_key].values
    diff_ind = cl_vec[source] != cl_vec[target]
    return source[diff_ind], target[diff_ind]

    
def plot_edges_on_umap(est_adata, sources, targets, ax, vis_rep='X_umap', lw=0.1, alpha=0.1):
    rep_mat = est_adata.obsm[vis_rep]
    for source, target in zip(sources, targets):
        ax.plot(rep_mat[[source, target], 0], rep_mat[[source, target], 1], lw=lw, c='k', alpha=alpha)
    

def plot_inter_cluster_edges(est_adata, obs_key, ax, rep='X_vicdyf_zl', vis_rep='X_umap', nn=15):
    sources, targets = get_inter_cluster_edges(est_adata, obs_key, rep, nn=nn)
    rep_mat = est_adata.obsm[vis_rep]
    plot_edges_on_umap(est_adata, sources, targets, ax, vis_rep=vis_rep)
    

def plot_mean_flow(adata, cluster_key='leiden', legend_loc='on data', vel_key='norm_vicdyf_mean_velocity', du_key='X_vicdyf_mdumap', ax=None):
    m_coeff = 0.5 
    if vel_key in adata.layers.keys():
        vel_vec = np.linalg.norm(adata.layers[vel_key], axis=1, keepdims=True)
    else:
        vel_vec = adata.obs[vel_key].values.reshape(-1, 1)
    vel_vec = vel_vec / vel_vec.mean()
    adata.obsm[du_key] = vel_vec * adata.obsm[du_key] / np.linalg.norm(adata.obsm[du_key], axis=1, keepdims=True)
    arrow_ratio = 10
    scv.pl.velocity_embedding_grid(adata, X=adata.obsm['X_vicdyf_umap'], V=adata.obsm[du_key] * m_coeff, color=cluster_key, show=False, basis='X_vicdyf_umap', density=0.6, headwidth=arrow_ratio, headlength=arrow_ratio, headaxislength=arrow_ratio, width=0.002, arrow_length=3, arrow_color='black', alpha=0.5, legend_loc=legend_loc, sort_order=False, ax=ax)


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

