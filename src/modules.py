from turtle import forward
import torchsde
import math
from shutil import unregister_unpack_format
from tkinter import W
# from jax import jacrev
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F, norm, normal
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np
try:
    from funcs import calc_kld, calc_nb_loss, calc_poisson_loss, normalize_lp
except:
    from .funcs import calc_kld, calc_nb_loss, calc_poisson_loss, normalize_lp
import pytorch_lightning as pl
from functorch import jvp, vmap, jacfwd, grad, vjp, jacrev
from einops import rearrange, repeat

class LinearReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.GELU())

    def forward(self, x):
        h = self.f(x)
        return(h)


class LinearReLURes(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLURes, self).__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.GELU(), 
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.GELU(), 
            nn.Linear(input_dim, output_dim))

    def forward(self, x):
        dh = self.f(x)
        h = x + dh
        return(h)


class SeqNN(nn.Module):
    def __init__(self, num_steps, dim):
        super(SeqNN, self).__init__()
        modules = [
            LinearReLU(dim, dim)
            for _ in range(num_steps)
        ]
        self.f = nn.Sequential(*modules)

    def forward(self, pre_h):
        post_h = self.f(pre_h)
        return(post_h)


class Encoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim, enc_dist=dist.Normal, norm_input=False, time_coeff=1.0):
        super(Encoder, self).__init__()
        self.norm_input = norm_input
        if norm_input:
            x_dim = x_dim + 1
        self.x2h = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)
        self.softplus = nn.Softplus()
        self.time_coeff = time_coeff
        self.dist = enc_dist

    def forward(self, x):
        if self.norm_input:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x / x_mean
            x = torch.cat([x, x_mean], dim=-1)
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        qz = self.dist(mu, self.softplus(logvar))
        z = qz.rsample()
        return(z, qz)


class EncoderSDE(Encoder):
    def __init__(self, *args, **kwargs):
        super(EncoderSDE, self).__init__(*args, **kwargs)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.time_coeff = 1.0

    def f(self, t, z):
        pre_h = self.x2h(z)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        return mu * self.time_coeff

    def g(self, t, z):
        pre_h = self.x2h(z)
        post_h = self.seq_nn(pre_h)
        logvar = self.h2logvar(post_h)
        return logvar * math.sqrt(self.time_coeff)

class GumbelSoftmax:
    def __init__(self, lrho, ref_z, temp):
        self.lrho = normalize_lp(lrho)
        self.data_size = lrho.size()[0]
        self.ref_z = ref_z
        self.n_comps = self.lrho.size()[1]
        self.dist = dist.Gumbel(torch.zeros_like(lrho), torch.ones_like(lrho))
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)
        self.rho = self.softmax(self.lrho)

    def make_values(self, g):
        soft_k = self.softmax((g + self.lrho) / self.temp)
        # amax_k = torch.nn.functional.one_hot(soft_k.argmax(dim=-1), num_classes=g.size()[-1])
        # k = soft_k + (amax_k - soft_k).detach()
        z = (soft_k.unsqueeze(-1) * self.ref_z).sum(dim=-2)
        return z

    def sample(self, num=None):
        if num == None:
            g = self.dist.sample()
        else:
            g = self.dist.sample(num)
        z = self.make_values(g)
        return z

    def rsample(self, num=None):
        if num == None:
            g = self.dist.sample()
        else:
            g = self.dist.sample(num)
        z = self.make_values(g)
        return z

    @property
    def loc(self):
        z = (self.rho.unsqueeze(-1) * self.ref_z).sum(dim=-2)
        return z


class DecreasingTemp:
    def __init__(self, dec_temp_steps=300, init_temp=1000, last_temp=1):
        self.temp = init_temp
        self.last_temp = last_temp
        self.dec_temp_rate = math.pow(last_temp / init_temp, 1.0 / dec_temp_steps)

    def step(self):
        self.temp = max(self.dec_temp_rate * self.temp, self.last_temp)

    @property
    def value(self):
        return self.temp


class DiscreteEncoder(nn.Module):
    def __init__(self, *args, init_temp=1, last_temp=0.5, dec_temp_steps=500, **kwargs):
        super(DiscreteEncoder, self).__init__()
        self.enc = Encoder(*args, **kwargs)
        self.softmax = nn.Softmax(dim=-1)
        self.temp = init_temp
        self.last_temp = 0.5
        self.dec_temp_rate = math.pow(last_temp / init_temp, 1.0 / dec_temp_steps)

    def forward(self, x, ref_z):
        raw_z, q_raw_z = self.enc(x)
        lrho_k = q_raw_z.log_prob(ref_z).sum(dim=-1).permute(1, 0)
        qz = GumbelSoftmax(lrho_k, ref_z.permute(1, 0, 2), self.temp)
        z = qz.rsample()
        return z, qz

    def decrease_temp(self):
        self.temp = max(self.dec_temp_rate * self.temp, self.last_temp)

        

def apply_with_batch(f, x, s):
    xs = torch.cat([x, s], dim=-1)
    out = f(xs)
    return out


class EncoderBatch(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim, batch_num, enc_dist=dist.Normal, use_mmd_loss=False, use_batch_bias=False, use_ambient=False, norm_input=False):
        super(EncoderBatch, self).__init__()
        self.norm_input = norm_input
        if use_ambient:
            self.encode_z = Encoder(num_h_layers, x_dim, h_dim, z_dim, enc_dist=dist.Normal, norm_input=norm_input)
            self.main = self.encode_no_batch_z
        else:
            self.encode_z = Encoder(num_h_layers, x_dim + batch_num, h_dim, z_dim, enc_dist=dist.Normal)
            self.main = self.encode_batch_z_concat

    def encode_batch_z_concat(self, x, s):
        z, qz = apply_with_batch(self.encode_z, x, s)
        return z, qz

    def encode_no_batch_z(self, x, s):
        z, qz = self.encode_z(x)
        return z, qz

    def forward(self, x, s):
        z, qz = self.main(x, s)
        return(z, qz)


class DecoderBatch(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim, batch_num, use_mmd_loss=False, use_batch_bias=False, use_ambient=False):
        super(DecoderBatch, self).__init__()
        if (use_batch_bias or use_ambient):
            self.decode_z = Decoder(num_h_layers, z_dim, h_dim, x_dim)
            self.main = self.decode_no_batch_z
        elif use_mmd_loss:
            self.dec1 = nn.Linear(z_dim + batch_num, h_dim)
            self.dec2 = Decoder(num_h_layers, h_dim, h_dim, x_dim)
            self.decode_z = nn.Sequential(
                self.dec1,
                self.dec2
            )
        else:
            self.decode_z = Decoder(num_h_layers, z_dim + batch_num, h_dim, x_dim)
        self.batch_bias_dict = nn.Embedding(batch_num, z_dim)
        if use_batch_bias:
            self.main = self.decode_batch_z_add
        elif use_ambient:
            self.main = self.decode_no_batch_z
        else:
            self.main = self.decode_batch_z_concat

    def decode_no_batch_z(self, z, s):
        px_z_ld = self.decode_z(z) 
        return px_z_ld
        
    def decode_batch_z_add(self, z, s):
        zs = z + self.batch_bias_dict(s.argmax(dim=-1))
        px_z_ld = self.decode_z(zs)
        return px_z_ld
        
    def decode_batch_z_concat(self, z, s):
        px_z_ld = apply_with_batch(self.decode_z, z, s)
        return px_z_ld
        
    def forward(self, x, s):
        px_z_ld = self.main(x, s)
        return(px_z_ld)


class Energy(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(Energy, self).__init__()
        self.x2h = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2e = nn.Linear(h_dim, 1)

    def forward(self, x):
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        e = self.h2e(post_h)
        return e


class DynEncoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(DynEncoder, self).__init__()
        self.x2e = nn.Sequential(
            LinearReLU(x_dim, h_dim),
            SeqNN(num_h_layers - 1, h_dim),
            nn.Linear(h_dim, 1)
        )
        self.x2logvar = nn.Sequential(
            LinearReLU(x_dim, h_dim),
            SeqNN(num_h_layers - 1, h_dim),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

    def forward(self, x):
        logvar = self.x2logvar(x)
        f_e = lambda vz: self.x2e(vz)
        mu = vmap(jacrev(f_e), in_dims=(0,))(x)
        mu = mu.squeeze(1)
        qd = dist.Normal(mu, logvar)
        d = qd.rsample()
        return(d, qd)


class Decoder(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        self.z2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        pre_h = self.z2h(z)
        post_h = self.seq_nn(pre_h)
        ld = self.h2ld(post_h)
        correct_ld = self.softplus(ld)
        normalize_ld = correct_ld  / correct_ld.mean(dim=-1, keepdim=True)
        return(normalize_ld)

def max_normalize(d, val):
    d_len = torch.linalg.norm(d, dim=-1, keepdim=True)
    coeff = val * torch.tanh(d_len) / d_len
    return  coeff * d

class EnvDyn(pl.LightningModule):
    def __init__(
            self,
            x_dim, z_dim,
            enc_z_h_dim, enc_d_h_dim, dec_z_h_dim,
            num_enc_z_layers, num_enc_d_layers,
            num_dec_z_layers, norm_input=False, edyn=False, use_vamp=False, vamp_comps=128, fixt=False, no_d_kld=False, decreasing_temp=False, loss_mode='nb', dec_temp_steps=30000, dyn_loss_weight=0.0, **kwargs):
        super(EnvDyn, self).__init__()
        self.enc_z = Encoder(num_enc_z_layers, x_dim, enc_z_h_dim, z_dim, norm_input=norm_input)
        if edyn:
            self.enc_d = DynEncoder(num_enc_d_layers, z_dim, enc_d_h_dim, z_dim)
        else:
            self.enc_d = Encoder(num_enc_d_layers, z_dim, enc_d_h_dim, z_dim)
        self.enc_l = Encoder(num_enc_z_layers, x_dim, enc_z_h_dim, 1, enc_dist=dist.LogNormal)
        self.enc_ul = Encoder(num_enc_z_layers, x_dim, enc_z_h_dim, 1, enc_dist=dist.LogNormal)
        self.dec_z = Decoder(num_enc_z_layers, z_dim, dec_z_h_dim, x_dim)
        self.dt = 1
        self.gamma_mean = 0.05
        self.maxgamma = 1
        self.maxbeta = 0.1
        self.d_coeff = 0.1
        self.loggamma = Parameter(torch.Tensor(x_dim))
        self.logbeta = Parameter(torch.Tensor(x_dim))
        self.logstheta = Parameter(torch.Tensor(x_dim))
        self.logutheta = Parameter(torch.Tensor(x_dim))
        self.l_mu = Parameter(torch.Tensor(1))
        self.l_logvar = Parameter(torch.Tensor(1))
        self.ul_mu = Parameter(torch.Tensor(1))
        self.ul_logvar = Parameter(torch.Tensor(1))
        self.softplus = nn.Softplus()
        self.softplus_kinetics = nn.Softplus(beta=20.0)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.fixt = fixt
        self.no_d = False
        self.no_lu = False
        self.no_z_kld = False
        self.dec_temp_mode = decreasing_temp
        self.no_d_kld = no_d_kld
        self.loss_mode = loss_mode
        self.norm_input = norm_input
        self.relu = nn.ReLU()
        self.dyn_loss_weight = dyn_loss_weight
        self.use_vamp = use_vamp
        self.dec_temp = DecreasingTemp(dec_temp_steps=dec_temp_steps)
        self.raw_p = Parameter(torch.zeros(vamp_comps))
        self.pz_u = Parameter(torch.Tensor(vamp_comps, x_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.loggamma)
        init.normal_(self.logbeta)
        init.normal_(self.logstheta)
        init.normal_(self.logutheta)
        init.normal_(self.l_logvar)
        init.normal_(self.l_mu)
        init.normal_(self.ul_logvar)
        init.normal_(self.ul_mu)
        init.normal_(self.raw_p) 
        init.normal_(self.pz_u) 
        
    def encode_l(self, batch):
        x, u, snorm_mat, unorm_mat, *others = batch
        l, ql = self.enc_l(x)
        return l, ql

    def encode_z_spec(self, sx, batch):
        x, u, snorm_mat, unorm_mat, *others = batch
        z, qz = self.enc_z(sx)
        return z, qz

    def encode_z(self, batch):
        x, u, snorm_mat, unorm_mat, *others = batch
        return self.encode_z_spec(x, batch)

    def decode_z(self, z, batch):
        px_ld = self.dec_z(z)
        return px_ld

    def decode_z_multi(self, z, batch):
        return self.decode_z(z, batch)

    def encode_d(self, z, batch):
        d, qd = self.enc_d(z)
        return d, qd

    def calculate_kinetics(self, diff_px_zd_ld, px_z_ld, batch):
        gamma = self.softplus(self.loggamma) * self.dt
        # gamma = self.maxgamma *  self.sigmoid((self.loggamma)) * self.dt
        beta = self.softplus((self.logbeta)) * self.dt
        # normalize beta mean to 0.01
        raw_u_ld = diff_px_zd_ld * beta + px_z_ld * gamma
        pu_zd_ld = self.softplus_kinetics(raw_u_ld)
        # pu_zd_ld = self.softplus(raw_u_ld)
        # deviation
        # dyn_loss = 0 # self.relu(- diff_px_zd_ld * beta - px_z_ld * gamma).sum(dim=1).mean(dim=0)
        # kinetics are set to maximizing stable state
        # dyn_loss = torch.norm(raw_u_ld.detach()  - px_z_ld.detach() * gamma, dim=1, p=1).mean(dim=0)
        dyn_loss = 0
        return(pu_zd_ld, dyn_loss)

    def calculate_diff_x(self, z, d, batch):
        zd = z + self.d_coeff * d
        zmd = z - self.d_coeff * d
        diff_px_zd_ld = self.decode_z(zd, batch) - self.decode_z(zmd, batch)
        return diff_px_zd_ld

    def calculate_diff_x_grad(self, z, d, batch):
        dec_f = lambda vz: self.decode_z(vz, batch)
        dec_jvp = lambda vz, vd: jvp(dec_f, (vz, ), (vd, ))[1]
        diff_px_zd_ld = self.d_coeff * vmap(dec_jvp, in_dims=(0, 0))(z, d)
        return diff_px_zd_ld

    def calculate_umean(self, z, d, px_z_ld, batch):
        # deconst z + d
        if self.no_d:
            diff_px_zd_ld = torch.zeros_like(px_z_ld)
        else:
            diff_px_zd_ld = self.calculate_diff_x_grad(z, d, batch)
        pu_zd_ld, dyn_loss = self.calculate_kinetics(diff_px_zd_ld, px_z_ld, batch)
        return(pu_zd_ld, dyn_loss)
        
    def forward(self, batch):
        z, qz = self.encode_z(batch)
        # decode z
        px_z_ld = self.decode_z(z, batch)
        d, qd = self.encode_d(z, batch)
        pu_zd_ld, dyn_loss = self.calculate_umean(z, d, px_z_ld, batch)
        l, ql = self.encode_l(batch)
        l = ql.rsample()
        ul = l
        return(z, d, qz, qd, px_z_ld, pu_zd_ld, l, ql, ul, ql, dyn_loss)

    def log_pz_vamp(self, z, raw_p, batch):
        p = self.softmax(raw_p)
        uz, quz = self.encode_z_spec(self.pz_u, batch)
        log_pz_k = p.log() + quz.log_prob(z.unsqueeze(-2)).sum(dim=-1) 
        log_pz = log_pz_k.logsumexp(dim=-1)
        return log_pz

    def calc_z_kld_vamp(self, qz, raw_p, batch):
        # kld of pz and qz
        z = qz.rsample()
        log_pz = self.log_pz_vamp(z, raw_p, batch)
        hqz = qz.entropy().sum(dim=-1)
        z_kld = - log_pz - hqz
        return z_kld
        
    def calc_z_kld_unimodal(self, qz, batch):
        # kld of pz and qz
        z_kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
        z_kld = z_kld.sum(dim=-1)
        return z_kld

    def calc_z_kld(self, qz, batch):
        if self.use_vamp:
            z_kld = self.calc_z_kld_vamp(qz, self.raw_p, batch)
        else:
            z_kld = self.calc_z_kld_unimodal(qz, batch)
        return z_kld

    def calc_d_kld(self, qd, batch):
        d_kld = -0.5 * (1 + qd.scale.pow(2).log() - qd.loc.pow(2) - qd.scale.pow(2))
        d_kld = d_kld.sum(dim=-1)
        return d_kld

    def calc_l_kld(self, ql, batch):
        pl = dist.LogNormal(torch.zeros_like(ql.loc), self.softplus(self.l_logvar))
        l_kld = kl_divergence(ql, pl)
        l_kld = l_kld.sum(dim=-1)
        return l_kld

    def calc_kld(self, qz, qd, ql, qul, batch):
        # kld of pz and qz
        z_kld = self.calc_z_kld(qz, batch)
        # aggregate kld
        if self.no_d_kld:
            d_kld = torch.zeros_like(z_kld)
        else:
            d_kld = self.calc_d_kld(qd, batch)
        kld = z_kld  + d_kld
        # kld for size factor
        l_kld = self.calc_l_kld(ql, batch)
        return kld, l_kld 
        
    def calc_reconst_loss(self, px_z_ld, l, pu_zd_ld, ul, batch):
        x, u, snorm_mat, unorm_mat, *others = batch
        if self.loss_mode == 'nb':
            loss_func = lambda ld, norm_mat, obs: calc_nb_loss(ld, norm_mat, self.softplus(self.logstheta), obs).sum(dim=-1)
        else:
            loss_func = lambda ld, norm_mat, obs: calc_poisson_loss(ld, norm_mat, obs).sum(dim=-1)
        # reconst loss of unspliced x
        lx = loss_func(px_z_ld * l, snorm_mat, x)
        if self.no_lu:
            lu = torch.zeros_like(lx)
        else:
            lu = loss_func(pu_zd_ld * ul, unorm_mat, u)
        if self.dec_temp_mode and self.training:
            self.dec_temp.step()
            lu = lu / self.dec_temp.value
        return lx, lu

    def elbo_loss(self, batch):
        z, d, qz, qd, px_z_ld, pu_zd_ld, l, ql, ul, qul, dyn_loss = self(batch)
        kld, l_kld = self.calc_kld(qz, qd, ql, qul, batch)
        lx, lu = self.calc_reconst_loss(px_z_ld, l, pu_zd_ld, ul, batch)
        elbo_loss = torch.mean(kld + lx + lu + l_kld)
        loss_dict = {
            "elbo_loss": elbo_loss,
            "dyn_loss": dyn_loss * self.dyn_loss_weight
        }
        return(loss_dict)
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.elbo_loss(batch)
        total_loss = sum([l for l in loss_dict.values()])
        self.log('train_loss', total_loss)
        return(total_loss)
            
    def validation_step(self, batch, batch_idx):
        loss_dict = self.elbo_loss(batch)
        total_loss = sum([l for l in loss_dict.values()])
        self.log('val_loss', total_loss)
        for key, value in loss_dict.items():
            self.log(key, value)
        return(total_loss)
            
    def configure_optimizers(self, lr=0.001):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        lr_schedulers = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5), 'monitor': 'val_loss'}
        return([optimizer], [lr_schedulers])

    def vae_mode(self):
        self.no_d_kld = True
        self.no_d = True
        self.no_lu = True
        for param in self.enc_d.parameters():
            param.requires_grad = False
        for param in self.dec_z.parameters():
            param.requires_grad = True
        for param in self.enc_z.parameters():
            param.requires_grad = True

    def dyn_mode(self):
        # self.no_d_kld = False
        self.no_d = False
        self.no_lu = False
        for param in self.enc_d.parameters():
            param.requires_grad = True
        for param in self.dec_z.parameters():
            param.requires_grad = False
        for param in self.enc_z.parameters():
            param.requires_grad = False
        for param in self.enc_l.parameters():
            param.requires_grad = False

    def calc_mean_std(self, z, qd, batch, sample_num=50):
        cell_num = z.shape[0]
        gene_num = z.shape[1]
        dec_f = lambda vz, vs: self.dec_z(vz, vs)
        v_f_d = lambda vz, vd, vs: jvp(lambda vzz: dec_f(vzz, vs))(vz, vd)
        d_batch = rearrange(qd.sample((sample_num,)), 'b c h -> (b c) h')
        z_batch = repeat(z, 'c h -> (b c) h', b=sample_num)
        px_ld_batch = rearrange(self.calculate_diff_x_grad(z_batch, d_batch, batch), '(b c) g -> b c g', b=sample_num)
        batch_std_mat = torch.std(px_ld_batch, dim=0)
        batch_mean_mat = torch.mean(px_ld_batch, dim=0)
        return batch_mean_mat, batch_std_mat

    def encode_size(self, batch):
        res = batch
        sl, qsl = self.encode_l(batch)
        sll = qsl.loc
        ull = qsl.loc
        return sll, ull

    def extract_info(self, batch, sample_num=50):
        # z, zl, d, dl, px_z_ld, pu_zd_ld, sll, ull, gene_vel, mean_gene_vel, batch_std_mat
        z, qz, d, qd, px_z_ld, pu_zd_ld, gene_vel, gene_vel_mean, gene_vel_std = self.mean_forward_batch(batch)
        return z, qz, d, qd, px_z_ld, pu_zd_ld, gene_vel, gene_vel_mean, gene_vel_std

    def calculate_diff_x_std(self, z, dscale, batch):
        d_id = torch.eye(z.size()[1], z.size()[1]).to(z.device)
        gene_vel_std = sum([
            (self.calculate_diff_x_grad(z, repeat(delta_d, 'd -> b d', b=z.size()[0]), batch) * dscale[:, i].unsqueeze(1))**2 
            for delta_d, i in zip(d_id, range(dscale.size()[1]))]).sqrt()
        return gene_vel_std

    def mean_forward_batch(self, batch):
        z, qz = self.encode_z(batch)
        zl = qz.loc
        d, qd = self.encode_d(zl, batch)
        dl = qd.loc
        px_z_ld = self.decode_z(zl, batch)
        pu_zd_ld, dyn_loss = self.calculate_umean(zl, dl, px_z_ld, batch)
        # px_zd_ld = self.decode_z(zl + self.d_coeff * d, batch)
        gene_vel = self.calculate_diff_x_grad(zl, d, batch) 
        gene_vel_mean = self.calculate_diff_x_grad(zl, dl, batch) 
        gene_vel_std = self.calculate_diff_x_std(zl, qd.scale, batch) 
        return z, qz, d, qd, px_z_ld, pu_zd_ld, gene_vel, gene_vel_mean, gene_vel_std
        

    

class VicDyfMb(EnvDyn):
    def __init__(
            self,
            x_dim, z_dim,
            enc_z_h_dim, enc_d_h_dim, dec_z_h_dim,
            num_enc_z_layers, num_enc_d_layers,
            num_dec_z_layers, batch_num, norm_input=False, use_mmd_loss=False, use_batch_bias=False, use_ambient=False, use_vamp=False, vamp_comps=64, **kwargs):
        super(VicDyfMb, self).__init__(
            x_dim, z_dim,
            enc_z_h_dim, enc_d_h_dim, dec_z_h_dim,
            num_enc_z_layers, num_enc_d_layers,
            num_dec_z_layers, norm_input=norm_input, use_vamp=use_vamp, vamp_comps=vamp_comps, **kwargs)
        self.enc_l = Encoder(num_enc_z_layers, x_dim + batch_num,  enc_z_h_dim, 1, enc_dist=dist.LogNormal)
        self.enc_ul = Encoder(num_enc_z_layers, x_dim + batch_num, enc_z_h_dim, 1, enc_dist=dist.LogNormal)
        if use_mmd_loss:
            self.mmd_loss_weight = 10
        else:
            self.mmd_loss_weight = 0
        self.enc_z = EncoderBatch(num_enc_z_layers, x_dim, enc_z_h_dim, z_dim, batch_num, use_mmd_loss=use_mmd_loss, use_batch_bias=use_batch_bias, use_ambient=use_ambient, norm_input=norm_input)
        self.dec_z = DecoderBatch(num_dec_z_layers, z_dim, dec_z_h_dim, x_dim, batch_num, use_mmd_loss=use_mmd_loss, use_batch_bias=use_batch_bias, use_ambient=use_ambient)
        if use_ambient:
            self.s_obs_coeffs = nn.Embedding(batch_num, x_dim)
            self.s_ambients = nn.Embedding(batch_num, x_dim)
            self.u_obs_coeffs = nn.Embedding(batch_num, x_dim)
            self.u_ambients = nn.Embedding(batch_num, x_dim)
            self.epsilons = nn.Embedding(batch_num, 1)
        self.use_batch_bias = use_batch_bias
        self.use_mmd_loss = use_mmd_loss
        self.use_ambient = use_ambient
        self.register_buffer('tau', 1 / torch.tensor([0.5, 0.1, 0.05, 0.01]))
        pseudo_batch = torch.zeros(vamp_comps, batch_num)
        pseudo_batch[:, 0] = 1
        self.register_buffer('pseudo_batch', pseudo_batch)
        self.reset_parameters()

    def encode_l(self, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        sl, qsl = apply_with_batch(self.enc_l, x, s)
        return sl, qsl

    def encode_z_spec(self, sx, batch):
        z, qz = self.enc_z(sx, self.pseudo_batch)
        return z, qz

    def encode_z(self, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        z, qz = self.enc_z(x, s)
        return z, qz

    def decode_z(self, z, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        px_ld = self.dec_z(z, s)
        return px_ld

    def decode_z_multi(self, z, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        px_ld = self.dec_z(z, s.repeat(z.size()[0], 1, 1))
        return px_ld

    def calculate_diff_x_grad(self, z, d, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        b = int(z.size()[0] / s.size()[0])
        if b > 1:
            s = repeat(s, 'c s -> (b c) s', b=b)
        dec_f = lambda vz, vs: self.dec_z(vz, vs)
        dec_jvp = lambda vz, vd, vs: jvp(lambda vzz: dec_f(vzz, vs), (vz, ), (vd, ))[1]
        diff_px_zd_ld = vmap(dec_jvp, in_dims=(0, 0, 0))(z, d, s)
        return diff_px_zd_ld

    def mmd_loss_each(self, z, s, counts, i, j):
        ni = counts[i]
        nj = counts[j]
        if ni * nj > 0:
            zi = z[s[:, i] > 0]
            zj = z[s[:, j] > 0]
            dij = torch.linalg.norm(zi.unsqueeze(0) - zj.unsqueeze(1), dim=2, keepdim=True)
            rij2 = - (self.softplus(self.tau) * dij)**2
            mmd_each = torch.exp(rij2).sum() / (ni * nj)
        else:
            mmd_each = 0
        return mmd_each 

    def mmd_loss(self, z, batch):
        x, u, snorm_mat, unorm_mat, s = batch
        dh1 = apply_with_batch(self.dec_z.dec1, z, s)
        s_counts = s.sum(dim=0)
        mmd_loss = 0
        for i in range(s.size()[1]):
            mmd_loss += self.mmd_loss_each(dh1, s, s_counts, i, i) * ((s_counts > 0).sum() - 1)
            for j in range(i + 1, s.size()[1]):
                mmd_loss -= 2 * self.mmd_loss_each(z, s, s_counts, i, j)
        return mmd_loss

    def add_ambient(self, px_z_ld, pu_zd_ld, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        sidx = s.argmax(dim=1)
        epsilons = self.sigmoid(self.epsilons(sidx))
        px_z_ld = (1 - epsilons) * (px_z_ld * self.softplus(self.s_obs_coeffs(sidx))) + epsilons * self.softplus(self.s_ambients(sidx))
        pu_zd_ld = (1 - epsilons) * (pu_zd_ld * self.softplus(self.u_obs_coeffs(sidx))) + epsilons * self.softplus(self.u_ambients(sidx))
        return px_z_ld, pu_zd_ld

    def calc_z_kld(self, qz, batch):
        x, u, snorm_mat, unorm_mat, s, *others = batch
        if self.use_vamp:
            sidx = s.argmax(dim=1)
            z_kld = self.calc_z_kld_vamp(qz, self.raw_p, batch)
        else:
            z_kld = self.calc_z_kld_unimodal(qz, batch)
        return z_kld

    def elbo_loss(self, batch):
        z, d, qz, qd, px_z_ld, pu_zd_ld, l, ql, ul, qul, dyn_loss = self(batch)
        if self.use_ambient:
            px_z_ld, pu_zd_ld = self.add_ambient(px_z_ld, pu_zd_ld, batch)
        kld, l_kld = self.calc_kld(qz, qd, ql, qul, batch)
        lx, lu = self.calc_reconst_loss(px_z_ld, l, pu_zd_ld, ul, batch)
        elbo_loss = torch.mean(kld + lx + lu + l_kld)
        loss_dict = {
            "elbo": elbo_loss,
            "dyn_loss": dyn_loss * self.dyn_loss_weight
        }
        return(loss_dict)
 


class Cvicdyf(VicDyfMb):
    def __init__(self, **kwargs):
        super(Cvicdyf, self).__init__(**kwargs)
        self.enc_d = Encoder(kwargs['num_enc_d_layers'], kwargs['z_dim'] + kwargs['c_dim'], kwargs['enc_d_h_dim'], kwargs['z_dim'])
        if kwargs.get('dyn_mode', 'stochastic') == 'deterministic':
            self.no_d_kld = True

    def encode_d(self, z, batch):
        x, u, snorm_mat, unorm_mat, s, t, *others = batch
        d, qd = apply_with_batch(self.enc_d, z, t)
        return d, qd

class CvicdyfMultiKinetics(Cvicdyf):
    def __init__(self, **kwargs):
        super(CvicdyfMultiKinetics, self).__init__(**kwargs)
        self.loggamma = Parameter(torch.Tensor(kwargs['class_num'], kwargs['x_dim']))
        self.logbeta = Parameter(torch.Tensor(kwargs['class_num'], kwargs['x_dim']))
        init.normal_(self.loggamma)
        init.normal_(self.logbeta)

    def calculate_kinetics(self, diff_px_zd_ld, px_z_ld, batch):
        x, u, snorm_mat, unorm_mat, s, t, c, *others = batch
        gamma = self.softplus((self.loggamma * c.unsqueeze(-1)).sum(-2)) * self.dt
        # gamma = self.maxgamma *  self.sigmoid((self.loggamma)) * self.dt
        beta = self.softplus((self.logbeta * c.unsqueeze(-1)).sum(-2)) * self.dt
        # normalize beta mean to 0.01
        raw_u_ld = diff_px_zd_ld * beta + px_z_ld * gamma
        pu_zd_ld = self.softplus_kinetics(raw_u_ld)
        # kinetics are set to maximizing stable state
        dyn_loss = torch.norm(raw_u_ld.detach()  - px_z_ld.detach() * gamma, dim=1, p=1).mean(dim=0)
        return(pu_zd_ld, dyn_loss)

    def encode_d(self, z, batch):
        x, u, snorm_mat, unorm_mat, s, t, *others = batch
        d, qd = apply_with_batch(self.enc_d, z, t)
        return d, qd

    # def calculate_diff_x_grad(self, z, d, batch):
    #     dec_f = lambda vz: self.decode_z(vz, batch)
    #     dec_jvp = lambda vz, vd: jvp(dec_f, (vz, ), (vd, ))[1]
    #     diff_px_zd_ld = self.d_coeff * vmap(dec_jvp, in_dims=(0, 0))(z, d)
    #     return diff_px_zd_ld

class VicDyfP(EnvDyn):
    def __init__(self, **kwargs):
        super(VicDyfP, self).__init__(**kwargs)
        self.enc_d = EncoderSDE(kwargs['num_enc_d_layers'], kwargs['z_dim'], kwargs['enc_d_h_dim'], kwargs['z_dim'])
        self.softmax_qz = nn.Softmax(dim=1)

    def calc_z_kld(self, qz, batch):
        if self.use_vamp:
            z_kld = self.calc_z_kld_vamp(qz, self.raw_p, batch)
        else:
            z_kld = self.calc_z_kld_unimodal(qz, batch)
        ts = torch.linspace(0, 0.1, 2)
        z = qz.rsample()
        pred_z = torchsde.sdeint(self.enc_d, z, ts, method='euler')[-1].unsqueeze(1)
        lp = torch.logsumexp(qz.log_prob(pred_z).sum(dim=-1) - math.log(z.size()[0]), dim=1)
        return - lp + z_kld

class TFVicdyf(EnvDyn):
    def __init__(self, **kwargs):
        super(TFVicdyf, self).__init__(**kwargs)
        self.tf_idxs = kwargs['tf_idxs']
        self.softmax_qz = nn.Softmax(dim=1)
        self.A = Parameter(torch.zeros(kwargs['x_dim'], kwargs['tf_idxs'].shape[0]))
        self.b = Parameter(torch.zeros(kwargs['x_dim']))
        self.max_alpha = Parameter(torch.zeros(kwargs['x_dim']))
        self.vloss_coeff = 1
        self.asp_coeff = 100
        self.cos = nn.CosineSimilarity(dim=-1)
        self.l1_loss = nn.L1Loss()
        self.no_d_kld = False
        self.stable =False
        self._v_scale = Parameter(torch.zeros(kwargs['x_dim']))
        self.reset_parameters_add()

    def reset_parameters_add(self):
        init.normal(self.max_alpha)
        init.normal(self.A)
        init.normal(self.b)
        init.normal(self._v_scale)
    
    @property
    def gamma(self):
        return self.softplus(self.loggamma)

    @property
    def v_scale(self):
        return self.softplus(self._v_scale)

    def calculate_diff_x_model(self, x_hat):
        tf_x = x_hat[..., self.tf_idxs]
        act = torch.einsum('...gt,...t->...g', self.A, tf_x) / tf_x.size()[-1]  + self.b
        alpha = self.softplus(act) 
        # gamma = gamma / gamma.mean()
        diff_x = alpha - self.gamma * x_hat
        return diff_x

    def forward(self, batch):
        x, u, snorm_mat, unorm_mat, *others = batch
        z, qz = self.encode_z((x / snorm_mat, u, snorm_mat, unorm_mat, *others))
        # decode z
        px_z_ld = self.decode_z(z, batch)
        d, qd = self.encode_d(z, batch)
        v_from_d = self.calculate_diff_x_grad(z, qd.loc, batch)
        l, ql = self.encode_l(batch)
        l = ql.rsample()
        # v_from_model = self.calculate_diff_x_model(px_z_ld)
        v_from_model = self.calculate_diff_x_model(x / (snorm_mat))
        return(z, d, qz, qd, px_z_ld, v_from_d, v_from_model, l, ql)

    def calc_reconst_loss(self, px_z_ld, l, batch):
        x, u, snorm_mat, unorm_mat, *others = batch
        if self.loss_mode == 'nb':
            loss_func = lambda ld, norm_mat, obs: calc_nb_loss(ld, norm_mat, self.softplus(self.logstheta), obs).sum(dim=-1)
        else:
            loss_func = lambda ld, norm_mat, obs: calc_poisson_loss(ld, norm_mat, obs).sum(dim=-1)
        # reconst loss of unspliced x
        lx = loss_func(px_z_ld, snorm_mat, x)
        return lx

    def calc_vloss(self, v_from_d, v_from_model):
       if not self.stable:
        vloss = - dist.Normal(v_from_d, self.v_scale).log_prob(v_from_model).sum(axis=1)
        # vloss = self.vloss_coeff * ((v_from_d - v_from_model)**2).sum(axis=1)
        # vloss = - self.vloss_coeff * self.cos(v_from_d, v_from_model)
       else:
        vloss = self.vloss_coeff * ((v_from_model)**2).sum(axis=1)
       return vloss
    
    def elbo_loss(self, batch):
        z, d, qz, qd, px_z_ld, v_from_d, v_from_model, l, ql = self(batch)
        kld, l_kld = self.calc_kld(qz, qd, ql, ql, batch)
        lx = self.calc_reconst_loss(px_z_ld, l, batch)
        elbo_loss = torch.mean(kld + lx + l_kld)
        loss_dict = {
            "elbo_loss": elbo_loss
        }
        if not self.no_d:
            lv = torch.mean(self.calc_vloss(v_from_d, v_from_model))
            a_sp = self.asp_coeff * self.l1_loss(self.A, torch.zeros_like(self.A))
            pgamma = - dist.LogNormal(0, 1).log_prob(self.gamma).sum()
            loss_dict['elbo_loss'] = 0
            loss_dict['vloss'] = lv
            loss_dict['a_sp'] = a_sp
            loss_dict['pgamma'] = pgamma
            self.log('min_gamma', self.gamma.min())
            # a_sp = self.asp_coeff * (self.A**2).mean()
        return(loss_dict)
    
    def stable_mode(self):
        self.no_d_kld = True
        self.no_d = False
        self.stable = True
        for param in self.enc_d.parameters():
            param.requires_grad = False
        for param in self.dec_z.parameters():
            param.requires_grad = False
        for param in self.enc_z.parameters():
            param.requires_grad = False

    # def configure_optimizers(self, lr=3e-4):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
    #     return(optimizer)

class SoftSoftMax(nn.Module):
    def __init__(self, dim=-1):
        super(SoftSoftMax, self).__init__()
        self.dim = dim
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.softplus(x)
        x = x / torch.linalg.norm(x, dim=self.dim, keepdim=True)
        return x
        
class LigandDiff(pl.LightningModule):
    def __init__(self, lt_mat, h_dim=50, z_dim=10, ld=1.0):
        super(LigandDiff, self).__init__()
        self.ligand_num = lt_mat.shape[1]
        self.target_num = lt_mat.shape[0]
        self.register_buffer('lt_mat', lt_mat) 
        self.l_bias = Parameter(torch.Tensor(self.ligand_num))
        self.l_a = Parameter(torch.Tensor(self.ligand_num))
        self.l_c = Parameter(torch.Tensor(self.ligand_num))
        self.t_a = Parameter(torch.Tensor(self.target_num, 1))
        self.total_scale = Parameter(torch.Tensor(1))
        self.f_act = nn.Sequential(
            LinearReLU(z_dim, h_dim),
            SeqNN(2, h_dim),
            nn.Linear(h_dim, self.ligand_num)
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.softmax = SoftSoftMax(dim=-1)
        self.row_softmax = SoftSoftMax(dim=-2)
        self.mse = nn.MSELoss()
        self.l1_reg = nn.L1Loss()
        self.ld = ld
        self.act_norm = nn.BatchNorm1d(self.ligand_num)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.l_bias)
        init.normal_(self.l_a)
        init.normal_(self.l_c)
        init.normal_(self.t_a)
        init.normal_(self.total_scale)
        
    def forward(self, z):
        act = self.softmax(self.l_c) * self.act_norm(self.f_act(z)).unsqueeze(1)
        t_a = self.row_softmax(self.t_a)
        w = t_a * self.sigmoid(0.1 * self.lt_mat + self.l_bias)
        est_cdiff = self.total_scale * (w * act).sum(dim=2)
        return est_cdiff, act, w

    def loss(self, z, cdiff):
        est_cdiff, act, w = self(z)
        error = self.mse(est_cdiff, cdiff)
        t_a = self.row_softmax(self.t_a)
        reg = self.l1_reg(self.softmax(self.l_c), torch.zeros_like(self.l_c))
        return error, reg 

    def training_step(self, batch, batch_idx):
        z, cdiff = batch
        error, reg = self.loss(z, cdiff)
        total_loss = error + self.ld * reg
        self.log('train_loss', total_loss)
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        z, cdiff = batch
        error, reg = self.loss(z, cdiff)
        total_loss = error + self.ld * reg
        self.log('error', error)
        self.log('reg', reg)
        self.log('val_loss', total_loss)
        return total_loss
        
    def configure_optimizers(self, lr=3e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return(optimizer)


class CdiffTFActivity(pl.LightningModule):
    def __init__(self, gene_num, tf_num, known_edges, h_dim=50, z_dim=10, ld=1.0):
        super(CdiffTFActivity, self).__init__()
        self.act_f = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            SeqNN(2, h_dim),
            nn.Linear(h_dim, tf_num))
        mask = 1 - known_edges
        self.register_buffer('mask', mask)
        self.w = Parameter(torch.Tensor(tf_num, gene_num))
        self.b = Parameter(torch.Tensor(gene_num))
        self.ld = ld
        self.l1_reg = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.reset_parameters()
    
    def reset_parameters(self):
        init.normal_(self.w)
        init.normal_(self.b)
        
    def forward(self, z):
        act = self.act_f(z)
        est_cdiff = torch.matmul(act, self.w) + self.b
        return est_cdiff

    def loss(self, z, cdiff):
        est_cdiff = self(z)
        error = self.mse(est_cdiff, cdiff).mean()
        reg = self.l1_reg(self.w * self.mask, torch.zeros_like(self.w)).mean()
        loss = error + self.ld * reg
        return loss, error, reg
    
    def training_step(self, batch, batch_idx):
        z, cdiff = batch
        loss, error, reg = self.loss(z, cdiff)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        z, cdiff = batch
        loss, error, reg = self.loss(z, cdiff)
        self.log('val_loss', loss)
        self.log('error', error)
        self.log('zero_rate', (self.w.abs() < 1e-3).float().mean())
        self.log('reg', reg)
        return loss

    def configure_optimizers(self, lr=0.01):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return(optimizer)

class OptNormal(nn.Module):
    def __init__(self, num, dim):
        super().__init__()
        self.loc = nn.Parameter(torch.Tensor(num, dim))
        self.raw_scale = nn.Parameter(torch.Tensor(num, dim))
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.loc)
        init.normal_(self.raw_scale)

    def forward(self):
        return dist.Normal(self.loc, self.softplus(self.raw_scale))

class Dvicdyf(EnvDyn):
    def __init__(self, num_enc_z_layers=2, x_dim=100, enc_z_h_dim=50, z_dim=10, num_enc_d_layers=2, enc_d_h_dim=50, norm_input=True, vamp_comps=1024, total_size=100, ref_z_var=True, **kwargs):
        super(Dvicdyf, self).__init__(num_enc_z_layers=num_enc_z_layers, x_dim=x_dim, enc_z_h_dim=enc_z_h_dim, z_dim=z_dim, num_enc_d_layers=num_enc_d_layers, enc_d_h_dim=enc_d_h_dim, norm_input=norm_input, vamp_comps=vamp_comps, **kwargs)
        self.enc_z = DiscreteEncoder(num_enc_z_layers, x_dim, enc_z_h_dim, z_dim, norm_input=norm_input)
        self.enc_d = DiscreteEncoder(num_enc_d_layers, z_dim, enc_d_h_dim, z_dim)
        self.total_size = total_size
        self.ref_z_var = ref_z_var
        if self.ref_z_var:
            self.ref_z_dist = OptNormal(vamp_comps, z_dim)
            self.comp_p = nn.Parameter(torch.zeros(vamp_comps))
        else:
            self.ref_z = nn.Parameter(torch.Tensor(vamp_comps, z_dim))
            init.normal_(self.ref_z)
            self.register_buffer('comp_p', (torch.zeros(vamp_comps)))

    def encode_d(self, z, batch):
        d, qd = self.enc_d(z, self.ref_z.unsqueeze(-2) - z.unsqueeze(-3))
        self.log('total_d', d.norm(dim=-1).mean())
        return d, qd

    def encode_z_spec(self, sx, batch):
        z, qz = self.enc_z(sx, self.ref_z.unsqueeze(-2))
        self.log('max_assing', qz.rho.max(dim=1)[0].mean())
        self.log('assined nodes', (qz.rho.sum(dim=0) > 0.1 * qz.rho.sum(dim=0).mean()).float().mean())
        return z, qz

    def calc_lp_kkp(self, batch):
        ref_z_diff = self.ref_z.unsqueeze(-2) - self.ref_z.unsqueeze(-3)
        orig_dist = dist.Normal(torch.zeros_like(self.ref_z), (0.01 ** (1/self.ref_z.size()[1])) * torch.ones_like(self.ref_z))
        lprobs = normalize_lp(orig_dist.log_prob(ref_z_diff).sum(dim=-1).permute(1, 0))
        return lprobs

    def calc_lq_kkp(self, batch):
        d, qd = self.encode_d(self.ref_z, batch)
        return qd.lrho

    def calc_ref_z_kl(self, batch):
        kld = dist.kl.kl_divergence(self.ref_z_dist(), dist.Normal(0, 1)).sum()
        return kld
        
    def calc_ref_z_kl_map(self, batch):
        lp = - (dist.Normal(0, 1).log_prob(self.ref_z)).sum()
        return lp

    def calc_zd_kld(self, qz, qd, batch):
        lp = normalize_lp(self.comp_p) 
        l_qp = qz.lrho - lp
        lp_kkp = self.calc_lp_kkp(batch)
        lq_kkp = self.calc_lq_kkp(batch)
        l_qp_kkp = (self.softmax(lq_kkp) * (lq_kkp - lp_kkp)).sum(dim=-1)
        kld = (qz.rho * (l_qp + l_qp_kkp)).sum(dim=-1) 
        return kld

    def calc_kld(self, qz, qd, ql, qul, batch):
        kld = self.calc_zd_kld(qz, qd, batch)
        # kld for size factor
        l_kld = self.calc_l_kld(ql, batch)
        return kld, l_kld 

    def validation_step(self, batch, batch_idx):
        total_loss = super().validation_step(batch, batch_idx)
        self.enc_z.decrease_temp()
        self.enc_d.decrease_temp()
        self.log('d_temp', self.enc_d.temp)
        return total_loss
        
    def elbo_loss(self, batch):
        if self.ref_z_var:
            self.ref_z = self.ref_z_dist().rsample()
        loss_dict = super().elbo_loss(batch)
        if self.ref_z_var:
            ref_z_kld = self.calc_ref_z_kl(batch) / self.total_size
            loss_dict['ref_z_kld'] = ref_z_kld
        else:
            ref_z_kld = self.calc_ref_z_kl_map(batch) / self.total_size
            loss_dict['ref_z_kld'] = ref_z_kld
        return loss_dict


class VLossMixin:
    def elbo_loss(self, batch):
       loss_dict = super().elbo_loss(batch)
       loss_dict

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(dim))

    def forward(self, t):
        angles = t * self.weights
        embed = torch.cat((t, angles.sin(), angles.cos()), dim=-1)
        return embed

class NoisePredictor(nn.Module):
    def __init__(self, t_max, z_dim, h_dim):
        super().__init__()
        half_h_dim = int(h_dim / 2)
        self.embedding = SinusoidalEmbedding(half_h_dim)
        self.z2h = LinearReLU(2 * z_dim, h_dim)
        self.ff_layers = nn.Sequential(
            LinearReLU(h_dim * 2 + 1, h_dim),
            SeqNN(2, h_dim),
            nn.Linear(h_dim, 2 * z_dim)
        )
    
    def forward(self, zt, t):
        t_h = self.embedding(t)
        x_h = self.z2h(zt)
        in_zt = torch.cat(
            [x_h, t_h], dim=-1
        )
        pred_eps = self.ff_layers(in_zt)
        return pred_eps



class DiffusionPrior(nn.Module):
    def __init__(self, t_max, z_dim, h_dim):
        super(DiffusionPrior, self).__init__()
        self.t_max = t_max
        cum_alphas = 1 - torch.cumsum(torch.nn.functional.softmax(torch.ones(t_max)), 0)
        alphas = torch.cat([torch.tensor([cum_alphas[0]]), cum_alphas[1:] / cum_alphas[:-1]])
        self.register_buffer('cum_alphas', cum_alphas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('t_vec', torch.arange(t_max))
        self.noise_predictor = NoisePredictor(t_max, z_dim, h_dim)
        self.dec_temp = DecreasingTemp(dec_temp_steps=1000)

    def forward(self, qz, qd):
        mu_z0 = torch.cat([qz.loc, qd.loc], dim=-1)
        scale_z0 = torch.cat([qz.scale, qd.scale], dim=-1)
        sub_t = torch.randint_like(mu_z0[:, 0], self.t_max - 1).unsqueeze(-1).long()
        sub_cum_alphas = self.cum_alphas[sub_t]
        sub_alphas = self.alphas[sub_t]
        bp = 1 + sub_cum_alphas * (scale_z0 ** 2 - 1)
        coeff = self.t_max * 0.5 * (1 - sub_alphas) * bp / ((1 - sub_cum_alphas)**2 * sub_alphas)
        orig_eps = torch.randn_like(mu_z0)
        z_t = sub_cum_alphas * mu_z0 + torch.sqrt((1 - sub_cum_alphas + sub_cum_alphas * (scale_z0 ** 2))) * orig_eps
        pred_eps = self.noise_predictor(z_t, sub_t.float() / self.t_max)
        kld = torch.sum(coeff * (orig_eps - pred_eps) ** 2, axis=-1)
        return kld


    
class VicDyfDP(EnvDyn):
    def __init__(self, **kwargs):
        super(VicDyfDP, self).__init__(**kwargs)
        self.diff_prior = DiffusionPrior(1000, kwargs['z_dim'], kwargs['enc_z_h_dim'])

    def calc_kld(self, qz, qd, ql, qul, batch):
        # kld of pz and qz
        kld = self.diff_prior(qz, qd)
        # kld for size factor
        l_kld = self.calc_l_kld(ql, batch)
        return kld, l_kld 
