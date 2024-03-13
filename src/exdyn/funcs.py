import torch.distributions as dist


def calc_kld(qz):
    kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
    return(kld)


def calc_poisson_loss(ld, norm_mat, obs):
    p_z = dist.Poisson(ld * norm_mat + 1.0e-16)
    l = - p_z.log_prob(obs)
    return(l)
        
    
def calc_nb_loss(ld, norm_mat, theta, obs):
    ld = norm_mat * ld
    ld = ld + 1.0e-16
    theta = theta + 1.0e-16
    lp =  ld.log() - (theta).log()
    p_z = dist.NegativeBinomial(theta, logits=lp)
    l = - p_z.log_prob(obs)
    return(l)


def normalize_lp(p):
    return p - p.logsumexp(dim=-1, keepdim=True)