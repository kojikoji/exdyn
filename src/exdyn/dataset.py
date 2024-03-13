# evndyn
import torch
import numpy as np

class EnvDynDataSet(torch.utils.data.Dataset):
    def __init__(self, s, u, snorm_mat, unorm_mat, transform=None, pre_transform=None):
        self.s = s
        self.u = u
        self.snorm_mat = snorm_mat
        self.unorm_mat = unorm_mat

    def __len__(self):
        return(self.s.shape[0])

    def __getitem__(self, idx):
        idx_s = self.s[idx]
        idx_u = self.u[idx]
        idx_snorm_mat = self.snorm_mat[idx]
        idx_unorm_mat = self.unorm_mat[idx]
        return(idx_s, idx_u, idx_snorm_mat, idx_unorm_mat)


class VicDyfMbDataSet(torch.utils.data.Dataset):
    def __init__(self, s, u, snorm_mat, unorm_mat, b, transform=None, pre_transform=None):
        self.s = s
        self.u = u
        self.snorm_mat = snorm_mat
        self.unorm_mat = unorm_mat
        self.b = b

    def __len__(self):
        return(self.s.shape[0])

    def __getitem__(self, idx):
        idx_s = self.s[idx]
        idx_u = self.u[idx]
        idx_snorm_mat = self.snorm_mat[idx]
        idx_unorm_mat = self.unorm_mat[idx]
        idx_b = self.b[idx]
        return(idx_s, idx_u, idx_snorm_mat, idx_unorm_mat, idx_b)

class CvicDyfDataSet(torch.utils.data.Dataset):
    def __init__(self, s, u, snorm_mat, unorm_mat, b, t,  transform=None, pre_transform=None):
        self.s = s
        self.u = u
        self.snorm_mat = snorm_mat
        self.unorm_mat = unorm_mat
        self.b = b
        self.t = t

    def __len__(self):
        return(self.s.shape[0])

    def __getitem__(self, idx):
        idx_s = self.s[idx]
        idx_u = self.u[idx]
        idx_snorm_mat = self.snorm_mat[idx]
        idx_unorm_mat = self.unorm_mat[idx]
        idx_b = self.b[idx]
        idx_t = self.t[idx]
        return(idx_s, idx_u, idx_snorm_mat, idx_unorm_mat, idx_b, idx_t)



class EnvDynDataManager():
    def __init__(self, s, u, test_ratio, batch_size, num_workers, validation_ratio=0.1):
        s = s.float()
        u = u.float()
        norm_mat = torch.sum(s, dim=1).view(-1, 1) * torch.sum(s, dim=0).view(1, -1)
        norm_mat = torch.mean(s) * norm_mat / torch.mean(norm_mat)
        self.s = s
        self.u = u
        self.norm_mat = norm_mat
        total_num = s.shape[0]
        validation_num = int(total_num * validation_ratio)
        test_num = int(total_num * test_ratio)
        np.random.seed(42)
        idx = np.random.permutation(np.arange(total_num))
        validation_idx, test_idx, train_idx = idx[:validation_num], idx[validation_num:(validation_num +  test_num)], idx[(validation_num +  test_num):]
        self.validation_idx, self.test_idx, self.train_idx = validation_idx, test_idx, train_idx
        self.validation_s = s[validation_idx]
        self.validation_u = u[validation_idx]
        self.validation_norm_mat = norm_mat[validation_idx]
        self.test_s = s[test_idx]
        self.test_u = u[test_idx]
        self.test_norm_mat = norm_mat[test_idx]
        self.train_eds = EnvDynDataSet(s[train_idx], u[train_idx], norm_mat[train_idx])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

class LigandDiffDataset(torch.utils.data.Dataset):
    def __init__(self, z, cdiff, transform=None, pre_transform=None):
        self.z = z
        self.cdiff = cdiff

    def __len__(self):
        return(self.z.shape[0])

    def __getitem__(self, idx):
        idx_z = self.z[idx]
        idx_cdiff = self.cdiff[idx]
        return(idx_z, idx_cdiff)

