import torch
from torch import Tensor
from torch.nn.functional import pad, mse_loss
import random
from typing import Literal, Tuple, Optional, List



class MSE_w_padding(torch.nn.Module):

    def __init__(self, dim_matching='zero_pad', reduction="mean"):
        super(MSE_w_padding, self).__init__()
        self.dim_matching = dim_matching
        self.reduction = reduction
        
    def forward(self, X, Y):
        if X.shape[:-1] != Y.shape[:-1] or X.ndim != 3 or Y.ndim != 3:
            raise ValueError('Expected 3D input matrices to much in all dimensions but last.'
                             f'But got {X.shape} and {Y.shape} instead.')

        if X.shape[-1] != Y.shape[-1]:
            if self.dim_matching is None or self.dim_matching == 'none':
                raise ValueError(f'Expected same dimension matrices got instead {X.shape} and {Y.shape}. '
                                 f'Set dim_matching or change matrix dimensions.')
            elif self.dim_matching == 'zero_pad':
                size_diff = Y.shape[-1] - X.shape[-1]
                if size_diff < 0:
                    raise ValueError(f'With `zero_pad` dimension matching expected X dimension to be smaller then Y. '
                                     f'But got {X.shape} and {Y.shape} instead.')
                X = pad(X, (0, size_diff))
            elif self.dim_matching == 'pca':
                raise NotImplementedError
            else:
                raise ValueError(f'Unrecognized dimension matching {self.dim_matching}')

        return mse_loss(X,Y, reduction = self.reduction)
        
class LinearMeasure(torch.nn.Module):
    def __init__(self,
                 alpha=1, center_columns=True, dim_matching='zero_pad', svd_grad=False, reduction='mean', no_svd=True):
        super(LinearMeasure, self).__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        assert dim_matching in [None, 'none', 'zero_pad', 'pca']
        self.dim_matching = dim_matching
        self.center_columns = center_columns
        self.svd_grad = svd_grad
        self.reduction = reduction
        self.no_svd = no_svd

    def partial_fit(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the mean centered columns. Can be replaced later by whitening transform for linear invarariances."""
        if self.center_columns:
            mx = torch.mean(X, dim=1, keepdim=True)
        else:
            mx = torch.zeros(X.shape[2], dtype=X.dtype, device=X.device)
        wx = X - mx

        return mx, wx

    def fit(self, X: Tensor, Y: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        mx, wx = self.partial_fit(X)
        my, wy = self.partial_fit(Y)
        if self.svd_grad:
            wxy = torch.bmm(wx.transpose(1, 2), wy)
            U, _, Vt = torch.linalg.svd(wxy,driver="gesvd", full_matrices=False)
        else:
            with torch.no_grad():
                wxy = torch.bmm(wx.transpose(1, 2), wy)
                U, _, Vt = torch.linalg.svd(wxy,driver="gesvd", full_matrices=False)
        wx = U
        wy = Vt.transpose(1, 2)
        return (mx, wx), (my, wy)

    def project(self, X: Tensor, m: Tensor, w: Tensor):
        if self.center_columns:
            return torch.bmm((X - m), w)
        else:
            return torch.bmm(X, w)

    def forward(self, X: Tensor, Y: Tensor):
        if X.shape[:-1] != Y.shape[:-1] or X.ndim != 3 or Y.ndim != 3:
            raise ValueError('Expected 3D input matrices to much in all dimensions but last.'
                             f'But got {X.shape} and {Y.shape} instead.')

        if X.shape[-1] != Y.shape[-1]:
            if self.dim_matching is None or self.dim_matching == 'none':
                raise ValueError(f'Expected same dimension matrices got instead {X.shape} and {Y.shape}. '
                                 f'Set dim_matching or change matrix dimensions.')
            elif self.dim_matching == 'zero_pad':
                size_diff = Y.shape[-1] - X.shape[-1]
                if size_diff < 0:
                    raise ValueError(f'With `zero_pad` dimension matching expected X dimension to be smaller then Y. '
                                     f'But got {X.shape} and {Y.shape} instead.')
                X = pad(X, (0, size_diff))
            elif self.dim_matching == 'pca':
                raise NotImplementedError
            else:
                raise ValueError(f'Unrecognized dimension matching {self.dim_matching}')

        if self.no_svd and X.shape[1] == 1:
            mx, wx = self.partial_fit(X)
            my, wy = self.partial_fit(Y)

            x_norm = torch.linalg.norm(wx, dim=(1, 2))
            y_norm = torch.linalg.norm(wy, dim=(1, 2))

            norms = (x_norm - y_norm) ** 2

        else:
            X_params, Y_params = self.fit(X, Y)
            norms = torch.linalg.norm(self.project(X, *X_params) - self.project(Y, *Y_params), ord="fro", dim=(1, 2))

        if self.reduction == 'mean':
            return norms.mean()
        elif self.reduction == 'sum':
            return norms.sum()
        elif self.reduction == 'none' or self.reduction is None:
            return norms
        else:
            raise ValueError(f'Unrecognized reduction {self.reduction}')


class CKA(torch.nn.Module):
    def __init__(self,dim_matching='zero_pad', reduction='mean', kernel="linear", similarity_token_strategy="flatten"):
        super(CKA, self).__init__()
        assert dim_matching in [None, 'none', 'zero_pad', 'pca']
        self.dim_matching = dim_matching
        self.reduction = reduction
        self.kernel=kernel
        self.similarity_token_strategy=similarity_token_strategy
        self.random_tokens=None


    def generate_random_token_index(self, token_size, selected_size=10):
        self.random_tokens = random.sample(range(token_size),selected_size) 
    
    def create_sim_matrix(self, X:Tensor, diag_zero=True):
        if self.similarity_token_strategy =="flatten":
            X=torch.flatten(X, end_dim=-2)
        elif self.similarity_token_strategy=="random":
            if not self.random_tokens:
                self.generate_random_token_index(X.shape[1])
            X=torch.fatten(X[:, self.random_tokens, :], end_dim=-2)
            
        """Similarity matrix"""
        if self.kernel == "linear":
            sim_m = X@ X.T
        else:
            raise NotImplementedError

        if diag_zero:
            diagonal_mask = torch.eye(sim_m.shape[-1], dtype=torch.bool).unsqueeze(0).to(X.device)

            #has a batch dimension
            if len(sim_m.shape)>2:
                sim_m.masked_fill_(diagonal_mask, 0)
            else:
                sim_m.masked_fill_(diagonal_mask[0], 0)
        
        return sim_m


    def HSIC(self, K,L):
        """ K, L are similarity matrices of form (B,N,N) where B is batch or (N,N)
        """
        n=K.shape[-1]
        if len(K.shape)==3:
            pass
            #TODO: IMPLEMENT BATCHED CKA
            """
            kl=torch.bmm(K,L)"""
            
        else:
            kl=K@L
            trace=kl.diagonal().sum()
            middle=(K.sum()*L.sum())/((n-1)*(n-2))
            last= -2*kl.sum()/(n-2)
            return (trace+middle+last)/(n*(n-3))


    def forward(self, X: Tensor, Y: Tensor):
        if X.shape[:-1] != Y.shape[:-1] or X.ndim != 3 or Y.ndim != 3:
            raise ValueError('Expected 3D input matrices to match in all dimensions but last.'
                             f'But got {X.shape} and {Y.shape} instead.')

        if X.shape[-1] != Y.shape[-1]:
            if self.dim_matching is None or self.dim_matching == 'none':
                raise ValueError(f'Expected same dimension matrices got instead {X.shape} and {Y.shape}. '
                                 f'Set dim_matching or change matrix dimensions.')
            elif self.dim_matching == 'zero_pad':
                size_diff = Y.shape[-1] - X.shape[-1]
                if size_diff < 0:
                    raise ValueError(f'With `zero_pad` dimension matching expected X dimension to be smaller then Y. '
                                     f'But got {X.shape} and {Y.shape} instead.')
                X = pad(X, (0, size_diff))
            elif self.dim_matching == 'pca':
                raise NotImplementedError
            else:
                raise ValueError(f'Unrecognized dimension matching {self.dim_matching}')

        X_sim_matrix = self.create_sim_matrix(X)
        Y_sim_matrix = self.create_sim_matrix(Y)

        self_hsic_x=self.HSIC(X_sim_matrix, X_sim_matrix)
        self_hsic_y= self.HSIC(Y_sim_matrix, Y_sim_matrix)
        cross_hsic=self.HSIC(X_sim_matrix, Y_sim_matrix)
        
        batched_cka = cross_hsic/ (torch.sqrt(self_hsic_x) * torch.sqrt(self_hsic_y))
            
        if self.reduction == 'mean':
            return batched_cka.mean()
        elif self.reduction == 'sum':
            return batched_cka.sum()
        elif self.reduction == 'none' or self.reduction is None:
            return batched_cka
        else:
            raise ValueError(f'Unrecognized reduction {self.reduction}')

class EnergyMetric(torch.nn.Module):

    def __init__(self, n_iter=100, tol=1e-6, dim_matching='zero_pad', reduction='mean'):
        super(EnergyMetric, self).__init__()
        self.n_iter = n_iter
        self.tol = torch.tensor(tol)
        assert dim_matching in [None, 'none', 'zero_pad', 'pca']
        self.dim_matching = dim_matching
        self.reduction = reduction

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor):

        n_x = X.shape[2]
        n_y = Y.shape[2]

        X = X.repeat_interleave(n_y, dim=2).flatten(start_dim=1, end_dim=2)
        Y = Y.tile(dims=(1, 1, n_x, 1)).flatten(start_dim=1, end_dim=2)

        if X.shape[1] != Y.shape[1]:
            raise ValueError(f"After permutation got {X.shape} and {Y.shape}")

        w = torch.ones(X.shape[0], X.shape[1])

        batch_loss = [torch.mean(torch.linalg.norm(X - Y, dim=-1), dim=-1)]
        for i in range(self.n_iter):
            T = self.get_orth_matrix(w[:, :, None] * X, w[:, :, None] * Y)
            iter_result = torch.linalg.norm(X - torch.bmm(Y, T), dim=-1)
            batch_loss.append(torch.mean(iter_result, dim=-1))
            w = 1 / torch.maximum(torch.sqrt(iter_result), self.tol)

        return w, T, batch_loss, X, Y

    def get_orth_matrix(self, X: torch.Tensor, Y: torch.Tensor):
        U, _, Vt = torch.linalg.svd(torch.bmm(X.transpose(1, 2), Y))
        return torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    def get_dist_energy(self, X: torch.Tensor):
        n = X.shape[2]
        combs = torch.combinations(torch.arange(n))
        X1 = torch.flatten(X[:, :, combs[:, 0], :], start_dim=1, end_dim=2)
        X2 = torch.flatten(X[:, :, combs[:, 1], :], start_dim=1, end_dim=2)

        return torch.mean(torch.linalg.norm(X2 - X1, dim=-1), dim=-1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):

        """Expected tensors to be of the form batch x class x repeats x activations"""

        if X.shape[:-2] != Y.shape[:-2] or X.ndim != 4 or Y.ndim != 4:
            raise ValueError('Expected 4D input matrices to much in all dimensions but last two.'
                             f'But got {X.shape} and {Y.shape} instead.')

        if X.shape[-1] != Y.shape[-1]:
            if self.dim_matching is None or self.dim_matching == 'none':
                raise ValueError(f'Expected same dimension matrices got instead {X.shape} and {Y.shape}. '
                                 f'Set dim_matching or change matrix dimensions.')
            elif self.dim_matching == 'zero_pad':
                size_diff = Y.shape[-1] - X.shape[-1]
                if size_diff < 0:
                    raise ValueError(f'With `zero_pad` dimension matching expected X dimension to be smaller then Y. '
                                     f'But got {X.shape} and {Y.shape} instead.')
                X = pad(X, (0, size_diff))
            elif self.dim_matching == 'pca':
                raise NotImplementedError
            else:
                raise ValueError(f'Unrecognized dimension matching {self.dim_matching}')

        # return self.fit(X,Y)
        w, T, fit_loss, X_prod, Y_prod = self.fit(X, Y)

        e_xx = self.get_dist_energy(X)
        e_yy = self.get_dist_energy(Y)
        Y_proj = torch.bmm(Y_prod, T)
        e_xy = torch.mean(torch.linalg.norm(X_prod - Y_proj, dim=-1), dim=-1)

        norms = torch.sqrt(torch.nn.functional.relu(e_xy - 0.5 * (e_xx + e_yy)))

        if self.reduction == 'mean':
            return norms.mean()
        elif self.reduction == 'sum':
            return norms.sum()
        elif self.reduction == 'none' or self.reduction is None:
            return norms
        else:
            raise ValueError(f'Unrecognized reduction {self.reduction}')
