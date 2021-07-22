"""
This file contains PyTorch implementation of methods from A Dirichlet Mixture Model 
of Hawkes Processes for Event Sequence Clustering
https://arxiv.org/pdf/1701.09177.pdf
"""

import torch
import numpy as np
from tqdm import tqdm, trange
from typing import List
from torch.nn import functional as F
import random
import scipy.integrate as integrate


def tune_basis_fn(ss, eps=1e5, not_tune=False):
    if not not_tune:
        w0_arr = np.linspace(0, 15, 1000) # 15

        all_seq = torch.cat(tuple(ss))
        T = torch.max(all_seq[:, 0]).item()
        N_events = np.sum([len(seq) for seq in ss])

        h = ((4*torch.std(all_seq)**5) / (3*N_events))**0.2
        const = N_events*np.sqrt(2*np.pi*h**2)
        upper_bound = lambda w: const * np.exp(-(w**2*h**2)/2)
        result = lambda w0: integrate.quad(upper_bound, w0, np.inf)

        basis_fs = []
        for w0 in w0_arr:
            if result(w0)[0] <= eps:
                M = int(np.ceil(T*w0/np.pi))
                sigma = 1 / w0
                basis_fs = [lambda t, m_=m: np.exp(-(t - (m_-1)*T/M)**2 / (2*sigma**2)) for m in range(1, M+1)]
                break

        print ('eps =', eps, 'w0 =', w0)
        print ('D =', len(basis_fs))
    
    else:
        D = 6
        basis_fs = [lambda t: torch.exp(-t**2 / (2*k**2)) for k in range(D)]
    
    return basis_fs


def integral(f, left, right, npts=10000):
    grid = torch.FloatTensor(np.linspace(left.tolist(), right.tolist(), npts))
    out = f(grid)
    int_f = (out[0, :] / 2. + out[-1, :] / 2. + out[1:-1, :]).sum(0) * (grid[1, :] - grid[0, :])
    return int_f    


class PointProcessStorage:
    """
    Class for storing point process' realizations

    Args:
    - seqs - list of realizations, where each item is a tensor of size (L, 2)
    - Tns - tensor of Tn for each realization (each event's timestamp lies in (0, Tn))

    """
    def __init__(self, seqs : List, Tns : torch.Tensor, basis_fs : List):
        self.N = len(seqs)
        self.seqs = seqs
        self.Tns = Tns
        self.basis_fs = basis_fs

    def __iter__(self):
        for n in range(len(self.seqs)):
            cs, ts = self.seqs[n][:, 1], self.seqs[n][:, 0]
            Tn = self.Tns[n]
            assert Tn >= ts[-1]
            yield cs.long(), ts.float(), Tn


class DirichletMixtureModel:
    """
    Dirichlet Mixture Model for Hawkes Processes

    Args:
    - K - number of clusters
    - C - number of event classes
    - D - number of basis functions
    - alpha - parameter of Dirichlet distribution
    - B - parameter of Reyleight distribution - prior of \mu
    - Sigma - parameter of Exp distribution - prior of A

    """
    def __init__(self, K, C, D, alpha, B=None, Sigma=None):
        self.alpha = alpha
        self.K = K
        concentration = torch.FloatTensor([alpha /  float(K) for _ in range(K)])
        self.dirich = torch.distributions.dirichlet.Dirichlet(concentration)
        self.pi = self.dirich.sample()
        self.cat = torch.distributions.categorical.Categorical(self.pi)
        self.k_pi = self.cat.sample()
        self.B = B  # C x K
        self.rayleigh = torch.distributions.weibull.Weibull(2**0.5 * B, 2 * torch.ones_like(B))
        self.mu = self.rayleigh.sample()
        self.Sigma = Sigma
        self.exp = torch.distributions.exponential.Exponential((Sigma)**(-1))
        self.A = self.exp.sample() # C x C x D x K

    def p_mu(self, mu=None, B=None):
        mu = mu if mu is not None else self.mu
        if B is not None:
            return torch.prod(torch.exp(torch.distributions.Weibull(2**0.5 * B, 2 * torch.ones_like(B)).log_prob(mu)))
        else:
            return torch.prod(torch.exp(self.rayleigh.log_prob(mu)))

    def p_A(self, A=None, Sigma=None):
        A = A if A is not None else self.A
        if Sigma is not None:
            return torch.prod(torch.exp(torch.distributions.Exponential((Sigma)**(-1)).log_prob(A)))
        else:
            return torch.prod(torch.exp(self.exp.log_prob(A)))

    def p_pi(self, pi=None, K=None):
        pi = pi if pi is not None else self.pi
        if K is not None:
            concentration = torch.FloatTensor([self.alpha /  float(K) for _ in range(K)])
            return torch.prod(torch.exp(torch.distributions.Dirichlet(concentration).log_prob(pi)))
        return torch.prod(torch.exp(self.dirich.log_prob(pi)))

    def e_logpi(self):
        """
        Returns expectation of logarithm of \pi
        """
        return (torch.digamma(torch.FloatTensor([self.alpha / self.K])) - torch.digamma(torch.FloatTensor([self.alpha]))).repeat(self.K)

    def e_mu(self):
        """
        Returns expectation of \mu
        """
        return (np.pi / 2)**.5 * self.B

    def e_A(self):
        """
        Returns expectation of \A
        """
        return self.Sigma

    def var_mu(self):
        """
        Returns variance of \mu
        """
        return (4 - np.pi) / 2 * self.B**2

    def var_A(self):
        """
        Returns variance of \A
        """
        return self.Sigma**2

    def update_A(self, A, Sigma):
        self.A = A
        self.Sigma = Sigma
        self.exp = torch.distributions.exponential.Exponential((Sigma)**(-1))

    def update_mu(self, mu, B):
        self.mu = mu
        self.B = B
        self.rayleigh = torch.distributions.weibull.Weibull(2**0.5 * B, 2 * torch.ones_like(B))

    def update_pi(self, pi):
        self.pi = pi
        self.K = pi.shape[0]
        concentration = torch.FloatTensor([self.alpha /  float(self.K) for _ in range(self.K)])
        self.dirich = torch.distributions.Dirichlet(concentration)


class EM_clustering:
    """
    Class for learning parameters of Hawkes Processes' clusters

    Args:
    - hp - PointProcessStorage object
    - model - DirichletMixtureModel object
    - n_inner - Number of inner iterations
    """
    def __init__(self, hp : PointProcessStorage, 
                    model : DirichletMixtureModel,
                    n_inner : int = 5):
        self.N = hp.N
        self.K = model.K
        self.C = model.mu.shape[0]
        self.hp = hp
        self.model = model
        self.n_inner = n_inner

        self.g = []
        self.int_g = []

        self.gg = []

    def learn_hp(self, niter=100, ninner=None):
        ninner = ninner if ninner is not None else self.n_inner
        if isinstance(ninner, int):
            ninner = [ninner]*niter
        assert len(ninner) == niter

        nll_history = []
        r_history = torch.empty(niter, self.N, self.K)
        r = torch.rand(self.N, self.K)
        r = r / r.sum(1)[:, None]
        for i, nin in tqdm(enumerate(ninner)):
            # commented so far - can't compute real nll
            # mu, A = self.m_step(r, nin)
            # nll1 = self.hp_nll(r.sum(0) / self.N, mu, A)

            r2 = self.e_step()
            print ('e_step ready')
            mu2, A2 = self.m_step(r2, nin)
            nll2 = self.hp_nll(r2.sum(0) / self.N, mu2, A2)

            # commented so far - can't compute real nll
            #if nll1 < nll2:
            if False:
                pi = r.sum(0) / self.N
                self.update_model(pi, mu, A)
            else:
                r = r2
                nll1 = nll2
                pi = r.sum(0) / self.N
                self.update_model(pi, mu2, A2)
            r_history[i] = r
            nll_history.append(nll1.item())
            print(f'\nNLL / N: {(nll1.item() / self.N):.4f}')

            # commented so far - can't compute real nll
            # K = self.K
            # print('\n', self.K)
            # self.update_num_clusters(nll1)
            # print('\n', self.K)
            # if self.K != K:
            #     r = self.e_step()

        return r, nll_history, r_history

    def update_model(self, pi, mu, A):
        Sigma = A
        B = (2 / np.pi)**.5 * mu
        self.model.update_A(A, Sigma)
        self.model.update_mu(mu, B)
        self.model.update_pi(pi)
        self.model.K = pi.shape[0]

    def hp_nll(self, pi, mu, A):
        """
        Computes negative log-likelihood lower bound (via Jensen inequality) given \pi, \mu, A

        Need doublechecking - no way to compute real log-likelihood of DMM because of long sequences leading to overflow 
        """
        assert torch.isclose(pi.sum(0), torch.ones_like(pi.sum(0))), pi
        nll = 0

        for n, (c, _, Tn) in enumerate(self.hp):
            g = self._get_g(n)
            A_g = (A[c.tolist(), c.tolist(), :, :] * g[:, :, :, None]).sum(2) # L x L x K
            lamda = mu[c.tolist(), :] + A_g.sum(1)

            int_g = self._get_int_g(n)
            A_g = (A[:, c.tolist(), :, :] * int_g[None, :, :, None]).sum(2)# C x K
            int_lambda = Tn * mu + A_g.sum(1)
            #integral = self.integral_lambda(t, c, Tn, mu, A, int_g=self._get_int_g(n))

            ll_lower_bound = (pi * (torch.log(lamda).sum(0) - int_lambda.sum(0))).sum(0)
            #assert (ll_lower_bound < 0).all(), ll_lower_bound
            nll -= ll_lower_bound

        return nll

    def _get_g(self, n):
        t = self.hp.seqs[n][:, 0]
        if len(self.g) <= n:
            tau = torch.tril(t.unsqueeze(1).repeat(1, t.shape[0]) - t[None, :], diagonal=-1)
            assert (tau >= 0).all()
            g = torch.stack([f(tau) for f in self.hp.basis_fs], dim=-1)
            g = g * (tau > 0)[:, :, None]
            self.g.append(g)
        else:
            g = self.g[n]

        return g

    def _get_int_g(self, n):
        t = self.hp.seqs[n][:, 0]
        Tn = self.hp.Tns[n]
        if len(self.int_g) <= n:
            int_g = torch.stack([integral(f, torch.zeros_like(t), Tn - t) for f in self.hp.basis_fs], dim=-1)
            self.int_g.append(int_g)
        else:
            int_g = self.int_g[n]
        
        return int_g

    def e_step(self):
        log_rho = torch.zeros(self.N, self.K)
        elogpi = self.model.e_logpi()
        log_rho += elogpi[None, :]
        
        e_mu = self.model.e_mu() # C x K
        e_A = self.model.e_A() # C x C x D x K
        var_mu = self.model.var_mu()

        for n, (c, _, Tn) in enumerate(self.hp):
            print ('e_step:', n)
            g = self._get_g(n)
            e_A_g = (e_A[c.tolist(), c.tolist(), :, :] * g[:, :, :, None]).sum(2).permute(2, 0, 1) # K x L x L
            e_lambda = e_mu[c.tolist(), :].permute(1, 0) + e_A_g.sum(-1)
            var_lambda = var_mu[c.tolist(), :].permute(1, 0) + ((e_A_g)**2).sum(1)
            log_rho[n, :] += (torch.log(e_lambda) - var_lambda / (2*e_lambda**2)).sum(1)

            int_g = self._get_int_g(n)
            int_lambda = (Tn * e_mu + (e_A[:, c.tolist(), :, :] * int_g[None, :, :, None]).sum(2).sum(1)).sum(0)
            log_rho[n, :] -= int_lambda

        rho = F.softmax(log_rho, -1)
        r = rho / rho.sum(1)[:, None]
        assert torch.isclose(r.sum(1), torch.ones_like(r.sum(1))).all(), r.sum(1)

        return r

    def m_step(self, r, niter=8):
        mu = (np.pi / 2)**.5 * self.model.B
        beta = self.model.B
        A = self.model.Sigma
        Sigma = self.model.Sigma
        C = mu.shape[0]
        
        for _ in range(niter):
            b = torch.zeros(self.K)
            c = -1 * torch.ones(self.C, self.K)
            s = 0
            d = 0
            for n, (cs, _, Tn) in enumerate(self.hp):
                b += r[n] * Tn # K 

                g = self._get_g(n)
                A_g = (A[cs.tolist(), cs.tolist(), :, :] * g[:, :, :, None]) # L x L x D x K
                lamda = mu[cs.tolist(), :] + A_g.sum(2).sum(1) 
                pii = mu[cs.tolist(), :] / lamda # L x K
                
                #A_g_ = (A[cs.tolist(), cs.tolist(), :, :] * g[:, :, :, None]) # L x L x D x K
                pijd = A_g / lamda[:, None, None, :] # L x L x D x K
                assert (pijd <= 1).all(), (pijd.max(), lamda.min(), mu.min())
                
                x = cs.unsqueeze(0).repeat(C, 1)
                eq = x == torch.tensor(np.arange(C)).unsqueeze(1).repeat(1, cs.shape[0]) # C x L 
                sum_pii = torch.stack([pii[torch.BoolTensor(mask)].sum(0) for mask in eq], dim=0) #None, :, :] * eq[:, :, None]).sum(1) # C x K
                c -= r[n][None, :] * sum_pii # C x K
                
                #sum_pijd = (pijd[None, :, :, :, :] * eq[:, :, None, None, None]).sum(1) # C L D K
                #sum_pijd = (sum_pijd[:, None, :, :, :] * eq[None, :, :, None, None]).sum(2) # C C D K
                sum_pijd = torch.stack([(pijd[torch.BoolTensor(mask)]).sum(0) for mask in eq], dim=0)
                sum_pijd = torch.stack([(sum_pijd[:, torch.BoolTensor(mask)]).sum(1) for mask in eq], dim=0)
                assert (sum_pijd >= 0).all()
                s += r[n][None, None, None, :] * sum_pijd # C C D K

                int_g = self._get_int_g(n)
                sum_int = (int_g[:, None, :] * eq.permute(1, 0)[:, :, None]).sum(0) # C D
                d += r[n][None, None, :] * sum_int[:, :, None] # C D K

            a = 1 / beta**2 # C x K 
            mu = 1e-7 + (-b[None, :] + (b[None, :]**2 - 4*a*c)**.5)/ (2*a)
            A = s / (Sigma**(-1) + d[None, :, :, :])
            assert (A >= 0).all()
            assert (mu > 0).all()

        return mu, A

    def update_num_clusters(self, nll):
        """
        Need doublechecking
        """
        for _ in range(10):
            if self.K == 1:
                qs = 1.
            else:
                qs = .5

            old_A = self.model.A
            old_mu = self.model.mu
            old_pi = self.model.pi
            old_K = self.K

            split = (random.random() < qs)
            if split: # perform split of cluster
                k = random.randint(0, old_K-1)
                a = torch.distributions.Beta(1, 1).sample()
                pi1 = a * old_pi[k:k+1]
                pi2 = (1 - a) * old_pi[k:k+1]
                pi = torch.cat([old_pi[:k], old_pi[k+1:], pi1, pi2], 0)
                assert pi.shape[0] == old_K+1, (old_pi.shape[0], pi.shape, old_K+1)

                A1 = 1. / (2 * a) * old_A[..., k:k+1]
                A2 = 1. / (2 * (1 - a)) * old_A[..., k:k+1]

                mu1 = 1. / (2 * a) * old_mu[..., k:k+1]
                mu2 = 1. / (2 * (1 - a)) * old_mu[..., k:k+1]

                A = torch.cat([old_A[..., :k], old_A[..., k+1:], A1, A2], -1)
                mu = torch.cat([old_mu[..., :k], old_mu[..., k+1:], mu1, mu2], -1)

                K = old_K + 1

            else: # perform merge of clusters
                (k1, k2) = np.sort(np.random.choice(np.arange(old_K), 2, replace=False))
                pi = torch.cat([old_pi[:k2], old_pi[k2+1:]], 0)
                pi[k1] = old_pi[k1] + old_pi[k2]

                A = torch.cat([old_A[..., :k2], old_A[..., k2+1:]], -1)
                A[..., k1] = old_pi[k1] / pi[k1] * old_A[..., k1] + old_pi[k2] / pi[k1] * old_A[..., k2]

                mu = torch.cat([old_mu[..., :k2], old_mu[..., k2+1:]], -1)
                mu[..., k1] = old_pi[k1] / pi[k1] * old_mu[..., k1] + old_pi[k2] / pi[k1] * old_mu[..., k2]

                K = old_K - 1

            Sigma = A
            B = (2 / np.pi)**.5 * mu

            new_nll = self.hp_nll(pi, mu, A)
            p = (torch.exp((nll - new_nll)/self.N) * \
                    self.model.p_A(A, Sigma) * self.model.p_mu(mu, B) * self.model.p_pi(pi, K) / \
                    (self.model.p_A() * self.model.p_mu() * self.model.p_pi() + 1e-5)).item()
            print(nll, new_nll, p, old_K, K)
            p = min(1, p)
            if random.random() < p:
                self.update_model(pi, mu, A)
                self.K = K
                nll = new_nll
