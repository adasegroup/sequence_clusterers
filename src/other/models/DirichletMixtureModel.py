import numpy as np
import torch

from utils.preprocessing import load_data_dmhp


class DirichletMixtureModel:
    """
    Dirichlet Mixture Model for Hawkes Processes
    """

    def __init__(self, K, C, D, alpha, B=None, Sigma=None):
        """
        K - number of clusters
        C - number of event classes
        D - number of basis functions
        alpha - parameter of Dirichlet distribution
        B - parameter of Rayleigh distribution (prior of \mu)
        Sigma - parameter of Exp distribution (prior of A)
        """
        self.alpha = alpha
        self.K = K
        if Sigma is not None:
            assert Sigma.shape == [C, C, D, K]
            self.Sigma = Sigma
        else:
            self.Sigma = torch.rand(C, C, D, K)
        if B is not None:
            assert B.shape == [C, K]
            self.B = B
        else:
            self.B = torch.rand(C, K)

        concentration = torch.FloatTensor([alpha / float(K)] * K)
        self.dirich = torch.distributions.dirichlet.Dirichlet(concentration)
        self.pi = self.dirich.sample()
        self.cat = torch.distributions.categorical.Categorical(self.pi)
        self.k_pi = self.cat.sample()
        self.rayleigh = torch.distributions.weibull.Weibull(
            2 ** 0.5 * self.B, 2 * torch.ones_like(self.B)
        )
        self.mu = self.rayleigh.sample()
        self.exp = torch.distributions.exponential.Exponential((self.Sigma) ** (-1))
        self.A = self.exp.sample()  # C x C x D x K

    def p_mu(self, mu=None, B=None):
        mu = mu if mu is not None else self.mu
        if B is not None:
            return torch.prod(
                torch.exp(
                    torch.distributions.Weibull(
                        2 ** 0.5 * B, 2 * torch.ones_like(B)
                    ).log_prob(mu)
                )
            )
        else:
            return torch.prod(torch.exp(self.rayleigh.log_prob(mu)))

    def p_A(self, A=None, Sigma=None):
        A = A if A is not None else self.A
        if Sigma is not None:
            return torch.prod(
                torch.exp(torch.distributions.Exponential((Sigma) ** (-1)).log_prob(A))
            )
        else:
            return torch.prod(torch.exp(self.exp.log_prob(A)))

    def p_pi(self, pi=None, K=None):
        pi = pi if pi is not None else self.pi
        if K is not None:
            concentration = torch.FloatTensor([self.alpha / float(K)] * K)
            return torch.prod(
                torch.exp(torch.distributions.Dirichlet(concentration).log_prob(pi))
            )
        return torch.prod(torch.exp(self.dirich.log_prob(pi)))

    def e_logpi(self):
        """
        Returns expectation of logarithm of \pi
        """
        return (
            torch.digamma(torch.FloatTensor([self.alpha / self.K]))
            - torch.digamma(torch.FloatTensor([self.alpha]))
        ).repeat(self.K)

    def e_mu(self):
        """
        Returns expectation of \mu
        """
        return (np.pi / 2) ** 0.5 * self.B

    def e_A(self):
        """
        Returns expectation of \A
        """
        return self.Sigma

    def var_mu(self):
        """
        Returns variance of \mu
        """
        return (4 - np.pi) / 2 * self.B ** 2

    def var_A(self):
        """
        Returns variance of \A
        """
        return self.Sigma ** 2

    def update_A(self, A, Sigma):
        self.A = A
        self.Sigma = Sigma
        self.exp = torch.distributions.exponential.Exponential((Sigma) ** (-1))

    def update_mu(self, mu, B):
        self.mu = mu
        self.B = B
        self.rayleigh = torch.distributions.weibull.Weibull(
            2 ** 0.5 * B, 2 * torch.ones_like(B)
        )

    def update_pi(self, pi):
        self.pi = pi
        self.K = pi.shape[0]
        concentration = torch.FloatTensor([self.alpha / float(self.K)] * self.K)
        self.dirich = torch.distributions.Dirichlet(concentration)
