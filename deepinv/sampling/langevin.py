import torch.nn as nn
import torch
import numpy as np
import time as time

import deepinv.optim
from deepinv.models import ScoreDenoiser
from tqdm import tqdm
from deepinv.optim.utils import check_conv


class Welford:
    r'''
     Welford's algorithm for calculating mean and variance

     https://doi.org/10.2307/1266577
    '''
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self):
        return self.M

    def var(self):
        return self.S / (self.k - 1)


def refl_projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=lower, max=upper)


class MCMC(nn.Module):
    r'''
        Base class for Markov Chain Monte Carlo sampling.

        This class can be used to create new MCMC samplers, by only defining their kernel inside a torch.nn.Module:

        ::

            # define custom Markov kernel
            class MyKernel(torch.nn.Module):
                def __init__(self, iterator_params)
                    super().__init__()
                    self.iterator_params = iterator_params

                def forward(self, x):
                    # run one sampling kernel iteration
                    new_x = f(x, iterator_params)
                    return new_x

            class MySampler(MCMC):
                def __init__(self, prior, data_fidelity, iterator_params,
                             max_iter=1e3, burnin_ratio=.1, clip=(-1,2), verbose=True):
                    # generate an iterator
                    iterator = MyKernel(step_size=step_size, alpha=alpha)
                    # set the params of the base class
                    super().__init__(iterator, prior, data_fidelity, alpha=alpha,  max_iter=max_iter,
                                     burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)

            # create the sampler
            sampler = MySampler(prior, data_fidelity, iterator_params)

            # compute posterior mean and variance of reconstruction of measurement y
            mean, var = sampler(y, physics)


        This class computes the mean and variance of the chain using Welford's algorithm, which avoids storing the whole
        MCMC chain.

        :param deepinv.models.ScoreDenoiser prior: negative log-prior based on a trained or model-based denoiser.
        :param deepinv.optim.DataFidelity data_fidelity: negative log-likelihood function linked with the
            noise distribution in the acquisition physics.
        :param int max_iter: number of Monte Carlo iterations.
        :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
            samples to compute posterior statistics).
        :param float burnin_ratio: percentage of iterations used for burn-in period, should be set between 0 and 1.
            The burn-in samples are discarded constant with a numerical algorithm.
        :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
            If ``None``, the algorithm will not project the samples.
        :param float crit_conv: Threshold for verifying the convergence of the mean and variance estimates.
        :param bool verbose: prints progress of the algorithm.

    '''
    def __init__(self, iterator: torch.nn.Module, prior: ScoreDenoiser, data_fidelity: deepinv.optim.DataFidelity,
                 max_iter=1e3, burnin_ratio=.2, thinning=10, clip=(-1., 2.), crit_conv=1e-3, verbose=False):
        super(MCMC, self).__init__()

        self.iterator = iterator
        self.prior = prior
        self.likelihood = data_fidelity
        self.C_set = clip
        self.thinning = thinning
        self.max_iter = int(max_iter)
        self.crit_conv = crit_conv
        self.burnin_iter = int(burnin_ratio*max_iter)
        self.verbose = verbose
        self.mean_convergence = False
        self.var_convergence = False

    def forward(self, y, physics, seed=None):
        r'''
        Runs an MCMC chain to obtain the posterior mean and variance of the reconstruction of the measurements y.

        :param torch.tensor y: Measurements
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param float seed: Random seed for generating the MCMC samples
        :return: (tuple of torch.tensor) containing the posterior mean and variance.
        '''
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            # Algorithm parameters
            if self.C_set:
                C_lower_lim = self.C_set[0]
                C_upper_lim = self.C_set[1]

            # Initialization
            x = physics.A_adjoint(y) #.cuda(device).detach().clone()

            # MCMC loop
            start_time = time.time()
            statistics = Welford(x)

            self.mean_convergence = False
            self.var_convergence = False
            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                x = self.iterator(x, y, physics, likelihood=self.likelihood,
                                  prior=self.prior)

                if self.C_set:
                    x = projbox(x, C_lower_lim, C_upper_lim)

                if it > self.burnin_iter and (it % self.thinning) == 0:
                    if it >= (self.max_iter - self.thinning):
                        mean_prev = statistics.mean().clone()
                        var_prev = statistics.var().clone()
                    statistics.update(x)

            if self.verbose:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                elapsed = end_time - start_time
                print(f'MCMC sampling finished! elapsed time={elapsed} seconds')

            if check_conv(mean_prev, statistics.mean(), it, self.crit_conv, self.verbose) and it>1:
                self.mean_convergence = True

            if check_conv(var_prev, statistics.var(), it, self.crit_conv, self.verbose) and it>1:
                self.var_convergence = True

        return statistics.mean(), statistics.var()

    def mean_has_converged(self):
        r'''
        Returns a boolean indicating if the posterior mean verifies the convergence criteria.
        '''
        return self.mean_convergence

    def var_has_converged(self):
        r'''
        Returns a boolean indicating if the posterior variance verifies the convergence criteria.
        '''
        return self.var_convergence


class ULAIterator(nn.Module):
    def __init__(self, step_size, alpha):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.noise_std = np.sqrt(2*step_size)

    def forward(self, x, y, physics, likelihood, prior):
        noise = torch.randn_like(x)*self.noise_std
        lhood = - likelihood.grad(x, y, physics)
        lprior = - prior(x) * self.alpha
        return x + self.step_size * (lhood+lprior) + noise


class ULA(MCMC):
    r'''
        Plug-and-Play Unadjusted Langevin Algorithm.

        The algorithm runs the following markov chain iteration
        https://arxiv.org/abs/2103.04715 :

        .. math::

            x_{k+1} = \Pi_{[a,b]} \left(x_{k} + \eta \nabla \log p(y|A,x_k) +
            \eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1} \right).

        where :math:`x_{k}` is the :math:`k` th sample of the Markov chain,
        :math:`\log p(y|x)` is the log-likelihood function, :math:`\log p(x)` is the log-prior
        :math:`\eta>0` is the step size, :math:`\alpha>0` controls the amount of regularization,
        :math:`\Pi_{[a,b]}(x)` projects the entries of :math:`x` to the interval :math:`[a,b]` and
        :math:`z\sim \mathcal{N}(0,I)` is a standard Gaussian vector.


        - PnP-ULA assumes that the denoiser is :math:`L`-Lipschitz differentiable
        - For convergence, ULA required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`

        :param deepinv.models.ScoreDenoiser prior: negative log-prior based on a trained or model-based denoiser.
        :param deepinv.optim.DataFidelity data_fidelity: negative log-likelihood function linked with the
            noise distribution in the acquisition physics.
        :param float step_size: step size :math:`\eta>0` of the algorithm.
            Tip: use :meth:`deepinv.physics.Physics.compute_norm()` to compute the Lipschitz constant of the forward operator.
        :param float alpha: regularization parameter :math:`\alpha`
        :param int max_iter: number of Monte Carlo iterations.
        :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
            samples to compute posterior statistics).
        :param float burnin_ratio: percentage of iterations used for burn-in period, should be set between 0 and 1.
            The burn-in samples are discarded constant with a numerical algorithm.
        :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
            If ``None``, the algorithm will not project the samples.
        :param float crit_conv: Threshold for verifying the convergence of the mean and variance estimates.
        :param bool verbose: prints progress of the algorithm.

    '''
    def __init__(self, prior, data_fidelity, step_size=1., alpha=1.,  max_iter=1e3, thinning=5, burnin_ratio=.2,
                 clip=(-1., 2.), crit_conv=1e-3, verbose=False):

        iterator = ULAIterator(step_size=step_size, alpha=alpha)
        super().__init__(iterator, prior, data_fidelity, max_iter=max_iter, thinning=thinning, crit_conv=crit_conv,
                         burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)


class SKRockIterator(nn.Module):
    def __init__(self, step_size, alpha, inner_iter, eta):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.eta = eta
        self.inner_iter = inner_iter
        self.noise_std = np.sqrt(2*step_size)

    def forward(self, x, y, physics, likelihood, prior):
        posterior = lambda u:  likelihood.grad(u, y, physics) \
                               + self.alpha * prior(u)

        # First kind Chebyshev function
        T_s = lambda s, u: np.cosh(s*np.arccosh(u))
        # First derivative Chebyshev polynomial first kind
        T_prime_s = lambda s, u: s*np.sinh(s*np.arccosh(u))/np.sqrt(u**2-1)

        w0 = 1 + self.eta/(self.inner_iter**2)  # parameter \omega_0
        w1 = T_s(self.inner_iter, w0)/T_prime_s(self.inner_iter, w0)  # parameter \omega_1
        mu1 = w1/w0  # parameter \mu_1
        nu1 = self.inner_iter*w1/2  # parameter \nu_1
        kappa1 = self.inner_iter*(w1/w0)  # parameter \kappa_1

        # sampling the variable x
        noise = np.sqrt(2*self.step_size)*torch.randn_like(x)  # diffusion term

        # first internal iteration (s=1)
        xts_2 = x.clone()
        xts = x.clone() - mu1*self.step_size*posterior(x + nu1*noise) + kappa1*noise

        for js in range(2, self.inner_iter+1):  # s=2,...,self.inner_iter SK-ROCK internal iterations
            xts_1 = xts.clone()
            mu = 2 * w1 * T_s(js-1, w0) / T_s(js, w0)  # parameter \mu_js
            nu = 2 * w0 * T_s(js-1, w0) / T_s(js, w0)  # parameter \nu_js
            kappa = 1-nu  # parameter \kappa_js
            xts = -mu * self.step_size*posterior(xts) + nu*xts + kappa*xts_2
            xts_2 = xts_1

        return xts  # new sample produced by the SK-ROCK algorithm


class SKRock(MCMC):
    r'''
        Plug-and-Play SKROCK algorithm.

        Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
        approximation to accelerate the standard Unadjusted Langevin Algorithm.

        https://arxiv.org/abs/1908.08845

        - SKROCK assumes that the denoiser is :math:`L`-Lipschitz differentiable
        - For convergence, SKROCK required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`

        :param deepinv.models.ScoreDenoiser prior: negative log-prior based on a trained or model-based denoiser.
        :param deepinv.optim.DataFidelity data_fidelity: negative log-likelihood function linked with the
            noise distribution in the acquisition physics.
        :param float step_size: Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
        :param float eta: :math:`\eta` SKROCK parameter.
        :param float alpha: regularization parameter :math:`\alpha`.
        :param int inner_iter: Number of inner SKROCK iterations.
        :param int max_iter: Number of outer iterations.
        :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
            samples to compute posterior statistics).
        :param float burnin_ratio: percentage of iterations used for burn-in period. The burn-in samples are discarded
            constant with a numerical algorithm.
        :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
            If ``None``, the algorithm will not project the samples.
        :param bool verbose: prints progress of the algorithm.

    '''
    def __init__(self, prior: ScoreDenoiser, data_fidelity, step_size=1., eta=0.05, alpha=1., inner_iter=10,
                 max_iter=1e3, thinning=10, burnin_ratio=.2, clip=(-1., 2.), crit_conv=1e-3, verbose=False):

        iterator = SKRockIterator(step_size=step_size, alpha=alpha, inner_iter=inner_iter, eta=eta)
        super().__init__(iterator, prior, data_fidelity, max_iter=max_iter, crit_conv=crit_conv, thinning=thinning,
                         burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)


if __name__ == "__main__":

    import deepinv as dinv
    import torchvision
    from deepinv.optim.data_fidelity import L2

    x = torchvision.io.read_image('../../datasets/celeba/img_align_celeba/085307.jpg')
    x = x.unsqueeze(0).float().to(dinv.device) / 255
    #physics = dinv.physics.CompressedSensing(m=50000, fast=True, img_shape=(3, 218, 178), device=dinv.device)
    #physics = dinv.physics.Denoising()
    physics = dinv.physics.Inpainting(mask=.95, tensor_size=(3, 218, 178), device=dinv.device)
    #physics = dinv.physics.BlurFFT(filter=dinv.physics.blur.gaussian_blur(sigma=(2,2)), img_size=x.shape[1:], device=dinv.device)

    sigma = .1
    physics.noise_model = dinv.physics.GaussianNoise(sigma)

    y = physics(x)

    likelihood = L2(sigma=sigma)

    #model_spec = {'name': 'median_filter', 'args': {'kernel_size': 3}}
    #model_spec = {'name': 'waveletprior', 'args': {'wv': 'db8', 'level': 4, 'device': dinv.device}}
    model_spec = {'name': 'dncnn', 'args': {'device': dinv.device, 'in_channels': 3, 'out_channels': 3,
                                            'pretrained': 'download_lipschitz'}}

    prior = ScoreDenoiser(model_spec=model_spec, sigma_denoiser=2/255)

    f = ULA(prior, likelihood, max_iter=10000, burnin_ratio=.3, verbose=True,
               alpha=.9, step_size=.01*(sigma**2), clip=(-1, 2))
    #f = SKRock(prior, likelihood, max_iter=1000, burnin_ratio=.3, verbose=True,
    #           alpha=.9, step_size=.1*(sigma**2), clip=(-1, 2))

    xmean, xvar = f(y, physics)

    print(str(f.mean_has_converged()))
    print(str(f.var_has_converged()))

    xnstd = xvar.sqrt()
    xnstd = xnstd/xnstd.flatten().max()

    dinv.utils.plot_debug([physics.A_adjoint(y), x, xmean, xnstd], titles=['meas.', 'ground-truth', 'mean', 'norm. std'])
