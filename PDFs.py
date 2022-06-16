'''
PDFs to be used for various distributions
log the data
use log distribution to compute pdf or pmf
add probailities instead of multiply

NOTE: the pdfs have been simplified to remove calculation of components that
are not needed for comparing probability proportions for samples
drawn from the same distributions. Therefore the PDFs do not generate
complete distributions

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gammaDistribution
from scipy.stats import invgamma as inverseGammaDistribution
from scipy.stats import loggamma as logGammaDistribution
from scipy.stats import poisson as poissonDistribution
import math
from scipy.special import beta as betaFunction
import scipy.stats as stats
import scipy
from scipy.special import loggamma

# normal
# def normalPDF(x, args):
#     mu = args["mu"]
#     sigma = args["sigma"]
#     try:
#         density = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
#     except:
#         density = 0
#     return density

def logNormalPDF(x, args):
    mu = args["mu"]
    sigma = args["sigma"]
    # density = stats.lognorm.pdf(x, s=1, loc=mu, scale=sigma)
    density = -(1 / 2) * (np.log(sigma) - (sigma * ((x - mu) ** 2)))
    return density * -1

# Gamma
# def gammaPDF(x, args):
#     shape = args["shape"]
#     scale = args["scale"]
#     return gammaDistribution.pdf(x, a=shape, scale=scale)

def logGammaPDF(x, args):
    shape = args["shape"]
    scale = args["scale"]
    if x < 0:
        density = 0
    else:
        density = loggamma(shape) + scale * np.log(scale) + (shape - 1) * np.log(x)
    return density * -1
    # return logGammaDistribution.pdf(x, c=shape, scale=scale)


def inverseLogGammaPDF(x, args):
    if x < 0:
        density = 0
    else:
        density = logGammaPDF(1 / x, args)
    return density * -1

# def inverseGammaPDF(x, args):
#     if x < 0:
#         density = 0
#     else:
#         density = gammaPDF(x, args)
#     return density
#
# xs = np.linspace(0,10,100)
# args = {"shape":2,"scale":7.5}
# ys1 = [gammaPDF(x, args) for x in xs]
# ys2 = [logGammaPDF(x, args) for x in xs]
# plt.title("vanilla gamma")
# plt.plot(xs, ys1, label="vanilla")
# plt.plot(xs, ys2, label="log")
# plt.legend()
# plt.show()

# xs = np.linspace(0,10,100)
# args = {"shape":2,"scale":7.5}
# ys1 = [inverseGammaPDF(x, args) for x in xs]
# ys2 = [inverseLogGammaPDF(x, args) for x in xs]
# plt.title("inverse gamma")
# plt.plot(xs, ys1, label="typical")
# plt.plot(xs, ys2, label="log")
# plt.legend()
# plt.show()

# Poisson
# def poissonPMF(k, args):
#     if k != int(k):
#         return 0
#     if k < 1:
#         return 0
#     lamda = args["lamda"]
#     density = np.log(poissonDistribution.pmf(k=k, mu=lamda))
#     return density * -1

def logPoissonPMF(k, args):
    if k != int(k):
        return 0
    if k < 0:
        return 0
    lamda = args["lamda"]
    density = -lamda + (k * np.log(lamda) - (loggamma(k - 1)))
    return density * -1
    # return np.log(poissonDistribution.pmf(k=k, mu=lamda))
# xs = np.arange(1,20)
# lamda = 2
# args = {"lamda":lamda}
# ys = [logPoissonPMF(k, args) for k in xs]
# plt.plot(xs, ys)
# plt.title("example log poisson distribution")
# plt.show()
# Beta
# def betaPDF(x, args):
#     if x < 0:
#         return 0
#     if x > 1:
#         return 0
#
#     alpha = args["alpha"]
#     beta = args["beta"]
#     density = x ** (alpha - 1) * (1 - x) ** (beta - 1) / betaFunction(alpha, beta)
#     return density

def logBetaPDF(x, args):
    if x < 0:
        return 0
    if x > 1:
        return 0

    alpha = args["alpha"]
    beta = args["beta"]
    density = loggamma(alpha + beta) - loggamma(alpha) - loggamma(beta) + ((alpha - 1) * np.log(x)) + ((beta - 1) * np.log(1 - x))
    return density * -1
    # return np.log(x ** (alpha - 1) * (1 - x) ** (beta - 1) / betaFunction(alpha, beta))


#
# #
# xs = np.linspace(0,1,100)
# alpha = 1
# beta = 0.0001
# args = {"alpha":alpha, "beta":beta}
# ys = [logBetaPDF(x, args) for x in xs]
# plt.plot(xs, ys)
# plt.show()
