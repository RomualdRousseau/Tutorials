from functools import partial

from tutorial2.pyflow.functions import wu_adadelta, wu_adam, wu_rmsprop


def adatdelta(rho=0.95):
    return partial(wu_adadelta, rho=rho)


def rmsprop(rho=0.9, lr=0.1):
    return partial(wu_rmsprop, rho=rho, lr=lr)


def adam(beta1=0.9, beta2=0.999, lr=0.001):
    return partial(wu_adam, beta1=beta1, beta2=beta2, lr=lr)