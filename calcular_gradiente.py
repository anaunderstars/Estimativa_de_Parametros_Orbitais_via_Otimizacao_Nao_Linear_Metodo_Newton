import math
from funcao_objetivo import funcao_objetivo

def calcular_gradiente (theta,t_obs,r_obs,mu,h =1e-8) :
    grad = [0.0] * len (theta)
    J0 = funcao_objetivo (theta,t_obs,r_obs,mu)

    for i in range(len(theta)) :
        theta_perturbed = theta.copy()
        theta_perturbed[i] += h

        J_perturbed = funcao_objetivo (theta_perturbed,t_obs,r_obs,mu)
        grad[i] = (J_perturbed - J0 )/h
    return grad
