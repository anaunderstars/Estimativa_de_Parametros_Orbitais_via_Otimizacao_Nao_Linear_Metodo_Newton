import math
from calcular_gradiente import calcular_gradiente

def calcular_hessiana(theta,t_obs,r_obs,mu,h=1e-8):
    n = len(theta)
    H = [[0.0] * n for _ in range(n)]

    grad0 = calcular_gradiente(theta,t_obs,r_obs,mu,h)

    for j in range(n):
        theta_perturbed = theta.copy()
        theta_perturbed[j] += h

        grad_perturbed = calcular_gradiente(theta_perturbed,t_obs,r_obs,mu,h)

        for i in range(n):
            H[i][j] = (grad_perturbed[i] - grad0[i]) / h

    return H
