import math
import numpy as np
from funcao_objetivo import funcao_objetivo
from calcular_gradiente import calcular_gradiente
from calcular_hessiana import calcular_hessiana

def newton_projetado(theta_init, t_obs, r_obs, mu,
                     alpha_init=1.0, beta=0.5, c=1e-4,
                     max_iter=500, tol=1e-6):

    theta = theta_init.copy()
    k = 0  # contador de iteracoes

    for iteration in range(max_iter):

        # Etapa 1
        J = funcao_objetivo(theta, t_obs, r_obs, mu)
        grad = calcular_gradiente(theta, t_obs, r_obs, mu)
        H = calcular_hessiana(theta, t_obs, r_obs, mu)
        
        # Etapa 2
        grad_norm = math.sqrt(sum(g*g for g in grad))
        if grad_norm < tol:
            break

        # Etapa 3
        d = resolver_sistema(H, [-g for g in grad])

        # Etapa 4
        alpha = busca_armijo(theta, J, grad, d, t_obs, r_obs, mu,
                             alpha_init, beta, c)
        
        # Etapa 5
        theta_new = [theta[i] + alpha * d[i] for i in range(len(theta))]
        
        # Etapa 6
        theta_new = projecao_total(theta_new)

        theta = theta_new

        k += 1
        
    print("Newton projetado convergiu em", k, "iteracoes")
    return theta

def busca_armijo(theta, J, grad, d, t_obs, r_obs, mu,
                 alpha_init, beta, c):

    alpha = alpha_init

    while True:

        theta_trial = [theta[i] + alpha * d[i] for i in range(len(theta))]
        theta_trial = projecao_total(theta_trial)

        J_trial = funcao_objetivo(theta_trial, t_obs, r_obs, mu)

        if J_trial <= J + c * alpha * sum(grad[i]*d[i]
                                          for i in range(len(theta))):
            break

        alpha *= beta

        if alpha < 1e-20:
            alpha = 0.0
            break

    return alpha

def projecao_total(theta):
    a, e, i, Omega, omega, tau = theta

    a = max(a, 1e-6)
    e = min(max(e, 0.0), 0.999)

    i = i % (2 * math.pi)
    Omega = Omega % (2 * math.pi)
    omega = omega % (2 * math.pi)

    return [a, e, i, Omega, omega, tau]

def resolver_sistema(H, b):
    H = np.array(H, dtype=float)
    b = np.array(b, dtype=float)
    return list(np.linalg.solve(H, b))
