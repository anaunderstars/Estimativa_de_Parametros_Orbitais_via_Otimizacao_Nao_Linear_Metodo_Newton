import numpy as np
import math

from posicao_orbital import posicao_orbital
from metodo_newton_p import newton_projetado
from funcao_objetivo import funcao_objetivo

print("=== Teste 1: Orbita Circular ===")

theta_init_circular = [7000.0, 0.01, math.pi/6, math.pi/4, math.pi/3, 0.0]

t_obs_circular = np.linspace(0, 2*3600, 50) # 2 horas de observacao

mu = 398600.4418
r_obs_circular = [posicao_orbital(t, theta_init_circular, mu) for t in t_obs_circular]

# Executar otimizacao
theta_opt_circular = newton_projetado(
    theta_init_circular,
    t_obs_circular,
    r_obs_circular,
    mu,
    max_iter=500,
    c=1e-4
)

theta_opt_circular = [float(x) for x in theta_opt_circular]

print("Parametros iniciais: ", theta_init_circular)
print("Parametros otimizados: ", theta_opt_circular)
print("Funcao objetivo inicial: ", funcao_objetivo(theta_init_circular, t_obs_circular, r_obs_circular, mu))
print("Funcao objetivo final: ", funcao_objetivo(theta_opt_circular, t_obs_circular, r_obs_circular, mu))
