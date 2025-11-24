import numpy as np
import math

from posicao_orbital import posicao_orbital
from metodo_newton_p import newton_projetado
from funcao_objetivo import funcao_objetivo

print("=== Teste 2: Orbita Elíptica ===")

theta_init_eliptica = [10000.0, 0.3, math.pi/4, math.pi/3, math.pi/2, 100.0]

t_obs_eliptica = np.linspace(0, 4*3600, 50) # 4 horas de observacao

mu = 398600.4418
r_obs_eliptica = [posicao_orbital(t, theta_init_eliptica, mu) for t in t_obs_eliptica]

# Executar otimizacao
theta_opt_eliptica = newton_projetado(
    theta_init_eliptica,
    t_obs_eliptica,
    r_obs_eliptica,
    mu,
    max_iter=500,
    c=1e-4
)

theta_opt_eliptica = [float(x) for x in theta_opt_eliptica]

print("Parametros iniciais: ", theta_init_eliptica)
print("Parametros otimizados: ", theta_opt_eliptica)
print("Funcao objetivo inicial: ", funcao_objetivo(theta_init_eliptica, t_obs_eliptica, r_obs_eliptica, mu))
print("Funcao objetivo final: ", funcao_objetivo(theta_opt_eliptica, t_obs_eliptica, r_obs_eliptica, mu))
