import numpy as np
import math
import matplotlib.pyplot as plt
from posicao_orbital import posicao_orbital
from metodo_newton_p import newton_projetado
from funcao_objetivo import funcao_objetivo

print("=== Teste 1: Ruido 0.1 km ===")

theta_init_circular = [7000.0, 0.01, math.pi/6, math.pi/4, math.pi/3, 0.0]

t_obs_circular = np.linspace(0, 2*3600, 50) # 2 horas de observacao
mu = 398600.4418
r_obs_circular = [posicao_orbital(t, theta_init_circular, mu) for t in t_obs_circular]

# Ruido 0.1 km
np.random.seed(123)  # para reprodutibilidade
ruido = 0.1
r_obs_ruidoso = [ [coord + ruido*np.random.randn() for coord in r] for r in r_obs_circular ]

# Executar otimizacao
theta_opt_ruidoso = newton_projetado(
    theta_init_circular,
    t_obs_circular,
    r_obs_ruidoso,
    mu,
    max_iter=500,
    c=1e-4
)

theta_opt_ruidoso = [float(x) for x in theta_opt_ruidoso]

theta_init_rounded = [round(x, 5) for x in theta_init_circular]
theta_opt_rounded = [round(x, 5) for x in theta_opt_ruidoso]

J_inicial = funcao_objetivo(theta_init_circular, t_obs_circular, r_obs_ruidoso, mu)
J_final = funcao_objetivo(theta_opt_ruidoso, t_obs_circular, r_obs_ruidoso, mu)
melhoria = 100 * (J_inicial - J_final) / J_inicial

print("Parâmetros iniciais: ", theta_init_rounded)
print("Parâmetros otimizados: ", theta_opt_rounded)
print("Função objetivo inicial: ", round(J_inicial, 5))
print("Função objetivo final: ", round(J_final, 5))
print("Melhoria na função objetivo: ", round(melhoria, 2), "%")

# --- Calcular resíduos ---
residuos = np.array(r_obs_ruidoso) - np.array([posicao_orbital(t, theta_opt_ruidoso, mu) for t in t_obs_circular])
norm_res = np.linalg.norm(residuos, axis=1)

# --- Plot ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Componentes x, y, z
axs[0].scatter(t_obs_circular, residuos[:,0], label='x', color='r')
axs[0].scatter(t_obs_circular, residuos[:,1], label='y', color='g')
axs[0].scatter(t_obs_circular, residuos[:,2], label='z', color='b')
axs[0].set_ylabel('Resíduo [km]')
axs[0].set_title('Resíduos por componente (x, y, z)')
axs[0].legend()
axs[0].grid(True)

# Norma do resíduo
axs[1].scatter(t_obs_circular, norm_res, label='||res||', color='k')
axs[1].set_xlabel('Tempo [s]')
axs[1].set_ylabel('Norma do resíduo [km]')
axs[1].set_title('Norma do resíduo ao longo do tempo')
axs[1].grid(True)

plt.tight_layout()
plt.show()
