import math

def project_to_feasible(theta):
    a, e, i, Omega, omega, Tau = theta

    a_proj = max(a, 1e-6)
    e_proj = max(1e-6, min(e, 1 - 1e-6))
    i_proj = max(0, min(i, math.pi))
    Omega_proj = Omega % (2 * math.pi)
    omega_proj = omega % (2 * math.pi)
    Tau_proj = Tau

    return [a_proj, e_proj, i_proj, Omega_proj, omega_proj, Tau_proj]
