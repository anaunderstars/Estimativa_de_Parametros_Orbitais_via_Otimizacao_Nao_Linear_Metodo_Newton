import math
from resolucao_kepler import resolucao_kepler

def posicao_orbital(t, theta, mu):

    a, e, i, Omega, omega, tau = theta

    n = math.sqrt(mu / a**3)
    M = n * (t - tau)
    E = resolucao_kepler(M, e)

    x = a * (math.cos(E) - e)
    y = a * math.sqrt(1 - e**2) * math.sin(E)
    z = 0.0

    cos_Omega = math.cos(Omega)
    sin_Omega = math.sin(Omega)
    cos_omega = math.cos(omega)
    sin_omega = math.sin(omega)
    cos_i = math.cos(i)
    sin_i = math.sin(i)

    X = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x + \
        (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y

    Y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x + \
        (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y

    Z = (sin_omega * sin_i) * x + (cos_omega * sin_i) * y

    return [X, Y, Z]
