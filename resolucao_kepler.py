import math

def resolucao_kepler(M, e, tol=1e-12, max_iter=50):

    if e < 0 or e >= 1:
        raise ValueError('Excentricidade deve estar no intervalo [0 , 1)')

    E = M

    for i in range(max_iter):
        f = E - e * math.sin(E) - M
        f_primo = 1 - e * math.cos(E)

        if abs(f_primo) < 1e-15:
            break

        delta_E = -f / f_primo
        E += delta_E

        if abs(delta_E) < tol:
            break

    return E
