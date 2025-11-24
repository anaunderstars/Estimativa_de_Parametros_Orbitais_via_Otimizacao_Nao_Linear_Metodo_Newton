import math
from posicao_orbital import posicao_orbital

def funcao_objetivo ( theta , t_obs , r_obs , mu ) :
    total_error = 0.0
    for k in range ( len ( t_obs ) ) :
        r_pred = posicao_orbital ( t_obs [ k ] , theta , mu )
        residual = [ r_obs [ k ][0] - r_pred [0] ,
        r_obs [ k ][1] - r_pred [1] ,
        r_obs [ k ][2] - r_pred [2]]
        total_error += residual [0]**2 + residual [1]**2 + residual [2]**2
    return total_error
