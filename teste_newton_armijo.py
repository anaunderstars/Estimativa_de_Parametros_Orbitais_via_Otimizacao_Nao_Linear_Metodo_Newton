import numpy as np

# Funcao teste: J(theta) = (theta1-1)^2 + (theta2+2)^2
def funcao_objetivo_simples(theta):
    return (theta[0]-1)**2 + (theta[1]+2)**2

def gradiente_simples(theta):
    return [2*(theta[0]-1), 2*(theta[1]+2)]

def hessiana_simples(theta):
    return [[2, 0], [0, 2]]

# Busca de Armijo
def busca_armijo(theta, J, grad, d, alpha_init=1.0, beta=0.5, c=1e-4):
    alpha = alpha_init
    while True:
        theta_trial = [theta[i] + alpha * d[i] for i in range(len(theta))]
        J_trial = funcao_objetivo_simples(theta_trial)
        if J_trial <= J + c * alpha * sum(grad[i]*d[i] for i in range(len(theta))):
            break
        alpha *= beta
        if alpha < 1e-20:
            alpha = 0.0
            break
    return alpha


# Newton projetado
def newton_projetado_simples(theta_init, max_iter=20, tol=1e-8):
    theta = theta_init.copy()
    for k in range(max_iter):
        J = funcao_objetivo_simples(theta)
        grad = gradiente_simples(theta)
        H = np.array(hessiana_simples(theta))
        d = np.linalg.solve(H, [-g for g in grad])
        alpha = busca_armijo(theta, J, grad, d)
        theta = [theta[i] + alpha*d[i] for i in range(len(theta))]
        grad_norm = np.linalg.norm(grad)
        print(f"Iter {k+1}: theta = {theta}, J = {J:.6f}, ||grad|| = {grad_norm:.6f}, alpha = {alpha:.4f}")
        if grad_norm < tol:
            print("Convergência atingida!")
            break
    return theta

# Teste
theta_init = [0.0, 0.0]  # chute inicial longe do minimo
theta_opt = newton_projetado_simples(theta_init)

print("\nParâmetros otimizados:", theta_opt)
print("Função objetivo final:", funcao_objetivo_simples(theta_opt))
