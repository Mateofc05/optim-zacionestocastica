import random
import numpy as np
import pandas as pd

palabra = 'mate'
letras = 'abcdefghijklmnopqrstuvwxyz'

# =====================================================
# Simulacion: el chango teclea letras al azar
# =====================================================
n = 0            # cuantas letras lleva bien
intento = ''
num_intento = 1

while n < len(palabra):
    letra = random.choice(letras)
    intento = intento + letra

    if letra == palabra[n]:
        n = n + 1
        print("Intento", num_intento, "- letra correcta:", letra, "- progreso:", palabra[:n])
    else:
        if n > 0:
            print("Intento", num_intento, "- fallo con:", letra, "- se pierde:", intento)
        n = 0
        intento = ''
        num_intento = num_intento + 1

print("Palabra encontrada:", palabra, "en el intento", num_intento)


# =====================================================
# Cadena de Markov
# Estados: 0, 1, ..., N  (N = ya escribio la palabra)
# =====================================================
N = len(palabra)
p = 1 / 26       # probabilidad de acertar la letra que sigue

# Matriz de transicion
P = np.zeros((N + 1, N + 1))
for i in range(N):
    P[i, i + 1] = p        # acierta y avanza
    P[i, 0] = 1 - p        # falla y regresa al inicio
P[N, N] = 1                # estado absorbente

print("Matriz de transicion P:")
print(pd.DataFrame(P.round(4)))
print("Cada renglon suma 1:", np.allclose(P.sum(axis=1), 1))

# Vector inicial (todos los estados igual de probables)
pi0 = np.ones(N + 1) / (N + 1)
print("\n2) Vector inicial pi0 =", pi0.round(4))

# Distribucion de X_n usando descomposicion espectral
#    P^n = V * diag(lambda^n) * V^-1  y  pi_n = pi0 * P^n
valores, V = np.linalg.eig(P)
V_inv = np.linalg.inv(V)

pasos = 20
espectral = np.zeros((pasos, N + 1))
for n in range(1, pasos + 1):
    Pn = (V @ np.diag(valores ** n) @ V_inv).real
    espectral[n - 1] = pi0 @ Pn

print("Distribucion de X_n con descomposicion espectral (n = 1..20):")
print(pd.DataFrame(espectral.round(4), index=range(1, pasos + 1)))

# Lo mismo pero con la formula recurrente pi_n = pi_(n-1) * P
recurrente = np.zeros((pasos, N + 1))
pi = pi0.copy()
for n in range(pasos):
    pi = pi @ P
    recurrente[n] = pi

print("Distribucion de X_n con la formula recurrente:")
print(pd.DataFrame(recurrente.round(4), index=range(1, pasos + 1)))

print("\nLas dos tablas coinciden a 4 decimales:",
      np.array_equal(espectral.round(4), recurrente.round(4)))

# Probabilidad de que eventualmente escriba la palabra
#    El estado N es absorbente y se puede llegar desde cualquier estado,
#    asi que la probabilidad es 1. Lo grande es el tiempo de espera (~26^N).
prob_final = (pi0 @ np.linalg.matrix_power(P, 10000000))[N]
print("P(escribe la palabra eventualmente) =", round(prob_final, 6))
print("   Con solo", pasos, "pasos la probabilidad es apenas", round(recurrente[-1, N], 6))
print("   Pasos esperados (orden de magnitud): 26^" + str(N), "=", 26 ** N)
