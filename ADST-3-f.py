import numpy as np
import sympy as sp

# ---------------------------------------------------------
# Dados do problema (Probabilidades para k = 0, 1, 2, 3, 4)
# ---------------------------------------------------------
N = 10**6
p0 = 149800 / N
p1 = 250600 / N
p2 = 349500 / N
p3 = 180200 / N
p4 = 69900 / N

p_K = [p0, p1, p2, p3, p4]

# ---------------------------------------------------------
# Método 1: Cálculo das Probabilidades via Convolução
# ---------------------------------------------------------
# A multiplicação de PGFs equivale à convolução dos vetores de probabilidade
p_S = np.convolve(p_K, np.convolve(p_K, p_K))

print("=" * 50)
print("  PROBABILIDADES P(S = s) PARA s DE 0 A 12")
print("=" * 50)

for s, p in enumerate(p_S):
    print(f"P(S = {s:2d}) = {p:.8f}  ({p * 100:6.3f}%)")

print("-" * 50)
print(f"Soma de todas as probabilidades: {sum(p_S):.6f} (Deve ser igual a 1)")
print("=" * 50)

# ---------------------------------------------------------
# Método 2: Expansão Simbólica de G_S(z) usando SymPy
# ---------------------------------------------------------
z = sp.Symbol('z')
G_K = p0 + p1*z + p2*(z**2) + p3*(z**3) + p4*(z**4)
G_S = sp.expand(G_K**3)

print("\nPolinómio Expandido G_S(z):")
print(G_S)

# ---------------------------------------------------------
# Cálculo da média E[S] = G_S'(1)
# ---------------------------------------------------------
G_S_derivada = sp.diff(G_S, z)
E_S = G_S_derivada.subs(z, 1)

print("\nCálculo da média E[S] = G_S'(1):")
print(f"G_S'(1) = {float(E_S):.4f}")

# ---------------------------------------------------------
# Cálculo da variância Var(S) = G_S''(1) + G_S'(1) - [G_S'(1)]^2
# ---------------------------------------------------------
G_S_segunda_derivada = sp.diff(G_S, z, 2)
G_S_segunda_derivada_em_1 = G_S_segunda_derivada.subs(z, 1)
Var_S = G_S_segunda_derivada_em_1 + E_S - E_S**2

print("\nCálculo da variância Var(S):")
print(f"G_S''(1) = {float(G_S_segunda_derivada_em_1):.4f}")
print(f"Var(S) = {float(Var_S):.4f}")