import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Carregar os dados e centrar
# ---------------------------------------------------------
data = pd.read_csv('dataset.csv')
X1 = data.iloc[:, 0].values  # Débito
X2 = data.iloc[:, 1].values  # Jitter

# Matriz de dados centrados (2 x N)
Xc = np.vstack((X1 - np.mean(X1), X2 - np.mean(X2)))

# Matriz de Covariância Cx
Cx = np.cov(Xc, ddof=1)

# ---------------------------------------------------------
# 2. Valores e Vetores Próprios de Cx
# ---------------------------------------------------------
# np.linalg.eig calcula os valores e vetores próprios
valores_proprios, vetores_proprios = np.linalg.eig(Cx)

# Ordenar por ordem decrescente de valores próprios (lambda1 >= lambda2)
indices_ordenados = np.argsort(valores_proprios)[::-1]
lambda1, lambda2 = valores_proprios[indices_ordenados]

# Matriz P com os vetores próprios normalizados nas colunas
P = vetores_proprios[:, indices_ordenados]

print("=" * 60)
print(f"Valor Próprio lambda_1 (maior variância) : {lambda1:.6f}")
print(f"Valor Próprio lambda_2 (menor variância) : {lambda2:.6f}")
print("-" * 60)
print("Matriz de Transformação P (vetores próprios nas colunas):")
print(P)
print("=" * 60)

# ---------------------------------------------------------
# 3. Aplicar a Transformação Linear Z = P^T * Xc
# ---------------------------------------------------------
# Z terá dimensão (2 x N), onde a linha 0 é Z1 e a linha 1 é Z2
Z = np.dot(P.T, Xc)
Z1 = Z[0, :]
Z2 = Z[1, :]

# ---------------------------------------------------------
# 4. Verificação Computacional da Descorrelação
# ---------------------------------------------------------
# Matriz de Covariância das variáveis transformadas CZ
Cz = np.cov(Z, ddof=1)

# Coeficiente de correlação de Pearson entre Z1 e Z2
rho_Z = np.corrcoef(Z1, Z2)[0, 1]

print("Matriz de Covariância C_Z (deve ser diagonal com lambda1 e lambda2):")
print(f"[{Cz[0,0]:12.6f}  {Cz[0,1]:12.6f}]")
print(f"[{Cz[1,0]:12.6f}  {Cz[1,1]:12.6f}]")
print("-" * 60)
print(f"Coeficiente de Correlação de Pearson r(Z1, Z2) = {rho_Z:.8e}")

if np.isclose(rho_Z, 0, atol=1e-10):
    print("-> CONFIRMADO: As novas componentes Z1 e Z2 estão descorrelacionadas!")
print("=" * 60)