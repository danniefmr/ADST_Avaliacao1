import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Carregar os dados
# ---------------------------------------------------------
data = pd.read_csv('dataset.csv')
X1 = data.iloc[:, 0].values  # Débito
X2 = data.iloc[:, 1].values  # Jitter

# ---------------------------------------------------------
# 2. Normalizar as variáveis (Z-score: média 0, desvio-padrão 1)
# ---------------------------------------------------------
X1_norm = (X1 - np.mean(X1)) / np.std(X1, ddof=1)
X2_norm = (X2 - np.mean(X2)) / np.std(X2, ddof=1)

# Matriz de dados normalizados (2 x N)
X_norm = np.vstack((X1_norm, X2_norm))

# ---------------------------------------------------------
# 3. Matriz de Covariância dos dados normalizados (Cx)
# (Nota: Coincide com a matriz de correlação de Pearson)
# ---------------------------------------------------------
Cx = np.cov(X_norm, ddof=1)

print("=" * 60)
print("Matriz de Covariância com Dados Normalizados (Cx):")
print(f"[{Cx[0,0]:12.6f}  {Cx[0,1]:12.6f}]")
print(f"[{Cx[1,0]:12.6f}  {Cx[1,1]:12.6f}]")
print("=" * 60)

# ---------------------------------------------------------
# 4. Valores e Vetores Próprios
# ---------------------------------------------------------
valores_proprios, vetores_proprios = np.linalg.eig(Cx)

# Ordenar por ordem decrescente de lambda
indices_ordenados = np.argsort(valores_proprios)[::-1]
lambda1, lambda2 = valores_proprios[indices_ordenados]
P = vetores_proprios[:, indices_ordenados]

print(f"Valor Próprio lambda_1 : {lambda1:.6f}")
print(f"Valor Próprio lambda_2 : {lambda2:.6f}")
print("-" * 60)
print("Matriz de Transformação P (vetores próprios normalizados):")
print(P)
print("=" * 60)

# ---------------------------------------------------------
# 5. Aplicar a Transformação Z = P^T * X_norm
# ---------------------------------------------------------
Z = np.dot(P.T, X_norm)
Z1, Z2 = Z[0, :], Z[1, :]

# ---------------------------------------------------------
# 6. Verificação Computacional
# ---------------------------------------------------------
Cz = np.cov(Z, ddof=1)
rho_Z = np.corrcoef(Z1, Z2)[0, 1]

print("Matriz de Covariância C_Z (Diagonalizada):")
print(f"[{Cz[0,0]:12.6f}  {Cz[0,1]:12.6f}]")
print(f"[{Cz[1,0]:12.6f}  {Cz[1,1]:12.6f}]")
print("-" * 60)
print(f"Coeficiente de Correlação r(Z1, Z2) = {rho_Z:.8e}")
print("=" * 60)