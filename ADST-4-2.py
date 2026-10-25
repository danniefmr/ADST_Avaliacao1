import numpy as np
import pandas as pd

# 1. Carregar os dados
data = pd.read_csv('dataset.csv')
X1 = data.iloc[:, 0].values  # Débito (Throughput)
X2 = data.iloc[:, 1].values  # Jitter

# Unir num vetor de dados X (dimensão 2 x N)
X = np.vstack((X1, X2))

# 2. Calcular as Médias (mu1, mu2)
mu1 = np.mean(X1)
mu2 = np.mean(X2)
mu = np.array([[mu1], [mu2]])

print("=" * 50)
print(f"Média do Débito (mu1): {mu1:.4f}")
print(f"Média do Jitter (mu2): {mu2:.4f}")
print("=" * 50)

# 3. Centrar os Dados (Xc = X - mu)
Xc1 = X1 - mu1
Xc2 = X2 - mu2
Xc = np.vstack((Xc1, Xc2))  # Matriz de dados centrados (2 x N)

# 4. Determinar a Matriz de Covariância (Cx)
# Nota: ddof=0 para a covariância populacional ou ddof=1 para amostral (geralmente ddof=1)
Cx = np.cov(Xc, ddof=1)

print("Matriz de Covariância Cx:")
print(f"[{Cx[0,0]:12.6f}  {Cx[0,1]:12.6f}]")
print(f"[{Cx[1,0]:12.6f}  {Cx[1,1]:12.6f}]")
print("=" * 50)

# Confirmação dos elementos da matriz:
print(f"Variância de X1 (sigma1^2)  : {Cx[0,0]:.6f}")
print(f"Variância de X2 (sigma2^2)  : {Cx[1,1]:.6f}")
print(f"Covariância (sigma12)       : {Cx[0,1]:.6f}")