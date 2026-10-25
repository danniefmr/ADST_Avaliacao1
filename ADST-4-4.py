import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Carregar Dados e Efetuar Transformação
# ---------------------------------------------------------
data = pd.read_csv('dataset.csv')
X1 = data.iloc[:, 0].values  # Débito (Throughput)
X2 = data.iloc[:, 1].values  # Jitter

# Médias e desvios-padrão amostrais
mu1, mu2 = np.mean(X1), np.mean(X2)
sigma1, sigma2 = np.std(X1, ddof=1), np.std(X2, ddof=1)

# Dados normalizados (Z-score)
X1_norm = (X1 - mu1) / sigma1
X2_norm = (X2 - mu2) / sigma2
X_norm = np.vstack((X1_norm, X2_norm))

# Covariância dos dados normalizados
Cx = np.cov(X_norm, ddof=1)
valores_proprios, vetores_proprios = np.linalg.eig(Cx)

# Ordenar por ordem decrescente de lambda
indices = np.argsort(valores_proprios)[::-1]
lambda1, lambda2 = valores_proprios[indices]
P = vetores_proprios[:, indices]  # Colunas: v1 e v2

# Dados transformados Z = P^T * X_norm
Z = np.dot(P.T, X_norm)
Z1, Z2 = Z[0, :], Z[1, :]

# ---------------------------------------------------------
# 2. Construção dos Gráficos Lado a Lado
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- GRÁFICO 1: Dados Normalizados e Vetores Próprios ---
ax1 = axes[0]
ax1.scatter(X1_norm, X2_norm, alpha=0.4, color='royalblue',
            label='Dados Normalizados (X1, X2)', s=15)
ax1.plot(0, 0, 'ro', markersize=8, label='Média dos dados normalizados (0, 0)')

# Desenhar os Vetores Próprios escalados pela raiz de lambda para visibilidade
# Se lambda2 == 0, usamos uma escala mínima para podermos ver a direção de v2
escala_v1 = np.sqrt(lambda1) if lambda1 > 0 else 1.0
escala_v2 = np.sqrt(lambda2) if lambda2 > 0 else escala_v1 * 0.2

# Vetor Próprio 1 (v1)
v1 = P[:, 0]
ax1.quiver(0, 0, v1[0]*escala_v1, v1[1]*escala_v1, angles='xy', scale_units='xy', scale=1,
           color='red', width=0.008, label='Vetor Próprio 1 (v1 - Maior Variância)')

# Vetor Próprio 2 (v2)
v2 = P[:, 1]
ax1.quiver(0, 0, v2[0]*escala_v2, v2[1]*escala_v2, angles='xy', scale_units='xy', scale=1,
           color='green', width=0.008, label='Vetor Próprio 2 (v2 - Ortogonal)')

ax1.set_title('Espaço Normalizado: Dados X com Vetores Próprios', fontsize=12, fontweight='bold')
ax1.set_xlabel('Débito / Throughput (X1 normalizado)', fontsize=11)
ax1.set_ylabel('Jitter (X2 normalizado)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', fontsize=9)

# --- GRÁFICO 2: Dados Transformados no Espaço Z ---
ax2 = axes[1]
ax2.scatter(Z1, Z2, alpha=0.4, color='darkorange', label='Dados Transformados (Z1, Z2)', s=15)
ax2.axhline(0, color='black', linestyle='--', linewidth=1, label='Eixo Z1 (Componente Principal 1)')
ax2.axvline(0, color='gray', linestyle=':', linewidth=1, label='Eixo Z2 (Componente Principal 2)')

ax2.set_title('Espaço Transformado: Componentes Principais Z = Pᵀ X_norm', fontsize=12, fontweight='bold')
ax2.set_xlabel('Componente Principal 1 (Z1) - Contém Variância λ1', fontsize=11)
ax2.set_ylabel('Componente Principal 2 (Z2) - Variância λ2', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()