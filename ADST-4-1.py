import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Carregar o dataset.csv e extrair as variáveis
# ---------------------------------------------------------
# Garante que o ficheiro 'dataset.csv' está na mesma pasta do script
data = pd.read_csv('dataset.csv')

# Extrair as duas primeiras colunas
X1 = data.iloc[:, 0].values  # Throughput (Débito)
X2 = data.iloc[:, 1].values  # Jitter

# ---------------------------------------------------------
# 2. Calcular o Coeficiente de Correlação de Pearson
# ---------------------------------------------------------
# Pode ser calculado com numpy.corrcoef ou pandas.corr
matriz_corr = np.corrcoef(X1, X2)
rho_pearson = matriz_corr[0, 1]

print("=" * 50)
print(f"Coeficiente de Correlação de Pearson (rho): {rho_pearson:.6f}")
print("=" * 50)

# Interpretação rápida da correlação
if abs(rho_pearson) > 0.7:
    print("Existe uma forte correlação linear entre Throughput e Jitter.")
elif abs(rho_pearson) > 0.3:
    print("Existe uma correlação linear moderada entre Throughput e Jitter.")
else:
    print("Existe uma fraca correlação linear entre Throughput e Jitter.")

# ---------------------------------------------------------
# 3. Representação Gráfica (Débito vs Jitter)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X2, X1, alpha=0.5, color='blue', edgecolors='k', s=20)
plt.title('Débito (Throughput) em função do Jitter (Dados Reais 5G)')
plt.xlabel('Jitter (X2)')
plt.ylabel('Débito / Throughput (X1)')
plt.grid(True, linestyle='--', alpha=0.6)

# Adicionar o valor do coeficiente de correlação no gráfico
plt.text(0.05, 0.92, f'Pearson r = {rho_pearson:.4f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()