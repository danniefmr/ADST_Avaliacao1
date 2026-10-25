import numpy as np
import matplotlib.pyplot as plt

# 1. Parâmetros da Simulação
N = 1_000_000  # 10^6 amostras
lam = 0.4      # Parâmetro da exponencial

# Fixar a semente para resultados reprodutíveis (opcional)
np.random.seed(42)

# 2. Geração das variáveis aleatórias X (Exponencial) e Y (Uniforme)
# Nota: Em numpy, a distribuição exponencial usa o parâmetro 'scale' = 1/lambda
X = np.random.exponential(scale=1/lam, size=N)
Y = np.random.uniform(0, 2, size=N)

# 3. Cálculo das variáveis combinadas
S = X + Y  # Soma (Caso 1)
D = X - Y  # Diferença (Caso 2)

# 4. Definição das Funções Teóricas das PDFs
def pdf_S(s):
    res = np.zeros_like(s)
    # 0 <= s <= 2
    m1 = (s >= 0) & (s <= 2)
    res[m1] = 0.5 * (1 - np.exp(-lam * s[m1]))
    # s > 2
    m2 = s > 2
    res[m2] = 0.5 * np.exp(-lam * s[m2]) * (np.exp(lam * 2) - 1)
    return res

def pdf_D(d):
    res = np.zeros_like(d)
    # -2 <= d <= 0
    m1 = (d >= -2) & (d <= 0)
    res[m1] = 0.5 * (1 - np.exp(-lam * (d[m1] + 2)))
    # d > 0
    m2 = d > 0
    res[m2] = 0.5 * (np.exp(-lam * d[m2]) - np.exp(-lam * (d[m2] + 2)))
    return res

# 5. Plotagem dos Gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico para S = X + Y
ax1.hist(S, bins=120, density=True, alpha=0.6, color='skyblue', label='Simulação ($10^6$ amostras)')
s_vals = np.linspace(0, max(S), 500)
ax1.plot(s_vals, pdf_S(s_vals), 'r-', linewidth=2.5, label='PDF Teórica')
ax1.set_title('Soma dos Atrasos: $S = X + Y$', fontsize=12, fontweight='bold')
ax1.set_xlabel('s (Tempo Total)')
ax1.set_ylabel('Densidade de Probabilidade')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Gráfico para D = X - Y
ax2.hist(D, bins=120, density=True, alpha=0.6, color='lightgreen', label='Simulação ($10^6$ amostras)')
d_vals = np.linspace(min(D), max(D), 500)
ax2.plot(d_vals, pdf_D(d_vals), 'r-', linewidth=2.5, label='PDF Teórica')
ax2.set_title('Diferença de Atrasos: $D = X - Y$', fontsize=12, fontweight='bold')
ax2.set_xlabel('d (Diferença)')
ax2.set_ylabel('Densidade de Probabilidade')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()