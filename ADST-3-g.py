import numpy as np

# ---------------------------------------------------------
# 1. Parâmetros e Distribuição Empírica (Tabela 2)
# ---------------------------------------------------------
N_sim = 3 * 10**6  # 3 x 10^6 realizações independentes
N_testes = 10**6

# Probabilidades empíricas p_k = P(K = k) da Tabela 2
p_K = np.array([149800, 250600, 349500, 180200, 69900]) / N_testes
valores_k = np.array([0, 1, 2, 3, 4])

# 2. Função de Distribuição Acumulada (CDF) para a Transformação Inversa
# cdf = [F(0), F(1), F(2), F(3), F(4)]
cdf = np.cumsum(p_K)

# ---------------------------------------------------------
# 3. Método da Transformação Inversa (Vetorizado para Eficiência)
# ---------------------------------------------------------
# Gerar números aleatórios uniformes U ~ U(0, 1) para cada nó nas 3 x 10^6 realizações
np.random.seed(42)  # Semente fixa para reprodutibilidade

U1 = np.random.uniform(0, 1, N_sim)
U2 = np.random.uniform(0, 1, N_sim)
U3 = np.random.uniform(0, 1, N_sim)

# Aplicação da Transformação Inversa:
# Mapeia U no menor índice k onde U <= CDF[k]
K1 = np.searchsorted(cdf, U1)
K2 = np.searchsorted(cdf, U2)
K3 = np.searchsorted(cdf, U3)

# Soma dos três nós para cada realização
S_sim = K1 + K2 + K3

# ---------------------------------------------------------
# 4. Cálculo das Probabilidades Simuladas vs Teóricas (Alínea f)
# ---------------------------------------------------------
# Probabilidades teóricas da alínea f via convolução
p_S_teorico = np.convolve(p_K, np.convolve(p_K, p_K))

# Contagem de ocorrências de cada valor de S (0 a 12)
contagens = np.bincount(S_sim, minlength=13)
p_S_simulado = contagens / N_sim

# ---------------------------------------------------------
# 5. Apresentação e Validação dos Resultados
# ---------------------------------------------------------
print("=" * 70)
print(f"{'s':^3} | {'P(S=s) Teórico (f)':^20} | {'P(S=s) Simulado (g)':^20} | {'Erro Absoluto':^15}")
print("=" * 70)

for s in range(13):
    teorico = p_S_teorico[s]
    simulado = p_S_simulado[s]
    erro = abs(simulado - teorico)
    print(f"{s:3d} | {teorico:20.8f} | {simulado:20.8f} | {erro:15.8f}")

print("=" * 70)
print(f"Soma P(S_simulado): {np.sum(p_S_simulado):.6f}")