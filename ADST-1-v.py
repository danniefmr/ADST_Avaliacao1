import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
duracao_tentativa = 2.0          # ms por tentativa
max_tentativas = 4
tamanho_trama_bits = 1024 * 8    # 8192 bits

# Variação de p (probabilidade de erro por bit - BER) em escala logarítmica
p_bit = np.logspace(-5, -2, 500)

# Probabilidade de insucesso da trama (p_nst) em função da BER
# Uma trama falha se pelo menos 1 bit falhar
p_nst = 1 - (1 - p_bit)**tamanho_trama_bits

# 1. E[4] - Número médio de tentativas (todas as transmissões)
E_4 = (1 - p_nst**max_tentativas) / (1 - p_nst)

# 2. Probabilidade de falhar todas as 4 tentativas
p_falha_total = p_nst**max_tentativas

# 3. E[4|S] - Número médio de tentativas condicionado ao sucesso
E_4_dado_S = (E_4 - max_tentativas * p_falha_total) / (1 - p_falha_total)

# 4. Cálculo dos tempos médios (em ms)
T_F = duracao_tentativa * E_4
T_S = duracao_tentativa * E_4_dado_S

# --- Plotagem do Gráfico ---
plt.figure(figsize=(9, 5))
plt.plot(p_bit, T_F, label='$T_F$ (Todas as transmissões)', color='royalblue', linewidth=2)
plt.plot(p_bit, T_S, label='$T_S$ (Apenas bem-sucedidas)', color='forestgreen', linewidth=2)

# Configurações do gráfico
plt.xscale('log')
plt.xlabel('Probabilidade de erro por bit $p$', fontsize=12)
plt.ylabel('Tempo médio de transmissão (ms)', fontsize=12)
plt.title('Tempos Médios $T_F$ e $T_S$ em função de $p$', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()