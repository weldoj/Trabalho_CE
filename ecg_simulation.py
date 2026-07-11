# -*- coding: utf-8 -*-
"""
Simulação e Modelagem do Estágio Amplificador de Eletrodo Capacitivo para ECG/EEG
Baseado no artigo de Chi & Cauwenberghs (2010).

Autor: Engenheiro Eletrônico Sênior (Instrumentação Biomédica)
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARÂMETROS DO CIRCUITO (Valores Reais)
# =============================================================================
R4 = 1e6        # 1 MΩ (Resistência de polarização/filtro de entrada)
C3 = 1.5e-6    # 1.5 µF (Capacitância de acoplamento de entrada)
R5 = 100e3      # 100 kΩ (Resistência de realimentação do estágio de ganho)
R6 = 1e3        # 1 kΩ (Resistência de ganho do estágio diferencial)
C4 = 1.5e-9     # 1.5 nF (Capacitância de compensação em paralelo com R5)
R7 = 10e3       # 10 kΩ (Resistência do filtro passa-baixas de saída)
C5 = 0.15e-6    # 0.15 µF (Capacitância do filtro passa-baixas de saída)

print("=== Parametros do Circuito Carregados ===")
print(f"R4 = {R4/1e6:.1f} MOhm, C3 = {C3*1e6:.2f} uF")
print(f"R5 = {R5/1e3:.1f} kOhm, R6 = {R6/1e3:.1f} kOhm, C4 = {C4*1e9:.1f} nF")
print(f"R7 = {R7/1e3:.1f} kOhm, C5 = {C5*1e6:.2f} uF\n")

# =============================================================================
# 2. DEFINIÇÃO DAS FUNÇÕES DE TRANSFERÊNCIA (Domínio de Laplace)
# =============================================================================
# Estágio 1: Filtro Passa-Altas de Entrada
# H1(s) = (s * R4 * C3) / (1 + s * R4 * C3)
num1 = np.array([R4 * C3, 0.0])
den1 = np.array([R4 * C3, 1.0])

# Estágio 2: Amplificador Diferencial com Realimentação RC
# H2(s) = (1 + s * R5 * C4 + R5 / R6) / (1 + s * R5 * C4)
num2 = np.array([R5 * C4, 1.0 + (R5 / R6)])
den2 = np.array([R5 * C4, 1.0])

# Estágio 3: Filtro Passa-Baixas de Saída
# H3(s) = 1 / (1 + s * R7 * C5)
num3 = np.array([1.0])
den3 = np.array([R7 * C5, 1.0])

# Combinação dos estágios usando convolução de polinômios para obter H(s) total
num_total = np.convolve(num1, np.convolve(num2, num3))
den_total = np.convolve(den1, np.convolve(den2, den3))

# Criação do sistema LTI no SciPy
sys_total = signal.lti(num_total, den_total)

# =============================================================================
# A. DIAGRAMA DE BODE (Magnitude e Fase)
# =============================================================================
# Frequências de simulação: 0.01 Hz a 1000 Hz
f = np.logspace(-2, 3, 100000)  # Vetor de frequência em Hz (alta resolução)
w = 2 * np.pi * f               # Frequência angular em rad/s

# Cálculo da resposta em frequência
w_out, mag, phase = signal.bode(sys_total, w=w)

# Cálculo do ganho máximo e frequências de corte simuladas (-3 dB)
idx_max = np.argmax(mag)
max_gain_db = mag[idx_max]
cutoff_threshold = max_gain_db - 3.0

# Frequência de corte inferior (-3 dB) usando interpolação na subida do ganho
mag_low = mag[:idx_max]
f_low = f[:idx_max]
f_lower_sim = np.interp(cutoff_threshold, mag_low, f_low)

# Frequência de corte superior (-3 dB) usando interpolação na descida do ganho
mag_high = mag[idx_max:]
f_high = f[idx_max:]
# Inverte para que xp (magnitude) seja estritamente crescente
f_upper_sim = np.interp(cutoff_threshold, mag_high[::-1], f_high[::-1])

# Plotagem do Diagrama de Bode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot da Magnitude
ax1.semilogx(f, mag, 'b-', linewidth=2, label='Resposta Simulada')
ax1.axhline(max_gain_db, color='orange', linestyle='--', linewidth=1.5,
            label=f'Ganho Máximo: {max_gain_db:.2f} dB ({10**(max_gain_db/20):.1f} V/V)')
ax1.axhline(cutoff_threshold, color='red', linestyle='--', linewidth=1.5,
            label=f'Limite -3 dB ({cutoff_threshold:.2f} dB)')

# Marcação dos pontos de corte simulados
ax1.plot(f_lower_sim, cutoff_threshold, 'ro', markersize=8)
ax1.plot(f_upper_sim, cutoff_threshold, 'ro', markersize=8)
ax1.text(f_lower_sim * 1.3, cutoff_threshold - 4, f'{f_lower_sim:.3f} Hz', color='red', weight='bold')
ax1.text(f_upper_sim * 0.3, cutoff_threshold - 4, f'{f_upper_sim:.1f} Hz', color='red', weight='bold')

# Linhas de referência para valores teóricos
ax1.axvline(0.106, color='green', linestyle=':', linewidth=1.5, label='Corte Teórico Inferior (0.106 Hz)')
ax1.axvline(106.0, color='purple', linestyle=':', linewidth=1.5, label='Corte Teórico Superior (105 Hz)')

ax1.set_ylabel('Magnitude (dB)', fontsize=11)
ax1.set_title('Diagrama de Bode - Estágio Amplificador Diferencial para ECG/EEG', fontsize=13, weight='bold')
ax1.grid(True, which="both", ls="-", color='0.85')
ax1.legend(loc='lower center', fontsize=9)

# Subplot da Fase
ax2.semilogx(f, phase, 'g-', linewidth=2, label='Fase Simulada')
ax2.set_xlabel('Frequência (Hz)', fontsize=11)
ax2.set_ylabel('Fase (graus)', fontsize=11)
ax2.grid(True, which="both", ls="-", color='0.85')
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('bode_total.png', dpi=300)
plt.close()

# =============================================================================
# B. RESPOSTA AO DEGRAU (Step Response)
# =============================================================================
# Simulação para os primeiros 5 segundos
t_step = np.linspace(0, 5, 5000)
t_step_out, y_step = signal.step(sys_total, T=t_step)

plt.figure(figsize=(8, 5))
plt.plot(t_step_out, y_step, 'b-', linewidth=2, label='Resposta ao Degrau')
plt.title('Resposta ao Degrau Unitário (Dinâmica de 5s)', fontsize=12, weight='bold')
plt.xlabel('Tempo (s)', fontsize=11)
plt.ylabel('Amplitude de Saída (V)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('step_response.png', dpi=300)
plt.close()

# =============================================================================
# C. SIMULAÇÃO COM SINAL DE ENTRADA REALÍSTICO (ECG + DC OFFSET + RUIDO 60 HZ)
# =============================================================================
# Vetor de tempo de 10 segundos a 1000 Hz de amostragem
fs = 1000
t_ecg = np.linspace(0, 10, 10 * fs, endpoint=False)

# ECG Sintético: 5 pulsos gaussianos em 1s, 3s, 5s, 7s e 9s
# Amplitude de 0.1 V (100 mV), desvio padrão de 0.01 s
ecg_clean = np.zeros_like(t_ecg)
pulse_times = [1.0, 3.0, 5.0, 7.0, 9.0]
for t0 in pulse_times:
    ecg_clean += 0.1 * np.exp(-((t_ecg - t0) ** 2) / (2 * 0.01 ** 2))

# Adição de Offset DC (0.05 V) e interferência de 60 Hz (0.02 V)
offset_dc = 0.05
noise_60hz = 0.02 * np.sin(2 * np.pi * 60 * t_ecg)
u_in = ecg_clean + offset_dc + noise_60hz

# Simulação da saída do circuito usando lsim
t_sim, y_out, _ = signal.lsim(sys_total, U=u_in, T=t_ecg)

# Plotagem do sinal de entrada vs saída
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Entrada
ax1.plot(t_ecg, u_in, 'r-', linewidth=1.5, label='Entrada (ECG + 50mV DC + 20mV @ 60Hz)')
ax1.set_title('Simulação Temporal: Remoção de DC e Ganho de Sinal', fontsize=12, weight='bold')
ax1.set_ylabel('Amplitude de Entrada (V)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# Saída
ax2.plot(t_sim, y_out, 'b-', linewidth=1.5, label='Saída Amplificada/Filtrada')
ax2.set_xlabel('Tempo (s)', fontsize=11)
ax2.set_ylabel('Amplitude de Saída (V)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('ecg_response.png', dpi=300)
plt.close()

# Subplot adicional com Zoom em torno do primeiro pulso para visualizar detalhes
plt.figure(figsize=(10, 5))
# Mostra entre 0.8s e 1.5s
zoom_mask = (t_ecg >= 0.8) & (t_ecg <= 1.5)
plt.plot(t_ecg[zoom_mask], u_in[zoom_mask], 'r-', label='Entrada (Zoom)', alpha=0.7)
plt.plot(t_sim[zoom_mask], y_out[zoom_mask], 'b-', label='Saída (Zoom)', linewidth=2)
plt.title('Detalhe do Primeiro Pulso (Zoom de 0.8s a 1.5s)', fontsize=12, weight='bold')
plt.xlabel('Tempo (s)', fontsize=11)
plt.ylabel('Amplitude (V)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('ecg_response_zoom.png', dpi=300)
plt.close()

# =============================================================================
# D. ANÁLISE DE REJEIÇÃO DE MODO COMUM (CMRR - MODELADO PELA ATENUAÇÃO DE 60 HZ)
# =============================================================================
# Injeção exclusiva do ruído de 60 Hz (0.02 V)
u_60hz = 0.02 * np.sin(2 * np.pi * 60 * t_ecg)

# Simulação da saída para apenas 60 Hz
t_sim_60hz, y_out_60hz, _ = signal.lsim(sys_total, U=u_60hz, T=t_ecg)

# Cálculo do ganho em 60 Hz a partir da resposta em frequência
gain_60hz_db = np.interp(60.0, f, mag)
attenuation_db = max_gain_db - gain_60hz_db

# Gráfico da atenuação de 60 Hz (mostrando alguns ciclos, ex: 0.1s)
plt.figure(figsize=(8, 5))
zoom_60hz = t_ecg <= 0.1
plt.plot(t_ecg[zoom_60hz], u_60hz[zoom_60hz], 'r--', label='Entrada de Modo Comum (20 mV @ 60 Hz)')
plt.plot(t_sim_60hz[zoom_60hz], y_out_60hz[zoom_60hz], 'b-', linewidth=2,
         label=f'Saída resultante (Ganho: {gain_60hz_db:.2f} dB)')
plt.title('Resposta do Circuito ao Ruído de 60 Hz isolado (Janela de 0.1s)', fontsize=12, weight='bold')
plt.xlabel('Tempo (s)', fontsize=11)
plt.ylabel('Amplitude (V)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('cmrr_60Hz.png', dpi=300)
plt.close()

# =============================================================================
# IMPRESSÃO DO RESUMO DOS PARÂMETROS NO CONSOLE
# =============================================================================
print("=================== RESUMO DE DESEMPENHO ===================")
print(f"Ganho maximo na banda passante: {max_gain_db:.2f} dB (ou {10**(max_gain_db/20):.2f} V/V)")
print(f"Frequencia de corte inferior simulada (-3 dB): {f_lower_sim:.3f} Hz")
print(f"Frequencia de corte superior simulada (-3 dB): {f_upper_sim:.2f} Hz")
print(f"Ganho na frequencia da rede eletrica (60 Hz): {gain_60hz_db:.2f} dB")
print(f"Atenuacao de 60 Hz em relacao ao ganho maximo: {attenuation_db:.2f} dB")
print("============================================================")
print("Todas as figuras (PNGs) foram salvas com sucesso no diretorio atual.")
