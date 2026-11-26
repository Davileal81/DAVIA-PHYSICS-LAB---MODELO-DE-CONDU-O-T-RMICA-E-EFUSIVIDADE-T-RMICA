import math
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages


# ==============================================================================
# 1. NÚCLEO CIENTÍFICO (Reutilizável no DAVIA Physics Lab Desktop e Mobile)
# ==============================================================================
class ModeloTermicoPele:
    """
    Modelo unidimensional transiente de condução térmica na pele humana
    baseado em Stefano Oss (Eur. J. Phys. 43, 035101, 2022).
    """

    MATERIAIS = {
        "Madeira Dura": {"e": 360.0, "alpha": 1.2e-7, "cor": "#1f77b4"},
        "Acrílico (Plexiglas)": {"e": 560.0, "alpha": 1.1e-7, "cor": "#2ca02c"},
        "Mármore": {"e": 2100.0, "alpha": 1.2e-6, "cor": "#d62728"},
        "Ferro": {"e": 17000.0, "alpha": 2.3e-5, "cor": "#111111"},
    }

    def __init__(self, L=1.1e-3, e_s=1300.0, alpha_s=3.3e-8, k_s=0.24):
        self.L = L              # Espessura da derme + epiderme (m)
        self.e_s = e_s          # Efusividade da pele [W s^(1/2) / (m^2 K)]
        self.alpha_s = alpha_s  # Difusividade térmica da pele (m^2/s)
        self.k_s = k_s          # Condutividade térmica da pele [W / (m K)]
        self.tau_s = (self.L ** 2) / self.alpha_s  # Tempo característico (~36.7 s)

    @staticmethod
    def _erfc_vec(z):
        return np.vectorize(math.erfc)(z)

    def temperatura_inicial_pele(self, x, T_B, T_S0):
        """Equação (1): Perfil linear inicial dentro da pele (0 <= x <= L)."""
        return T_B + (T_S0 - T_B) * (x / self.L)

    def temperatura_contato_instantanea(self, T_S0, T_M0, e_m):
        """Equação (13): Média ponderada pelas efusividades (t -> 0)."""
        return (self.e_s * T_S0 + e_m * T_M0) / (self.e_s + e_m)

    def temperatura_contato_temporal(self, t, T_B, T_S0, T_M0, e_m):
        """Equação (11): Evolução da temperatura de contato T_C(t) para t << tau_S."""
        t_safe = np.maximum(t, 1e-6)
        termo_estatico = self.temperatura_contato_instantanea(T_S0, T_M0, e_m)
        termo_gradiente = (
            2.0
            * (self.e_s / (self.e_s + e_m))
            * ((T_B - T_S0) / self.L)
            * np.sqrt(self.alpha_s * t_safe / np.pi)
        )
        return termo_estatico + termo_gradiente

    def fluxo_contato(self, t, T_B, T_S0, T_M0, e_m):
        """Equação (12): Fluxo de calor na interface q_C(t) em W/m^2."""
        t_safe = np.maximum(t, 1e-3)
        fator_efusividade = (self.e_s * e_m) / (self.e_s + e_m)
        termo1 = fator_efusividade * (T_M0 - T_S0) / np.sqrt(np.pi * t_safe)
        termo2 = (e_m / (self.e_s + e_m)) * self.k_s * ((T_B - T_S0) / self.L)
        return np.abs(termo1 - termo2)

    def perfil_temperatura_pele(self, x, t, T_B, T_S0, T_M0, e_m):
        """Equação (7): Campo de temperatura T_S(x, t) no interior da pele."""
        t_safe = max(t, 1e-4)
        dist = self.L - x
        denom_diff = 2.0 * np.sqrt(self.alpha_s * t_safe)
        arg_erfc = dist / denom_diff

        T_ini = self.temperatura_inicial_pele(x, T_B, T_S0)
        termo_erfc = self._erfc_vec(arg_erfc) * (
            e_m * (T_M0 - T_S0) + self.e_s * (T_S0 - T_B) * (dist / self.L)
        )
        termo_exp = (
            2.0
            * self.k_s
            * np.sqrt(t_safe / np.pi)
            * ((T_B - T_S0) / self.L)
            * np.exp(-(dist ** 2) / (4.0 * self.alpha_s * t_safe))
        )
        return T_ini + (1.0 / (self.e_s + e_m)) * (termo_erfc + termo_exp)

    def taxa_variacao_temperatura(self, x, t, T_B, T_S0, T_M0, e_m):
        """Equação (15): Derivada temporal dT_S/dt (°C/s) — estímulo dos termorreceptores."""
        t_safe = max(t, 1e-4)
        dist = self.L - x
        prefator = (1.0 / np.sqrt(np.pi * self.alpha_s * t_safe)) * (
            e_m / (self.e_s + e_m)
        )
        colchete = (T_M0 - T_S0) * (dist / (2.0 * t_safe)) + (
            (self.e_s / e_m) * ((T_B - T_S0) / self.L) * self.alpha_s
        )
        gaussiana = np.exp(-(dist ** 2) / (4.0 * self.alpha_s * t_safe))
        return prefator * colchete * gaussiana

    def profundidade_pico_receptor(self, t):
        """Equação (16): Posição x_M onde |dT_S/dt| é máximo."""
        return max(0.0, self.L - math.sqrt(2.0 * self.alpha_s * t))

    def calcular_efusividade_experimental(self, T_S0, T_M0, T_C, e_ref_pele=1400.0):
        """Equação (18): Estima e_M a partir das temperaturas medidas na câmera IR."""
        if abs(T_C - T_M0) < 1e-6:
            raise ValueError("T_C não pode ser igual a T_M0 (divisão por zero).")
        razao_es_em = (T_C - T_M0) / (T_S0 - T_C)
        if razao_es_em <= 0:
            raise ValueError("Temperaturas fisicamente inconsistentes (T_C deve estar entre T_M0 e T_S0).")
        e_m_exp = e_ref_pele / razao_es_em
        return razao_es_em, e_m_exp


# ==============================================================================
# 2. GERADOR DE RELATÓRIOS EM PDF (Parâmetros + Resultados + Gráficos)
# ==============================================================================
class GeradorRelatorioPDF:
    @staticmethod
    def salvar_relatorio(caminho_arquivo, dados_simulacao, figura_graficos):
        """
        Gera um PDF de 2 páginas:
          - Página 1: Ficha técnica da simulação (parâmetros, equações e resultados)
          - Página 2: Painel com os 4 gráficos da simulação
        """
        with PdfPages(caminho_arquivo) as pdf:
            # --- PÁGINA 1: RELATÓRIO ESCRITO ---
            fig_texto = plt.figure(figsize=(8.27, 11.69))  # Padrão A4 em polegadas
            fig_texto.patch.set_facecolor("white")
            ax = fig_texto.add_axes([0.08, 0.06, 0.84, 0.88])
            ax.axis("off")

            # Cabeçalho
            ax.text(
                0.0, 0.97, "DAVIA PHYSICS LAB",
                fontsize=18, fontweight="bold", color="#101B2D"
            )
            ax.text(
                0.0, 0.94,
                "Relatório de Simulação — Condução Térmica na Pele Humana e Efusividade",
                fontsize=11, color="#1769AA", fontweight="bold"
            )
            ax.text(
                0.0, 0.915,
                f"Referência Teórica: Stefano Oss, Eur. J. Phys. 43 (2022) 035101  |  Gerado em: {dados_simulacao['data_hora']}",
                fontsize=8.5, color="#555555"
            )
            ax.plot([0, 1], [0.90, 0.90], color="#1769AA", linewidth=1.5, transform=ax.transAxes)

            # Seção 1: Parâmetros de Entrada
            ax.text(0.0, 0.865, "1. PARÂMETROS DA SIMULAÇÃO", fontsize=11, fontweight="bold", color="#101B2D")
            bloco_parametros = (
                f"• Material Externo Analisado:        {dados_simulacao['material']}\n"
                f"• Efusividade do Material (e_M):     {dados_simulacao['e_m']:.1f} W·s^(1/2)/(m²·K)\n"
                f"• Efusividade da Pele (e_S):         {dados_simulacao['e_s']:.1f} W·s^(1/2)/(m²·K)\n"
                f"• Difusividade da Pele (α_S):        {dados_simulacao['alpha_s']:.2e} m²/s\n"
                f"• Espessura da Camada Dérmica (L):   {dados_simulacao['L_mm']:.2f} mm\n"
                f"• Temperatura Interna Sangue (T_B):  {dados_simulacao['T_B']:.2f} °C\n"
                f"• Temperatura Inicial Pele (T_S0):   {dados_simulacao['T_S0']:.2f} °C\n"
                f"• Temperatura Inicial Objeto (T_M0): {dados_simulacao['T_M0']:.2f} °C\n"
                f"• Instante de Análise (t):           {dados_simulacao['t_alvo']:.2f} s\n"
                f"• Modo Comparativo (4 materiais):    {'Sim' if dados_simulacao['comparar_todos'] else 'Não'}"
            )
            ax.text(
                0.02, 0.84, bloco_parametros,
                fontsize=9.5, family="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F1F5F9", edgecolor="#CBD5E1")
            )

            # Seção 2: Resultados Obtidos
            ax.text(0.0, 0.56, "2. RESULTADOS CALCULADOS", fontsize=11, fontweight="bold", color="#101B2D")
            bloco_resultados = (
                f"• Temperatura de Contato Instantânea (T_C, Eq. 13):  {dados_simulacao['Tc_inst']:.2f} °C\n"
                f"• Temperatura de Contato em t = {dados_simulacao['t_alvo']:.1f} s (Eq. 11):     {dados_simulacao['Tc_t']:.2f} °C\n"
                f"• Variação Térmica na Superfície da Pele (ΔT):       {dados_simulacao['queda_temp']:+.2f} °C\n"
                f"• Fluxo de Calor na Interface (q_C, Eq. 12):         {dados_simulacao['fluxo_t']:.1f} W/m²\n"
                f"• Profundidade de Pico dos Receptores (Eq. 16):      {dados_simulacao['profundidade_mm']:.2f} mm (a partir da epiderme)\n"
                f"• Classificação da Sensação Térmica:                 {dados_simulacao['sensacao']}"
            )
            ax.text(
                0.02, 0.535, bloco_resultados,
                fontsize=9.5, family="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F4FD", edgecolor="#90CAF9")
            )

            # Seção 3: Fundamentação Matemática Aplicada
            ax.text(0.0, 0.33, "3. EQUAÇÕES DO MODELO TRANSIENTE (t << τ_S)", fontsize=11, fontweight="bold", color="#101B2D")
            bloco_equacoes = (
                "• Efusividade Térmica (Eq. 6):         e = √(k · ρ · c) = k / √α\n"
                "• Temperatura de Contato (Eq. 13):     T_C ≈ (e_S·T_S0 + e_M·T_M0) / (e_S + e_M)\n"
                "• Evolução Temporal de T_C (Eq. 11):   T_C(t) ≈ T_C(0) + 2·[e_S/(e_S+e_M)]·[(T_B - T_S0)/L]·√(α_S·t/π)\n"
                "• Fluxo de Calor na Interface (Eq. 14): q_C(t) ≈ [e_S·e_M / (e_S + e_M)] · (T_M0 - T_S0) / √(π·t)\n"
                "• Posição de Estímulo Máximo (Eq. 16): x_M(t) ≈ L - √(2 · α_S · t)"
            )
            ax.text(
                0.02, 0.305, bloco_equacoes,
                fontsize=9, family="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#E2E8F0")
            )

            # Rodapé Página 1
            ax.text(
                0.0, 0.02,
                "Documento gerado automaticamente pelo DAVIA Physics Lab — Página 1 de 2 (Dados e Parâmetros)",
                fontsize=8, color="#777777"
            )
            pdf.savefig(fig_texto, dpi=300)
            plt.close(fig_texto)

            # --- PÁGINA 2: GRÁFICOS DA SIMULAÇÃO ---
            figura_graficos.suptitle(
                f"DAVIA Physics Lab — Gráficos da Simulação ({dados_simulacao['material']}, t = {dados_simulacao['t_alvo']:.1f} s)",
                fontsize=11, fontweight="bold", y=0.99
            )
            pdf.savefig(figura_graficos, dpi=300, bbox_inches="tight")
            figura_graficos.suptitle("")  # Limpa título extra na tela interativa


# ==============================================================================
# 3. INTERFACE GRÁFICA PRINCIPAL
# ==============================================================================
class AppSimuladorPele:
    def __init__(self, root):
        self.root = root
        self.root.title("DAVIA Physics Lab — Condução Térmica na Pele (Oss, 2022)")
        self.root.geometry("1150x700")
        self.root.configure(bg="#101B2D")

        self.modelo = ModeloTermicoPele()
        self.ultima_simulacao = None

        self._configurar_estilos()
        self._construir_interface()
        self.executar_simulacao()

    def _configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101B2D")
        style.configure("Card.TFrame", background="#17263D", relief="flat")
        style.configure("TLabel", background="#17263D", foreground="#E8EEF5", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#101B2D", foreground="#4FC3F7", font=("Segoe UI", 14, "bold"))
        style.configure("Sub.TLabel", background="#17263D", foreground="#81D4FA", font=("Segoe UI", 11, "bold"))
        style.configure("Eq.TLabel", background="#0D1522", foreground="#A5D6A7", font=("Consolas", 9.5))
        style.configure("TButton", font=("Segoe UI", 9.5, "bold"), padding=6)

    def _construir_interface(self):
        # Cabeçalho
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=15, pady=(10, 5))
        ttk.Label(
            header,
            text="DAVIA PHYSICS LAB | Modelo de Condução Térmica na Pele e Efusividade (Oss, 2022)",
            style="Title.TLabel",
        ).pack(side="left")

        # Container Principal
        main_pane = ttk.Frame(self.root)
        main_pane.pack(fill="both", expand=True, padx=15, pady=10)

        # Coluna Esquerda: Controles, Fórmulas e Relatórios
        left_col = ttk.Frame(main_pane, style="Card.TFrame", padding=12)
        left_col.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(left_col, text="1. Parâmetros do Experimento", style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        self.var_material = tk.StringVar(value="Ferro")
        self.var_TB = tk.StringVar(value="36.4")
        self.var_TS0 = tk.StringVar(value="33.3")
        self.var_TM0 = tk.StringVar(value="23.8")
        self.var_tempo = tk.StringVar(value="1.0")
        self.var_modo_comparar = tk.BooleanVar(value=True)

        self._criar_campo_combo(left_col, "Material Externo:", self.var_material, list(self.modelo.MATERIAIS.keys()))
        self._criar_campo_entry(left_col, "Temp. Interna Sangue T_B (°C):", self.var_TB)
        self._criar_campo_entry(left_col, "Temp. Inicial Pele T_S0 (°C):", self.var_TS0)
        self._criar_campo_entry(left_col, "Temp. Inicial Objeto T_M0 (°C):", self.var_TM0)
        self._criar_campo_entry(left_col, "Instante de Análise t (s):", self.var_tempo)

        ttk.Checkbutton(
            left_col,
            text="Comparar os 4 materiais simultaneamente",
            variable=self.var_modo_comparar,
            command=self.executar_simulacao,
        ).pack(anchor="w", pady=6)

        # Botões de Ação
        btn_frame1 = ttk.Frame(left_col, style="Card.TFrame")
        btn_frame1.pack(fill="x", pady=(4, 4))
        ttk.Button(btn_frame1, text="Simular Agora", command=self.executar_simulacao).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(btn_frame1, text="Modo Lab IR (Eq. 18)", command=self.abrir_janela_laboratorio).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        # Botão de Exportação de Relatório
        ttk.Button(
            left_col,
            text="Salvar Relatório Completo (PDF)",
            command=self.exportar_relatorio_pdf,
        ).pack(fill="x", pady=(2, 8))

        # Bloco de Fórmulas
        ttk.Label(left_col, text="2. Equações do Modelo (t << 35 s)", style="Sub.TLabel").pack(anchor="w", pady=(6, 4))
        formulas_texto = (
            "Efusividade:   e = √(k·ρ·c) [W·s¹ᐟ²/(m²·K)]\n"
            "Contato (13):  T_C ≈ (e_S·T_S0 + e_M·T_M0)/(e_S + e_M)\n"
            "Fluxo (14):    q_C ≈ [e_S·e_M/(e_S+e_M)]·ΔT/√(πt)\n"
            "Receptor (16): x_M ≈ L - √(2·α_S·t)"
        )
        ttk.Label(left_col, text=formulas_texto, style="Eq.TLabel", padding=8).pack(fill="x", pady=(0, 8))

        # Bloco de Resultados Numéricos
        ttk.Label(left_col, text="3. Resultados Calculados", style="Sub.TLabel").pack(anchor="w", pady=(4, 4))
        self.lbl_resultados = ttk.Label(left_col, text="", font=("Consolas", 10), foreground="#FFF59D")
        self.lbl_resultados.pack(anchor="w", fill="x")

        # Coluna Direita: Gráficos
        right_col = ttk.Frame(main_pane, style="Card.TFrame", padding=10)
        right_col.pack(side="right", fill="both", expand=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(7.6, 5.8), dpi=100)
        self.fig.patch.set_facecolor("#F8FAFC")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_col)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _criar_campo_entry(self, parent, texto, variavel):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", pady=3)
        ttk.Label(frame, text=texto, width=27).pack(side="left")
        ttk.Entry(frame, textvariable=variavel, width=10).pack(side="right")

    def _criar_campo_combo(self, parent, texto, variavel, opcoes):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", pady=3)
        ttk.Label(frame, text=texto, width=20).pack(side="left")
        combo = ttk.Combobox(frame, textvariable=variavel, values=opcoes, state="readonly", width=17)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", lambda e: self.executar_simulacao())

    def executar_simulacao(self):
        try:
            T_B = float(self.var_TB.get().replace(",", "."))
            T_S0 = float(self.var_TS0.get().replace(",", "."))
            T_M0 = float(self.var_TM0.get().replace(",", "."))
            t_alvo = float(self.var_tempo.get().replace(",", "."))
            if t_alvo <= 0:
                raise ValueError("O tempo deve ser maior que zero.")
        except ValueError as err:
            messagebox.showerror("Entrada Inválida", f"Verifique os valores numéricos:\n{err}")
            return

        mat_selecionado = self.var_material.get()
        dados_mat = self.modelo.MATERIAIS[mat_selecionado]
        e_m = dados_mat["e"]

        Tc_inst = self.modelo.temperatura_contato_instantanea(T_S0, T_M0, e_m)
        Tc_t = self.modelo.temperatura_contato_temporal(t_alvo, T_B, T_S0, T_M0, e_m)
        fluxo_t = self.modelo.fluxo_contato(t_alvo, T_B, T_S0, T_M0, e_m)
        x_pico = self.modelo.profundidade_pico_receptor(t_alvo)
        profundidade_epiderme_mm = (self.modelo.L - x_pico) * 1000.0
        queda_temp = Tc_t - T_S0

        if queda_temp < -6.0:
            sensacao = "FRIO INTENSO (Choque térmico)"
        elif queda_temp < -2.5:
            sensacao = "FRIO MODERADO"
        elif queda_temp <= 1.0:
            sensacao = "TÉRMICO NEUTRO / AGRADÁVEL"
        else:
            sensacao = "QUENTE"

        # Armazena estado completo para geração de relatório
        self.ultima_simulacao = {
            "data_hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "material": mat_selecionado,
            "e_m": e_m,
            "e_s": self.modelo.e_s,
            "alpha_s": self.modelo.alpha_s,
            "L_mm": self.modelo.L * 1000.0,
            "T_B": T_B,
            "T_S0": T_S0,
            "T_M0": T_M0,
            "t_alvo": t_alvo,
            "comparar_todos": self.var_modo_comparar.get(),
            "Tc_inst": Tc_inst,
            "Tc_t": Tc_t,
            "queda_temp": queda_temp,
            "fluxo_t": fluxo_t,
            "profundidade_mm": profundidade_epiderme_mm,
            "sensacao": sensacao,
        }

        resumo = (
            f"Material: {mat_selecionado}\n"
            f"e_pele = {self.modelo.e_s:.0f} | e_mat = {e_m:.0f} W·s½/m²K\n"
            f"T_C instantânea (Eq.13): {Tc_inst:.2f} °C\n"
            f"T_C em t={t_alvo:.1f}s (Eq.11):  {Tc_t:.2f} °C\n"
            f"Fluxo de calor (Eq.12):  {fluxo_t:.0f} W/m²\n"
            f"Pico receptor (Eq.16):   {profundidade_epiderme_mm:.2f} mm da sup.\n"
            f"Percepção: {sensacao}"
        )
        self.lbl_resultados.config(text=resumo)
        self._atualizar_graficos(T_B, T_S0, T_M0, t_alvo, mat_selecionado)

    def _atualizar_graficos(self, T_B, T_S0, T_M0, t_alvo, mat_selecionado):
        for ax in self.axes.flat:
            ax.clear()
            ax.grid(True, linestyle="--", alpha=0.4)

        x_pele = np.linspace(0, self.modelo.L, 120)
        t_vetor = np.linspace(0.05, 5.0, 120)

        materiais_plot = (
            self.modelo.MATERIAIS.items()
            if self.var_modo_comparar.get()
            else [(mat_selecionado, self.modelo.MATERIAIS[mat_selecionado])]
        )

        # Gráfico 1: Perfil Espacial na Pele T_S(x)
        ax1 = self.axes[0, 0]
        for nome, props in materiais_plot:
            T_perfil = self.modelo.perfil_temperatura_pele(x_pele, t_alvo, T_B, T_S0, T_M0, props["e"])
            ax1.plot(x_pele * 1000, T_perfil, label=nome, color=props["cor"], linewidth=1.8)
        ax1.axvline(self.modelo.L * 1000, color="gray", linestyle=":", label="Interface x=L")
        ax1.set_title(f"Perfil na Derme após t = {t_alvo:.1f} s (Fig. 3)", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Posição x (mm)", fontsize=8)
        ax1.set_ylabel("Temperatura (°C)", fontsize=8)
        ax1.legend(fontsize=7)

        # Gráfico 2: Temperatura de Contato T_C(t)
        ax2 = self.axes[0, 1]
        for nome, props in materiais_plot:
            Tc_curva = self.modelo.temperatura_contato_temporal(t_vetor, T_B, T_S0, T_M0, props["e"])
            ax2.plot(t_vetor, Tc_curva, label=nome, color=props["cor"], linewidth=1.8)
        ax2.set_title("Temperatura de Contato T_C(t) (Fig. 4)", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Tempo t (s)", fontsize=8)
        ax2.set_ylabel("T_C (°C)", fontsize=8)

        # Gráfico 3: Fluxo de Calor na Interface q_C(t)
        ax3 = self.axes[1, 0]
        for nome, props in materiais_plot:
            q_curva = self.modelo.fluxo_contato(t_vetor, T_B, T_S0, T_M0, props["e"])
            ax3.plot(t_vetor, q_curva / 1000.0, label=nome, color=props["cor"], linewidth=1.8)
        ax3.set_title("Fluxo de Calor na Interface (Fig. 5)", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Tempo t (s)", fontsize=8)
        ax3.set_ylabel("Fluxo (kW/m²)", fontsize=8)

        # Gráfico 4: Resposta Dinâmica dos Receptores dT_S/dt
        ax4 = self.axes[1, 1]
        e_sel = self.modelo.MATERIAIS[mat_selecionado]["e"]
        for t_inst, estilo in [(0.5, "-."), (1.0, "-"), (2.0, "--"), (4.0, ":")]:
            dT_dt = self.modelo.taxa_variacao_temperatura(x_pele, t_inst, T_B, T_S0, T_M0, e_sel)
            ax4.plot(x_pele * 1000, dT_dt, linestyle=estilo, label=f"t = {t_inst} s")
        ax4.set_title(f"Estímulo Dinâmico dT/dt — {mat_selecionado} (Fig. 7)", fontsize=9, fontweight="bold")
        ax4.set_xlabel("Posição x (mm)", fontsize=8)
        ax4.set_ylabel("dT/dt (°C/s)", fontsize=8)
        ax4.legend(fontsize=7)

        self.fig.tight_layout()
        self.canvas.draw()

    def exportar_relatorio_pdf(self):
        """Abre diálogo para salvar o relatório PDF com parâmetros e gráficos."""
        if not self.ultima_simulacao:
            self.executar_simulacao()

        nome_padrao = (
            f"Relatorio_DAVIA_{self.ultima_simulacao['material'].replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        caminho = filedialog.asksaveasfilename(
            title="Salvar Relatório da Simulação",
            defaultextension=".pdf",
            initialfile=nome_padrao,
            filetypes=[("Arquivo PDF", "*.pdf")],
        )
        if not caminho:
            return

        try:
            GeradorRelatorioPDF.salvar_relatorio(caminho, self.ultima_simulacao, self.fig)
            self.canvas.draw()
            messagebox.showinfo(
                "Relatório Salvo",
                f"Relatório gerado com sucesso!\n\nArquivo:\n{caminho}",
            )
        except Exception as e:
            messagebox.showerror("Erro ao Salvar PDF", f"Não foi possível gerar o relatório:\n{e}")

    def abrir_janela_laboratorio(self):
        """Janela experimental para cálculo inverso da efusividade via Eq. (18)."""
        win = tk.Toplevel(self.root)
        win.title("Laboratório Experimental — Termografia IR (Eq. 18)")
        win.geometry("420x310")
        win.configure(bg="#17263D")

        ttk.Label(win, text="Cálculo Experimental de Efusividade (Eq. 18)", style="Sub.TLabel").pack(pady=10)

        v_ts0 = tk.StringVar(value="34.8")
        v_tm0 = tk.StringVar(value="23.8")
        v_tc = tk.StringVar(value="28.4")
        v_es = tk.StringVar(value="1400")

        for txt, var in [
            ("Temp. inicial da pele T_S0 (°C):", v_ts0),
            ("Temp. inicial do bloco T_M0 (°C):", v_tm0),
            ("Temp. medida após toque T_C (°C):", v_tc),
            ("Efusividade ref. da pele e_S:", v_es),
        ]:
            f = ttk.Frame(win, style="Card.TFrame")
            f.pack(fill="x", padx=20, pady=4)
            ttk.Label(f, text=txt).pack(side="left")
            ttk.Entry(f, textvariable=var, width=10).pack(side="right")

        lbl_res_exp = ttk.Label(win, text="", foreground="#A5D6A7", font=("Consolas", 10))
        lbl_res_exp.pack(pady=12)

        def calcular():
            try:
                razao, em_calc = self.modelo.calcular_efusividade_experimental(
                    float(v_ts0.get().replace(",", ".")),
                    float(v_tm0.get().replace(",", ".")),
                    float(v_tc.get().replace(",", ".")),
                    float(v_es.get().replace(",", ".")),
                )
                lbl_res_exp.config(
                    text=f"Razão (e_S / e_M) = {razao:.3f}\nEfusividade e_M = {em_calc:.1f} W·s½/(m²·K)"
                )
            except ValueError as e:
                messagebox.showerror("Erro Experimental", str(e), parent=win)

        ttk.Button(win, text="Calcular Efusividade do Material", command=calcular).pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = AppSimuladorPele(root)
    root.mainloop()