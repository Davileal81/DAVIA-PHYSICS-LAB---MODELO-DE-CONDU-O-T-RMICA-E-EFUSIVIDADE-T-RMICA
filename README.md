# DAVIA-PHYSICS-LAB---MODELO-DE-CONDU-O-T-RMICA-E-EFUSIVIDADE-T-RMICA
Simulador do experimento de condução térmica e difusividade térmica
Aqui está novamente o conteúdo completo para você salvar no seu arquivo README.md:
# DAVIA Physics Lab — Condução Térmica na Pele Humana

Simulador computacional unidimensional e transiente de condução térmica na pele humana e estimativa de efusividade térmica.

## Visão Geral

Este projeto implementa uma ferramenta de simulação científica baseada no referencial teórico de **Stefano Oss (*Eur. J. Phys.* 43, 035101, 2022)**. O software calcula e visualiza a resposta térmica térmica da derme e epiderme ao entrar em contato com diferentes materiais, permitindo analisar gradientes de temperatura, fluxo de calor e a resposta dinâmica dos termorreceptores.

## Funcionalidades Principais

* **Modelagem Térmica Transiente:** Aplicação de equações diferenciais para estimar a temperatura de contato instantânea e sua evolução temporal.
* **Comparação Multimaterial:** Análise simultânea de múltiplos materiais (Madeira Dura, Acrílico, Mármore e Ferro).
* **Interface Gráfica Intuitiva:** Painel construído em `tkinter` para entrada de parâmetros físicos e exibição de resultados numéricos calculados.
* **Visualização de Dados:** Geração de gráficos dinâmicos via `matplotlib` cobrindo perfis espaciais na derme, temperatura de contato, fluxo de calor e taxa de variação de temperatura ($dT_S/dt$).
* **Exportação de Relatórios:** Geração automática de relatórios técnicos em formato PDF (contendo ficha de parâmetros, equações aplicadas e painel de gráficos).
* **Modo Laboratório IR (Eq. 18):** Ferramenta auxiliar para cálculo inverso da efusividade térmica a partir de medições experimentais por termografia infravermelha.

## Requisitos

O projeto utiliza bibliotecas padrão de ciência de dados e interface gráfica em Python. Certifique-se de ter instalado:

* `numpy`
* `matplotlib`

## Estrutura do Código

O script principal está dividido em três módulos principais:

1. **Núcle Científico (`ModeloTermicoPele`):** Contém as equações físicas implementadas, incluindo perfis de temperatura inicial, função erro complementar (`erfc`) e cálculo de efusividade.
2. **Gerador de Relatórios (`GeradorRelatorioPDF`):** Responsável pela montagem e exportação do documento PDF de duas páginas utilizando `matplotlib.backends.backend_pdf`.
3. **Interface Gráfica (`AppSimuladorPele`):** Constrói a janela principal, gerencia os inputs do usuário, executa as simulações em tempo real e atualiza os gráficos na tela.

## Como Executar

1. Certifique-se de que os pré-requisitos estão instalados.
2. Execute o arquivo principal do projeto:

```bash
python main.py

Referência Bibliográfica
 * Oss, S. (2022). Thermal conduction in human skin and the perception of warmth/cold. European Journal of Physics, 43(3), 035101.
