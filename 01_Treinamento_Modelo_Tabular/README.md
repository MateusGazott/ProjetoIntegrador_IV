# Integrante 1 — Treinamento do Modelo Tabular
**Responsável:** Mateus

## Visão Geral
Este módulo é responsável pela preparação de dados, análise de atributos financeiros, treinamento comparativo de algoritmos de Machine Learning e exportação do modelo preditivo para o sistema de diagnóstico de insolvência.

---

## 1. Dados e Análise Exploratória
- **Base utilizada:** Polish Companies Bankruptcy Dataset (5 arquivos CSV correspondentes a horizontes de 1 a 5 anos de previsão).
- **Volume total:** 43.405 registros de empresas com 64 indicadores contábeis e a variável alvo `class` (0 = Saudável, 1 = Falida/Insolvente).
- **Desbalanceamento:** 41.314 empresas saudáveis (95,18%) contra 2.091 empresas insolventes (4,82%). Razão de desbalanceamento de aproximadamente 19,76:1.
- **Tratamento de nulos/infinitos:** Imputação robusta pela mediana amostral de cada atributo para preservar estabilidade numérica sem sofrer distorções por valores extremos.

---

## 2. Seleção de Atributos (Feature Selection)
Foram selecionados os 10 indicadores financeiros mais representativos e consolidados na literatura contábil:
- `Attr1`: Lucro Líquido / Ativo Total
- `Attr2`: Passivo Total / Ativo Total (Endividamento Geral)
- `Attr3`: Capital de Giro / Ativo Total
- `Attr4`: Ativo Circulante / Passivo Circulante (Liquidez Corrente)
- `Attr5`: [(Caixa + Títulos + Recebíveis - Passivo Circulante) / (Despesas Operacionais - Depreciação)] * 365
- `Attr6`: Lucros Retidos / Ativo Total
- `Attr7`: EBIT / Ativo Total (Margem Operacional dos Ativos)
- `Attr8`: Valor Contábil do Patrimônio Líquido / Passivo Total
- `Attr9`: Vendas / Ativo Total (Giro do Ativo)
- `Attr10`: Patrimônio Líquido / Ativo Total

---

## 3. Comparação de Modelos de Classificação
A divisão de dados seguiu proporção 80% treino (34.724 amostras) e 20% teste (8.681 amostras) estratificada.

| Modelo | Acurácia | Acurácia Balanceada | Precisão (Classe 1) | Recall (Classe 1) | F1-Score (Classe 1) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística (Baseline)** | 70,30% | 63,73% | 8,97% | 56,46% | 15,48% | 0,6746 |
| **Random Forest (Selecionado)** | **82,29%** | 67,99% | **14,02%** | 52,15% | **22,10%** | **0,7902** |
| **XGBoost** | 78,95% | **71,23%** | 13,55% | **62,68%** | 22,29% | **0,8048** |

### Justificativa da Escolha do Modelo:
O **Random Forest** foi selecionado como modelo de produção por apresentar a maior Acurácia geral (82,29%), alta Precisão e F1-Score, excelente estabilidade contra overfitting e perfeita compatibilidade de conversão para o formato padrão industrial **ONNX** via `skl2onnx`.

---

## 4. Importância dos Atributos (Feature Importance)
1. `Attr6` (Lucros Retidos / Ativo Total): **15,92%**
2. `Attr1` (Lucro Líquido / Ativo Total): **15,41%**
3. `Attr7` (EBIT / Ativo Total): **13,55%**
4. `Attr5` (Liquidez Financeira): **9,38%**
5. `Attr10` (Patrimônio Líquido / Ativo Total): **8,79%**
6. `Attr9` (Giro do Ativo): **8,05%**
7. `Attr4` (Liquidez Corrente): **7,80%**
8. `Attr8` (PL / Passivo): **7,42%**
9. `Attr2` (Passivo / Ativo): **6,92%**
10. `Attr3` (Capital de Giro / Ativo): **6,78%**

---

## 5. Artefatos Exportados
- `modelo_risco.onnx`: Modelo serializado no formato Open Neural Network Exchange para execução nativa no backend C# (.NET) e Python.
- `modelo_risco.joblib`: Modelo serializado em formato Joblib para uso rápido em ambientes Python.
- `model_metadata.json`: Metadados completos incluindo nomes de atributos, descrições, medianas calculadas, thresholds de classificação de risco e métricas.
- `Treinamento_e_Comparacao_Modelos.ipynb`: Notebook Jupyter contendo todas as células executadas, análises e visualizações.
- `treinar_modelos.py`: Script reprodutível para re-execução do pipeline de treinamento.
