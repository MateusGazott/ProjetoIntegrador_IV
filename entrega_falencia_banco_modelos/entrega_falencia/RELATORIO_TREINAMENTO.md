# Relatório de treinamento — Previsão de falência de empresas

**Data de execução:** 21/10/2026  
**Objetivo:** treinar e comparar modelos para prever a classe `class` (0 = não falida; 1 = falida) usando os CSVs de 1 a 5 anos.

## 1. Análise dos arquivos e notebook existente

Foram analisados `1year.csv`, `2year.csv`, `3year.csv`, `4year.csv`, `5year.csv` e o notebook original `dados/AnlisedeDados_original.ipynb`. Cada CSV contém 64 atributos financeiros (`Attr1` a `Attr64`) e a coluna-alvo `class`. Os cinco CSVs também foram carregados no banco `falencia_empresas.db`, nas tabelas `falencia_1_ano` a `falencia_5_ano`; o treinamento foi executado lendo essas tabelas SQL.

O notebook existente já concatenava os cinco arquivos, renomeava atributos para português e substituía valores ausentes pela mediana. Contudo, ele era principalmente exploratório: não continha uma comparação completa de classificadores com conjunto de teste independente. Para esta etapa, o tratamento foi colocado dentro do pipeline de cada modelo, evitando que a mediana fosse calculada usando os dados de validação/teste.

## 2. Banco de dados integrado

O script `criar_banco.py` importa somente os cinco CSVs locais e cria cinco tabelas independentes, a tabela `metadados_carga` e a view `falencia_consolidada` com a coluna `ano_previsao`. Os CSVs originais não são sobrescritos; os valores ausentes ficam como `NULL` no banco, e a imputação é aplicada pelo pipeline de Machine Learning. A tabela de metadados registra contagens, valores ausentes e hash SHA-256 de cada CSV para rastreabilidade.

## 3. Preparação dos dados

| Indicador | Resultado |
|---|---:|
| Registros empilhados originalmente | 43,405 |
| Duplicatas exatas removidas | 401 |
| Registros após remoção de duplicatas | 43,004 |
| Empresas não falidas (classe 0) | 40,921 |
| Empresas falidas (classe 1) | 2,083 |
| Percentual de classe falida | 4.84% |
| Células ausentes nos 64 atributos antes da imputação | 41,037 (1.49% das células de atributos) |

Os símbolos `?`, campos vazios, valores não numéricos e infinitos foram tratados como ausentes. As duplicatas removidas eram linhas idênticas em todos os 64 atributos e na classe. A imputação utiliza a **mediana aprendida apenas no conjunto de treino**; assim, os valores de validação/teste não influenciam esse tratamento.

A classe positiva é minoritária. Por isso, a acurácia isolada não representa adequadamente o desempenho: detectar empresas falidas (recall) e equilibrar precisão/recall (F1) são medidas importantes.

## 4. Divisão dos dados e seleção de atributos

Foi feita uma divisão estratificada e reproduzível, com `random_state=42`:

| Parte | Registros | Proporção |
|---|---:|---:|
| Treino inicial | 25,802 | 60% |
| Validação | 8,601 | 20% |
| Teste independente | 8,601 | 20% |

Foram selecionados os 20 atributos de maior correlação absoluta de Pearson com a classe, calculada **somente no conjunto de treino**. A lista e as descrições em português estão em `atributos_selecionados.csv`. A seleção por correlação é uma heurística simples alinhada à análise exploratória existente; ela pode deixar de captar relações não lineares.

## 5. Modelos comparados

- **Random Forest:** 35 árvores, profundidade máxima 10, `min_samples_leaf=5`, balanceamento de classes.
- **XGBoost:** 70 estimadores, profundidade máxima 3, taxa de aprendizado 0,04 e `scale_pos_weight` para compensar o desbalanceamento.

Os limiares de decisão foram ajustados pela validação para maximizar o F1 da classe falida (em empate, favorecendo recall). Os resultados de teste abaixo usam esses limiares; o conjunto de teste não foi usado para escolher o limiar.

## 6. Comparação no conjunto de teste

| Métrica | Random Forest | XGBoost |
|---|---:|---:|
| Limiar de decisão | 0.53 | 0.66 |
| Acurácia | 0.8706 (87.06%) | 0.8691 (86.91%) |
| Precisão da classe falida | 0.1882 (18.82%) | 0.1815 (18.15%) |
| Recall da classe falida | 0.5036 (50.36%) | 0.4844 (48.44%) |
| F1 da classe falida | 0.2740 | 0.2641 |
| ROC-AUC | 0.8228 | 0.8008 |
| Average precision | 0.1983 | 0.1566 |

### Matrizes de confusão

Linhas = classe real (`0` não falida, `1` falida); colunas = classe prevista (`0`, `1`).

**Random Forest**

| | Prevista: não falida | Prevista: falida |
|---|---:|---:|
| Real: não falida | 7,278 (verdadeiros negativos) | 906 (falsos positivos) |
| Real: falida | 207 (falsos negativos) | 210 (verdadeiros positivos) |

**XGBoost**

| | Prevista: não falida | Prevista: falida |
|---|---:|---:|
| Real: não falida | 7,273 (verdadeiros negativos) | 911 (falsos positivos) |
| Real: falida | 215 (falsos negativos) | 202 (verdadeiros positivos) |

## 7. Modelo selecionado e interpretação

O **Random Forest** foi selecionado porque obteve o melhor F1 da classe falida na validação (0,3158 contra 0,2766 do XGBoost) e também ficou ligeiramente à frente no teste. O arquivo `modelo_falencia.joblib` contém o pipeline final do Random Forest retreinado com todos os 43,004 registros, além dos 20 atributos usados e do limiar de decisão **0.53**.

No teste, o Random Forest encontrou 210 das 417 empresas falidas (recall de aproximadamente 50.4%), mas também sinalizou 906 empresas não falidas (falsos positivos). Assim, o modelo pode ajudar a **priorizar análises**, mas ainda não deve ser usado sozinho para decisões financeiras ou para afirmar que uma empresa vai falir. Recomenda-se testar outros métodos de seleção de atributos, calibrar probabilidades, ajustar limiar de acordo com o custo dos erros e validar em dados mais recentes/externos antes de produção.

Como referência, sempre prever “não falida” teria acurácia aproximada de 95.16%, mas recall zero para falências. Isso demonstra por que a acurácia precisa ser analisada junto às métricas da classe minoritária.

## 8. Arquivos entregues

- `modelo_falencia.joblib`: modelo principal para integração, treinado em todo o conjunto.
- `modelo_random_forest.joblib` e `modelo_xgboost.joblib`: versões comparáveis treinadas em treino + validação, usadas para avaliar o teste.
- `comparacao_modelos.csv`: métricas de validação/teste.
- `resultados_treinamento.json`: resumo detalhado, matrizes e métricas por classe.
- `atributos_selecionados.csv`: atributos que o modelo realmente utiliza.
- `prever_falencia.py`: função de inferência de exemplo para API/backend.
- `treinar_modelos.py`: script de treinamento reprodutível.
- `treinamento_modelos_falencia.ipynb`: notebook integrado de criação do banco, avaliação dos dados, treinamento e consulta dos resultados.
- `criar_banco.py`: importação dos CSVs para SQLite.
- `executar_pipeline.py`: execução encadeada de importação e treinamento.
- `requirements.txt`: versões das dependências usadas.

## 9. Observação de uso

O score retornado pelo classificador é uma estimativa produzida pelo modelo e **não foi calibrado** como probabilidade financeira. Os atributos `Attr1`–`Attr64` são indicadores financeiros do conjunto de origem; é necessário mapear corretamente os dados contábeis da aplicação para essas razões. Este protótipo não substitui análise contábil, auditoria ou aconselhamento financeiro.
