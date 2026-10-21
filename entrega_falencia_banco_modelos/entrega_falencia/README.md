# Previsão de falência — banco SQLite + Machine Learning

Projeto acadêmico que usa os cinco CSVs da base Polish Companies Bankruptcy Data para criar um banco SQLite, preparar os dados, comparar Random Forest e XGBoost, avaliar os resultados e salvar o modelo para integração com o sistema.

## O que foi integrado

1. **Banco de dados:** `criar_banco.py` importa somente `1year.csv` a `5year.csv` de `dados/` para tabelas separadas no SQLite.
2. **Validação/estrutura:** cria a view `falencia_consolidada` e a tabela `metadados_carga`, com quantidades, ausências e hash SHA-256 dos arquivos usados.
3. **Preparação:** converte atributos numéricos, transforma `?`, `NA` e infinitos em ausentes, remove linhas sem alvo válido e aplica imputação pela mediana dentro dos pipelines. O imputador é ajustado no treino, não antes da divisão.
4. **Atributos:** seleciona os 20 atributos mais relacionados ao alvo usando apenas os dados de treino.
5. **Treinamento e avaliação:** treina Random Forest e XGBoost, seleciona o limiar com dados de validação e avalia uma vez no conjunto de teste. Gera precisão, recall, F1-score, acurácia, ROC-AUC, average precision, relatório de classificação e matriz de confusão.
6. **Entrega:** salva `modelo_falencia.joblib`, os modelos comparativos, `comparacao_modelos.csv`, `resultados_treinamento.json` e `atributos_selecionados.csv`.

## Estrutura

```text
entrega_falencia/
├── dados/
│   ├── 1year.csv
│   ├── 2year.csv
│   ├── 3year.csv
│   ├── 4year.csv
│   ├── 5year.csv
│   └── AnlisedeDados_original.ipynb
├── criar_banco.py
├── executar_pipeline.py
├── treinar_modelos.py
├── prever_falencia.py
├── treinamento_modelos_falencia.ipynb
├── falencia_empresas.db          # banco SQLite já gerado; pode ser recriado
├── modelo_falencia.joblib        # gerado ao treinar
├── modelo_random_forest.joblib
├── modelo_xgboost.joblib
├── comparacao_modelos.csv
├── resultados_treinamento.json
├── atributos_selecionados.csv
├── RELATORIO_TREINAMENTO.md
└── requirements.txt
```

## Instalação

Recomendado: Python 3.11 ou 3.12, em um ambiente virtual.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

Linux/macOS:
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Executar o fluxo completo

No terminal, dentro da pasta `entrega_falencia`:

```bash
python executar_pipeline.py
```

O script usa somente os CSVs da pasta `dados/`. Não tenta obter ARFF nem baixar arquivos de outras fontes.

Também é possível executar por etapa:

```bash
python criar_banco.py --dados dados --banco falencia_empresas.db
python treinar_modelos.py --banco falencia_empresas.db --saida .
```

Para reimportar CSVs atualizados, execute `criar_banco.py` novamente. As cinco tabelas de dados e os metadados são recriados com base nos CSVs presentes.

## Estrutura do SQLite

- `falencia_1_ano`
- `falencia_2_ano`
- `falencia_3_ano`
- `falencia_4_ano`
- `falencia_5_ano`
- `metadados_carga`
- view `falencia_consolidada`, que acrescenta `ano_previsao` para identificar o horizonte de cada registro.

Cada tabela guarda os 64 atributos (`Attr1` a `Attr64`) e a classe original (`class`). Valores financeiros ausentes ficam como NULL no banco. Eles não são preenchidos na importação para que a base de origem permaneça rastreável.

## Notebook

- `dados/AnlisedeDados_original.ipynb`: notebook original de análise exploratória, mantido sem alterações.
- `treinamento_modelos_falencia.ipynb`: notebook integrado que cria o SQLite, resume os CSVs, apresenta a qualidade dos dados, treina os modelos a partir do banco e exibe a comparação.

Abra `treinamento_modelos_falencia.ipynb` a partir da raiz da pasta do projeto e execute as células em ordem.

## Usar o modelo no backend

Mantenha `prever_falencia.py` e `modelo_falencia.joblib` na mesma pasta:

```python
from prever_falencia import prever_falencia

# Exemplo apenas de formato. Substitua pelos indicadores financeiros reais.
dados_empresa = {"Attr29": 0.1, "Attr3": 0.2, "Attr51": 0.05}
resultado = prever_falencia(dados_empresa)
print(resultado)
```

Os atributos selecionados ausentes são imputados pela mediana aprendida no treino. A função rejeita uma entrada em que todos os atributos selecionados estejam ausentes. Para previsões válidas, informe indicadores financeiros calculados corretamente.

## Nota sobre a análise original

A base consolidada original tem 43.405 linhas antes da regra de deduplicação usada no treinamento e 65 colunas (64 atributos financeiros e `class`). Os dados são bastante desbalanceados: a falência é a classe minoritária. O notebook original faz preenchimento por mediana durante a análise exploratória; para avaliação de modelos, o novo fluxo ajusta a imputação dentro do pipeline de treino para reduzir vazamento de informação.

As métricas são uma avaliação experimental sobre essa base histórica. Não devem ser interpretadas como garantia de falência nem usadas como única base para decisão financeira. É recomendável validação temporal/externa e avaliação do custo de falsos positivos e falsos negativos antes de qualquer uso real.
