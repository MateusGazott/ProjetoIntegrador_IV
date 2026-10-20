# Previsão de falência — artefatos de Machine Learning

## Modelo principal

`modelo_falencia.joblib` contém um pipeline Random Forest treinado com os 43,004 registros consolidados, com imputação por mediana e uso de 20 atributos financeiros. O limiar escolhido na validação é 0.53.

## Estrutura dos arquivos

- `modelo_falencia.joblib`: modelo principal para inferência.
- `modelo_random_forest.joblib` e `modelo_xgboost.joblib`: modelos comparativos avaliados no conjunto de teste.
- `prever_falencia.py`: helper de previsão para integrar ao backend/API.
- `treinar_modelos.py`: script de treinamento a partir dos cinco CSVs.
- `treinamento_modelos_falencia.ipynb`: notebook da etapa.
- `comparacao_modelos.csv`: comparação entre os modelos.
- `resultados_treinamento.json`: resultados detalhados.
- `atributos_selecionados.csv`: 20 variáveis do modelo, com descrições.
- `RELATORIO_TREINAMENTO.md`: relatório de método, métricas e limitações.
- `dados/`: arquivos CSV e cópia do notebook de análise original.

## Como instalar

Use uma versão Python compatível com os pacotes do `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Como usar no sistema

Deixe `prever_falencia.py` e `modelo_falencia.joblib` na mesma pasta:

```python
from prever_falencia import prever_falencia

dados_empresa = {
    "Attr29": 0.0,
    "Attr3": 0.0,
    "Attr51": 0.0,
    "Attr6": 0.0,
    "Attr2": 0.0,
    "Attr1": 0.0,
    "Attr25": 0.0,
    "Attr35": 0.0,
    "Attr48": 0.0,
    "Attr55": 0.0,
    "Attr11": 0.0,
    "Attr24": 0.0,
    "Attr22": 0.0,
    "Attr57": 0.0,
    "Attr18": 0.0,
    "Attr14": 0.0,
    "Attr7": 0.0,
    "Attr10": 0.0,
    "Attr38": 0.0,
    "Attr5": 0.0
}

resultado = prever_falencia(dados_empresa)
print(resultado)
```

**Importante:** os zeros acima são apenas um exemplo sintático, não dados reais nem recomendação de entrada. Substitua-os pelos indicadores financeiros da empresa, calculados de acordo com a definição dos atributos no dataset. O helper aceita apenas os atributos selecionados pelo modelo; atributos ausentes podem ser imputados pela mediana de treinamento, e os campos imputados são reportados no retorno.

Exemplo de retorno:

```json
{
  "modelo": "Random Forest",
  "classe_predita": 0,
  "situacao_predita": "Não sinalizada pelo modelo",
  "score_estimado_falencia": 0.31,
  "limiar_decisao": 0.53,
  "atributos_imputados_por_mediana": []
}
```

Os valores desse retorno são ilustrativos. O score não foi calibrado como probabilidade real de falência; use a previsão como apoio à análise, nunca como decisão automatizada isolada.

## Retreinar

O script espera que `1year.csv` a `5year.csv` estejam na mesma pasta de dados:

```bash
python treinar_modelos.py --dados dados --saida artifacts
```

Consulte `RELATORIO_TREINAMENTO.md` para metodologia, métricas e limitações.

## Resultados principais no teste

| Modelo | Precisão (falida) | Recall (falida) | F1 (falida) | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.188 | 0.504 | 0.274 | 0.823 |
| XGBoost | 0.181 | 0.484 | 0.264 | 0.801 |
