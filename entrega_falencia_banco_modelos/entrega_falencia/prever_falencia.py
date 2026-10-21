"""Funções de inferência para o modelo de previsão de falência.

O arquivo modelo_falencia.joblib deve ficar na mesma pasta deste módulo.
Entrada: dicionário ou DataFrame com os atributos financeiros Attr1..Attr64
(para o modelo escolhido, os 20 atributos selecionados são usados).
"""
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().with_name("modelo_falencia.joblib")


def carregar_modelo():
    """Carrega uma vez o bundle do modelo e seus metadados."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def prever_falencia(dados: Union[dict, pd.DataFrame], bundle=None):
    """Retorna classe prevista, score do modelo e campos imputados.

    `dados` pode ser um dicionário para uma empresa ou um DataFrame para várias.
    Campos selecionados ausentes/não numéricos são tratados pela mediana aprendida
    durante o treinamento. Para uso real, informe os valores financeiros corretos.
    """
    bundle = bundle or carregar_modelo()
    selecao = list(bundle["features_selecionadas"])
    df = pd.DataFrame([dados]) if isinstance(dados, dict) else dados.copy()

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Informe um dicionário com os atributos ou um DataFrame não vazio.")

    # Constrói a entrada na ordem esperada pelo pipeline.
    for col in selecao:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(
            df[col].replace(["?", "", "NA", "N/A"], np.nan), errors="coerce"
        )

    if df[selecao].notna().sum(axis=1).eq(0).any():
        raise ValueError(
            "Cada empresa precisa ter ao menos um atributo selecionado informado. "
            "Campos faltantes são imputados pela mediana de treinamento."
        )

    pipeline = bundle["pipeline"]
    probabilidades = pipeline.predict_proba(df[selecao].to_numpy(dtype=float))[:, 1]
    limiar = float(bundle["limiar_decisao"])
    classes = (probabilidades >= limiar).astype(int)

    saidas = []
    for pos, (_, linha) in enumerate(df.iterrows()):
        ausentes = [c for c in selecao if pd.isna(linha[c])]
        saidas.append({
            "modelo": bundle["nome_modelo"],
            "classe_predita": int(classes[pos]),
            "situacao_predita": (
                "Sinalizada para análise de risco de falência"
                if classes[pos] == 1
                else "Não sinalizada pelo modelo"
            ),
            "score_estimado_falencia": round(float(probabilidades[pos]), 4),
            "limiar_decisao": limiar,
            "atributos_imputados_por_mediana": ausentes,
        })

    return saidas[0] if isinstance(dados, dict) else saidas


if __name__ == "__main__":
    # Exemplo sintético apenas para demonstrar o formato; substitua pelos dados reais.
    exemplo = {attr: 0.0 for attr in [
        "Attr29", "Attr3", "Attr51", "Attr6", "Attr2", "Attr1", "Attr25",
        "Attr35", "Attr48", "Attr55", "Attr11", "Attr24", "Attr22", "Attr57",
        "Attr18", "Attr14", "Attr7", "Attr10", "Attr38", "Attr5"
    ]}
    print(prever_falencia(exemplo))
