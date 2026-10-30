"""
Script de Treinamento e Comparação de Modelos de Classificação de Risco de Insolvência
Integrante 1 — Responsável: Mateus

Etapas:
1. Carregamento dos dados históricos de 1 a 5 anos (43.405 empresas).
2. Tratamento de valores ausentes por imputação de mediana.
3. Seleção de atributos financeiros (Attr1 a Attr10).
4. Treinamento comparativo entre Random Forest, XGBoost e Regressão Logística.
5. Avaliação completa com Acurácia, Precisão, Recall, F1-Score, ROC-AUC e Matriz de Confusão.
6. Exportação do modelo para ONNX e Joblib, gerando metadados atualizados.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import joblib

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    print("=" * 70)
    print("INTEGRANTE 1 — TREINAMENTO E COMPARAÇÃO DE MODELOS")
    print("=" * 70)

    # 1. Carregamento
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dados_dir = os.path.join(base_dir, "dados")
    
    csv_paths = [os.path.join(dados_dir, f"{i}year.csv") for i in range(1, 6)]
    dfs = []
    for i, p in enumerate(csv_paths, start=1):
        if not os.path.exists(p):
            # fallback para scratch se necessário
            p = f"C:/Users/guibi/.gemini/antigravity/scratch/bankruptcy-analysis/Trabalho-Projeto-Integrdor-III-main/{i}year.csv"
        print(f"Lendo ano {i}: {p}")
        df_ano = pd.read_csv(p, na_values="?")
        dfs.append(df_ano)

    df_full = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal de registros carregados: {df_full.shape[0]} linhas x {df_full.shape[1]} colunas.")

    # 2. Definição dos atributos e descrições
    features = [f"Attr{i}" for i in range(1, 11)]
    feature_descriptions = {
        "Attr1": "Lucro Líquido / Ativo Total",
        "Attr2": "Passivo Total / Ativo Total",
        "Attr3": "Capital de Giro / Ativo Total",
        "Attr4": "Ativo Circulante / Passivo Circulante (Liquidez Corrente)",
        "Attr5": "[(Caixa + Títulos + Recebíveis - Passivo Circulante) / (Despesas Operacionais - Depreciação)] * 365",
        "Attr6": "Lucros Retidos / Ativo Total",
        "Attr7": "EBIT / Ativo Total",
        "Attr8": "Patrimônio Líquido / Passivo Total",
        "Attr9": "Vendas / Ativo Total",
        "Attr10": "Patrimônio Líquido / Ativo Total"
    }

    # 3. Tratamento de valores ausentes e infinitos
    X = df_full[features].copy()
    y = df_full["class"].astype(int).copy()

    # Substituir infinitos por NaN antes de imputar
    X = X.replace([np.inf, -np.inf], np.nan)

    # Calcular medianas reais
    medians = X.median().round(4).to_dict()
    print("\nMedianas calculadas do conjunto completo:")
    for k, v in medians.items():
        print(f"  {k} ({feature_descriptions[k]}): {v}")

    # Imputar pela mediana
    X_imputed = X.fillna(X.median())

    # Distribuicao de classes
    qtd_0 = int((y == 0).sum())
    qtd_1 = int((y == 1).sum())
    prop_1 = qtd_1 / len(y)
    print(f"\nDistribuição da variável alvo (class):")
    print(f"  Empresas Saudáveis (Classe 0): {qtd_0} ({100*(1-prop_1):.2f}%)")
    print(f"  Empresas Falidas   (Classe 1): {qtd_1} ({100*prop_1:.2f}%)")
    scale_pos_weight = qtd_0 / qtd_1
    print(f"  Razão de desbalanceamento (scale_pos_weight): {scale_pos_weight:.2f}")

    # 4. Separação em Treino e Teste (80% treino / 20% teste estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nDivisão dos dados:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Teste:  {X_test.shape[0]} amostras")

    # 5. Modelos a comparar
    modelos = {
        "Regressão Logística (Baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    }

    resultados = {}
    print("\n" + "=" * 70)
    print("TREINAMENTO E AVALIAÇÃO COMPARATIVA")
    print("=" * 70)

    for nome, mod in modelos.items():
        print(f"\n>>> Treinando: {nome}...")
        mod.fit(X_train, y_train)

        y_pred = mod.predict(X_test)
        y_prob = mod.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        resultados[nome] = {
            "Acurácia": round(float(acc), 4),
            "Acurácia Balanceada": round(float(bal_acc), 4),
            "Precisão (Classe 1)": round(float(prec), 4),
            "Recall (Classe 1)": round(float(rec), 4),
            "F1-Score (Classe 1)": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Matriz de Confusão": cm.tolist()
        }

        print(f"Resultados {nome}:")
        print(f"  Acurácia:            {acc:.4f}")
        print(f"  Acurácia Balanceada: {bal_acc:.4f}")
        print(f"  Precisão (Classe 1): {prec:.4f}")
        print(f"  Recall (Classe 1):   {rec:.4f}")
        print(f"  F1-Score (Classe 1): {f1:.4f}")
        print(f"  ROC-AUC:             {auc:.4f}")
        print(f"  Matriz de Confusão:\n{cm}")

    # Tabela comparativa
    tabela_comp = pd.DataFrame(resultados).T
    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA DE DESEMPENHO")
    print("=" * 70)
    print(tabela_comp[["Acurácia", "Acurácia Balanceada", "Precisão (Classe 1)", "Recall (Classe 1)", "F1-Score (Classe 1)", "ROC-AUC"]])

    # 6. Escolha do melhor modelo para produção
    # O Random Forest equilibra estabilidade, facilidade de exportação ONNX e alto ROC-AUC
    melhor_nome = "Random Forest"
    melhor_modelo = modelos[melhor_nome]
    print(f"\nModelo selecionado para integração no sistema: {melhor_nome}")

    # Feature importances
    if hasattr(melhor_modelo, "feature_importances_"):
        importances = pd.Series(melhor_modelo.feature_importances_, index=features).sort_values(ascending=False)
        print("\nImportância dos Atributos no Random Forest:")
        for feat, imp in importances.items():
            print(f"  {feat:8s} ({feature_descriptions[feat]}): {imp:.4f}")

    # 7. Salvar Modelo em Joblib
    joblib_path = os.path.join(base_dir, "modelo_risco.joblib")
    joblib.dump(melhor_modelo, joblib_path)
    print(f"\nModelo salvo em Joblib: {joblib_path}")

    # 8. Converter e Salvar em ONNX
    print("\nConvertendo modelo para ONNX (skl2onnx)...")
    initial_type = [("float_input", FloatTensorType([None, len(features)]))]
    onnx_model = convert_sklearn(
        melhor_modelo,
        initial_types=initial_type,
        target_opset=12,
        options={id(melhor_modelo): {"zipmap": False}}
    )
    onnx_path = os.path.join(base_dir, "modelo_risco.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Modelo salvo em ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")

    # 9. Metadados do Modelo
    metadata = {
        "feature_names": features,
        "feature_descriptions": feature_descriptions,
        "medians": medians,
        "threshold_medio_risco": 0.35,
        "threshold_alto_risco": 0.65,
        "modelo_informacoes": {
            "nome_modelo": melhor_nome,
            "tipo": "Classificador Tabular Random Forest",
            "quantidade_amostras_treinamento": int(X_train.shape[0]),
            "quantidade_amostras_teste": int(X_test.shape[0]),
            "metricas_teste": resultados[melhor_nome],
            "importancia_atributos": importances.to_dict()
        }
    }

    meta_path = os.path.join(base_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Metadados salvos em: {meta_path}")

    # 10. Distribuir artefatos para Backend e Frontend
    destinos = [
        os.path.join(base_dir, "..", "02_Backend_Integracao", "RiskAnalysisBackend", "ModelsML"),
        os.path.join(base_dir, "..", "04_Frontend_Streamlit"),
        os.path.join("C:/Users/guibi/Downloads/Integrante4_Frontend_Streamlit/Integrante4_Frontend_Streamlit")
    ]
    for d in destinos:
        if os.path.exists(d):
            import shutil
            shutil.copy2(onnx_path, os.path.join(d, "modelo_risco.onnx"))
            shutil.copy2(meta_path, os.path.join(d, "model_metadata.json"))
            shutil.copy2(joblib_path, os.path.join(d, "modelo_risco.joblib"))
            print(f"Artefatos sincronizados com: {d}")

    print("\n" + "=" * 70)
    print("INTEGRANTE 1 — CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
