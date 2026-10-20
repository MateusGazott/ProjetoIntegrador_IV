from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=RuntimeWarning)
RANDOM_STATE = 42
TOP_K = 20
TARGET = 'class'
FEATURES = [f'Attr{i}' for i in range(1, 65)]
DESCRICOES = {
    'Attr1':'Lucro Líquido / Ativos Totais','Attr2':'Passivo Total / Ativos Totais','Attr3':'Capital de Giro / Ativos Totais','Attr4':'Ativo Circulante / Passivo Circulante','Attr5':'Lucros Retidos / Ativos Totais','Attr6':'EBIT / Ativos Totais','Attr7':'Patrimônio Líquido / Passivo Total','Attr8':'Vendas / Ativos Totais','Attr9':'Lucro Bruto / Ativos Totais','Attr10':'Capital Próprio / Ativos Totais','Attr11':'Lucro Operacional / Ativos Totais','Attr12':'Lucro Bruto / Passivo de Curto Prazo','Attr13':'Lucro Bruto / Vendas','Attr14':'Lucro Operacional / Passivo Total','Attr15':'Ativo Total / Passivo Total','Attr16':'Lucro Bruto / Ativo Circulante','Attr17':'Ativo Total / Passivo Circulante','Attr18':'Lucro Bruto / Ativos Totais','Attr19':'Lucro Bruto / Vendas Líquidas','Attr20':'Estoque / Vendas','Attr21':'Vendas / Contas a Receber','Attr22':'Lucro Líquido / Vendas','Attr23':'Lucro Líquido / Ativo Circulante','Attr24':'Lucro Líquido / Estoque','Attr25':'Patrimônio Líquido / Ativos Totais','Attr26':'Lucros Retidos / Patrimônio Líquido','Attr27':'Lucro Operacional / Patrimônio Líquido','Attr28':'Capital de Giro / Passivo Total','Attr29':'Logaritmo do Ativo Total','Attr30':'Passivo Total / Vendas','Attr31':'Lucro Operacional / Vendas','Attr32':'Passivo Circulante / Estoque','Attr33':'Passivo Circulante / Vendas','Attr34':'Capital de Giro / Custos Operacionais','Attr35':'Lucro Bruto / Depreciação','Attr36':'Vendas / Ativo Total Médio','Attr37':'Ativo Circulante / Ativo Total','Attr38':'Lucro Líquido / Patrimônio Líquido','Attr39':'Passivo de Longo Prazo / Ativos Totais','Attr40':'Capital de Giro / Estoque','Attr41':'Ativo Circulante / Passivo Total','Attr42':'Ativo Circulante / Vendas','Attr43':'Contas a Receber / Vendas','Attr44':'Estoque / Ativo Circulante','Attr45':'Estoque / Passivo Circulante','Attr46':'Vendas / Estoque','Attr47':'EBITDA / Ativos Totais','Attr48':'EBITDA / Vendas','Attr49':'EBITDA / Passivo Total','Attr50':'Caixa / Passivo Total','Attr51':'Caixa / Ativos Totais','Attr52':'Caixa / Vendas','Attr53':'Ativo Circulante / Receita Operacional','Attr54':'Passivo Circulante / Ativo Total','Attr55':'Passivo Circulante / Patrimônio Líquido','Attr56':'Capital de Giro / Receita Operacional','Attr57':'Lucro Operacional / Ativo Circulante','Attr58':'Lucro Líquido / Passivo Total','Attr59':'Lucro Líquido / Passivo Circulante','Attr60':'Lucro Líquido / Capital de Giro','Attr61':'Vendas / Patrimônio Líquido','Attr62':'Passivo Total / Patrimônio Líquido','Attr63':'Capital de Giro / Ativo Circulante','Attr64':'Capital de Giro / Passivo Circulante'
}


def carregar_dados(pasta_csv: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    partes = []
    resumo = {}
    for ano in range(1, 6):
        caminho = pasta_csv / f'{ano}year.csv'
        df_ano = pd.read_csv(caminho, na_values=['?', 'NA', 'N/A', ''])
        faltantes = int(df_ano[FEATURES].isna().sum().sum())
        df_ano[FEATURES] = df_ano[FEATURES].apply(pd.to_numeric, errors='coerce')
        df_ano[FEATURES] = df_ano[FEATURES].replace([np.inf, -np.inf], np.nan)
        df_ano[TARGET] = pd.to_numeric(df_ano[TARGET], errors='coerce')
        df_ano = df_ano.dropna(subset=[TARGET])
        df_ano[TARGET] = df_ano[TARGET].astype(int)
        resumo[f'{ano}year.csv'] = {
            'registros': int(len(df_ano)),
            'falidas': int(df_ano[TARGET].sum()),
            'nao_falidas': int((df_ano[TARGET] == 0).sum()),
            'valores_ausentes_atributos_antes_tratamento': faltantes,
        }
        partes.append(df_ano[FEATURES + [TARGET]])

    completo = pd.concat(partes, ignore_index=True)
    linhas_originais = len(completo)
    # Remove somente linhas integralmente duplicadas (atributos e classe iguais).
    completo = completo.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
    resumo['consolidado'] = {
        'registros_antes_remocao_duplicatas': int(linhas_originais),
        'duplicatas_removidas': int(linhas_originais - len(completo)),
        'registros_finais': int(len(completo)),
        'falidas': int(completo[TARGET].sum()),
        'nao_falidas': int((completo[TARGET] == 0).sum()),
        'percentual_falidas': round(float(completo[TARGET].mean() * 100), 4),
        'total_ausentes_nos_atributos': int(completo[FEATURES].isna().sum().sum()),
    }
    return completo[FEATURES], completo[TARGET], resumo


def escolher_atributos(X_train: pd.DataFrame, y_train: pd.Series, k: int = TOP_K) -> tuple[list[str], dict]:
    # A seleção é calculada apenas no conjunto de treino para evitar vazamento de dados.
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    corrs = X_imp.corrwith(y_train).replace([np.inf, -np.inf], np.nan).fillna(0).abs()
    ranking = corrs.sort_values(ascending=False)
    selecionados = ranking.head(k).index.tolist()
    return selecionados, {str(k): float(v) for k, v in ranking.items()}


def construir_modelos(y_train: pd.Series) -> dict:
    positivos = int((y_train == 1).sum())
    negativos = int((y_train == 0).sum())
    razao_classe = negativos / max(positivos, 1)

    rf = RandomForestClassifier(
        n_estimators=35,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb = XGBClassifier(
        n_estimators=70,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=razao_classe,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method='hist',
    )
    return {'Random Forest': rf, 'XGBoost': xgb}


def pipeline_para(modelo) -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classificador', modelo),
    ])


def selecionar_limiar(y_true, probabilidades) -> dict:
    melhor = None
    # A escolha do limiar usa apenas validação, não o conjunto de teste.
    for limiar in np.arange(0.05, 0.951, 0.01):
        pred = (probabilidades >= limiar).astype(int)
        item = {
            'limiar': float(round(limiar, 2)),
            'precision': float(precision_score(y_true, pred, zero_division=0)),
            'recall': float(recall_score(y_true, pred, zero_division=0)),
            'f1': float(f1_score(y_true, pred, zero_division=0)),
        }
        # Em empate no F1, favorece maior recall para reduzir falsos negativos.
        chave = (item['f1'], item['recall'], item['precision'])
        if melhor is None or chave > melhor[0]:
            melhor = (chave, item)
    return melhor[1]


def avaliar(y_true, probabilidades, limiar: float) -> dict:
    pred = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'limiar': float(limiar),
        'acuracia': float(accuracy_score(y_true, pred)),
        'precisao_falencia': float(precision_score(y_true, pred, zero_division=0)),
        'recall_falencia': float(recall_score(y_true, pred, zero_division=0)),
        'f1_falencia': float(f1_score(y_true, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, probabilidades)),
        'average_precision': float(average_precision_score(y_true, probabilidades)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'matriz_confusao': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'classification_report': classification_report(
            y_true, pred, labels=[0, 1], target_names=['Não falida', 'Falida'],
            output_dict=True, zero_division=0,
        ),
    }


def main(pasta_csv: str | Path, pasta_saida: str | Path) -> dict:
    pasta_csv = Path(pasta_csv)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print('Lendo dados...', flush=True)
    X, y, resumo_dados = carregar_dados(pasta_csv)
    print(f'Dados carregados: {len(X)} linhas', flush=True)
    X_treino_validacao, X_teste, y_treino_validacao, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_treino_validacao, y_treino_validacao, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_treino_validacao,
    )

    print('Selecionando atributos...', flush=True)
    atributos, ranking = escolher_atributos(X_treino, y_treino)
    print('Atributos selecionados', atributos, flush=True)
    modelos = construir_modelos(y_treino)
    resultados_validacao = {}
    limiares = {}

    for nome, estimador in modelos.items():
        print(f'Treinando para validação: {nome}', flush=True)
        pipe = pipeline_para(estimador)
        pipe.fit(X_treino[atributos], y_treino)
        print(f'Fim treino validação: {nome}', flush=True)
        prob_val = pipe.predict_proba(X_validacao[atributos])[:, 1]
        limiar = selecionar_limiar(y_validacao, prob_val)
        resultados_validacao[nome] = avaliar(y_validacao, prob_val, limiar['limiar'])
        resultados_validacao[nome]['limiar_otimizado_na_validacao'] = limiar
        limiares[nome] = limiar['limiar']

    # Escolha do modelo guiada pelo F1 da classe falida na validação.
    melhor_nome = max(
        resultados_validacao,
        key=lambda n: (resultados_validacao[n]['f1_falencia'],
                       resultados_validacao[n]['recall_falencia'],
                       resultados_validacao[n]['precisao_falencia']),
    )

    # Retreina os dois modelos com treino + validação e avalia uma única vez no teste.
    resultados_teste = {}
    pipes_finais = {}
    for nome, estimador in construir_modelos(y_treino_validacao).items():
        print(f'Treinando para teste: {nome}', flush=True)
        pipe = pipeline_para(estimador)
        pipe.fit(X_treino_validacao[atributos], y_treino_validacao)
        print(f'Fim treino teste: {nome}', flush=True)
        prob_teste = pipe.predict_proba(X_teste[atributos])[:, 1]
        resultados_teste[nome] = avaliar(y_teste, prob_teste, limiares[nome])
        pipes_finais[nome] = pipe

    melhor_pipe = pipes_finais[melhor_nome]
    prob_teste_melhor = melhor_pipe.predict_proba(X_teste[atributos])[:, 1]
    pred_teste_melhor = (prob_teste_melhor >= limiares[melhor_nome]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_teste, pred_teste_melhor, labels=[0, 1]).ravel()

    # Modelo final treinado com todos os dados disponíveis, mantendo atributos e limiar escolhidos.
    estimadores_finais = construir_modelos(y)
    modelo_final = pipeline_para(estimadores_finais[melhor_nome])
    print(f'Treinando modelo final {melhor_nome} com todos os dados...', flush=True)
    modelo_final.fit(X[atributos], y)
    print('Modelo final concluído', flush=True)
    bundle = {
        'pipeline': modelo_final,
        'features_entrada': FEATURES,
        'features_selecionadas': atributos,
        'limiar_decisao': limiares[melhor_nome],
        'classe_positiva': 1,
        'nome_modelo': melhor_nome,
        'versao': '1.0',
    }
    caminho_modelo = pasta_saida / 'modelo_falencia.joblib'
    joblib.dump(bundle, caminho_modelo, compress=3)
    # Também disponibiliza cada modelo para comparação / uso alternativo.
    for nome, pipe in pipes_finais.items():
        joblib.dump({
            'pipeline': pipe,
            'features_entrada': FEATURES,
            'features_selecionadas': atributos,
            'limiar_decisao': limiares[nome],
            'classe_positiva': 1,
            'nome_modelo': nome,
            'versao': '1.0',
        }, pasta_saida / ('modelo_' + ('random_forest' if nome == 'Random Forest' else 'xgboost') + '.joblib'), compress=3)

    resultado = {
        'problema': 'Classificação binária de falência empresarial',
        'alvo': {'coluna_original': 'class', '0': 'Não falida', '1': 'Falida'},
        'resumo_dados': resumo_dados,
        'divisao': {
            'treino': int(len(X_treino)), 'validacao': int(len(X_validacao)),
            'teste': int(len(X_teste)), 'total_sem_duplicatas': int(len(X)),
            'proporcao_treino_validacao_teste': '60% / 20% / 20%',
            'random_state': RANDOM_STATE,
        },
        'metodo_selecao_atributos': f'Top {TOP_K} por correlação absoluta de Pearson com o alvo, calculada apenas no treino',
        'atributos_selecionados': atributos,
        'ranking_correlacao_treino': ranking,
        'resultados_validacao': resultados_validacao,
        'resultados_teste': resultados_teste,
        'melhor_modelo_selecionado': melhor_nome,
        'limiar_final': limiares[melhor_nome],
        'matriz_confusao_teste_modelo_escolhido': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'classification_report_teste_modelo_escolhido': classification_report(
            y_teste, pred_teste_melhor, labels=[0, 1],
            target_names=['Não falida', 'Falida'], output_dict=True, zero_division=0,
        ),
        'observacao': 'As métricas de teste usam modelos treinados em 80% dos dados. O arquivo modelo_falencia.joblib é então retreinado com 100% dos dados para uso no sistema.',
    }
    with open(pasta_saida / 'resultados_treinamento.json', 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    linhas = []
    for nome in modelos:
        v = resultados_validacao[nome]
        t = resultados_teste[nome]
        linhas.append({
            'modelo': nome,
            'limiar_escolhido_validacao': limiares[nome],
            'validacao_precisao_falencia': v['precisao_falencia'],
            'validacao_recall_falencia': v['recall_falencia'],
            'validacao_f1_falencia': v['f1_falencia'],
            'teste_acuracia': t['acuracia'],
            'teste_precisao_falencia': t['precisao_falencia'],
            'teste_recall_falencia': t['recall_falencia'],
            'teste_f1_falencia': t['f1_falencia'],
            'teste_roc_auc': t['roc_auc'],
            'teste_average_precision': t['average_precision'],
            'teste_tn': t['tn'], 'teste_fp': t['fp'], 'teste_fn': t['fn'], 'teste_tp': t['tp'],
        })
    pd.DataFrame(linhas).to_csv(pasta_saida / 'comparacao_modelos.csv', index=False, encoding='utf-8-sig')

    pd.DataFrame({
        'atributo': atributos,
        'descricao': [DESCRICOES[a] for a in atributos],
        'correlacao_absoluta_treino': [ranking[a] for a in atributos],
    }).to_csv(pasta_saida / 'atributos_selecionados.csv', index=False, encoding='utf-8-sig')

    print(json.dumps({
        'dados': resumo_dados['consolidado'],
        'tamanhos': resultado['divisao'],
        'atributos_selecionados': atributos,
        'modelo_escolhido': melhor_nome,
        'validacao': {n: {k: round(resultados_validacao[n][k], 4) for k in ['limiar', 'acuracia', 'precisao_falencia', 'recall_falencia', 'f1_falencia', 'roc_auc', 'average_precision']} for n in modelos},
        'teste': {n: {k: round(resultados_teste[n][k], 4) for k in ['limiar', 'acuracia', 'precisao_falencia', 'recall_falencia', 'f1_falencia', 'roc_auc', 'average_precision', 'tn', 'fp', 'fn', 'tp']} for n in modelos},
        'model_size_bytes': os.path.getsize(caminho_modelo),
    }, ensure_ascii=False, indent=2))
    return resultado


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Treina e compara modelos de previsão de falência.')
    parser.add_argument('--dados', required=True, help='Pasta contendo 1year.csv até 5year.csv')
    parser.add_argument('--saida', default='artifacts', help='Pasta de saída para modelos e métricas')
    args = parser.parse_args()
    main(args.dados, args.saida)
