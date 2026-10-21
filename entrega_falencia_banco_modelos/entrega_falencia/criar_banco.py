"""Cria/atualiza o SQLite do projeto usando SOMENTE os cinco arquivos CSV locais.

Esperado em dados/: 1year.csv, 2year.csv, 3year.csv, 4year.csv e 5year.csv.
As tabelas preservam Attr1..Attr64 e class; a imputação não é feita aqui,
pois deve ser aprendida exclusivamente sobre o conjunto de treinamento.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text

BASE_DIR = Path(__file__).resolve().parent
PASTA_DADOS_PADRAO = BASE_DIR / "dados"
BANCO_PADRAO = BASE_DIR / "falencia_empresas.db"
FEATURES = [f"Attr{i}" for i in range(1, 65)]
ALVO = "class"
COLUNAS_ESPERADAS = FEATURES + [ALVO]


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def ler_csv(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, na_values=["?", "NA", "N/A", "NaN", ""] , low_memory=False)

    # Remove apenas colunas de índice acidentais, não atributos financeiros.
    colunas_indice = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if colunas_indice:
        df = df.drop(columns=colunas_indice)

    ausentes = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if ausentes:
        raise ValueError(
            f"O arquivo {caminho.name} não tem todas as colunas esperadas. "
            f"Colunas ausentes: {ausentes[:10]}"
        )

    # Usa somente os 64 atributos originais e a variável-alvo class.
    df = df[COLUNAS_ESPERADAS].copy()
    for coluna in FEATURES:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    df[ALVO] = pd.to_numeric(df[ALVO], errors="coerce")
    if df[ALVO].isna().any():
        linhas_invalidas = int(df[ALVO].isna().sum())
        raise ValueError(
            f"{caminho.name} tem {linhas_invalidas} rótulos 'class' ausentes/inválidos. "
            "Revise o CSV para não treinar com classe desconhecida."
        )
    classes = set(df[ALVO].unique().tolist())
    if not classes.issubset({0, 1}):
        raise ValueError(f"Valores inesperados na coluna class de {caminho.name}: {classes}")
    df[ALVO] = df[ALVO].astype("int64")
    return df


def criar_banco(
    pasta_csv: str | Path = PASTA_DADOS_PADRAO,
    caminho_banco: str | Path = BANCO_PADRAO,
) -> dict:
    """Importa os cinco CSVs para tabelas SQL e cria a view consolidada."""
    pasta_csv = Path(pasta_csv).resolve()
    caminho_banco = Path(caminho_banco).resolve()
    caminho_banco.parent.mkdir(parents=True, exist_ok=True)

    faltando = [pasta_csv / f"{ano}year.csv" for ano in range(1, 6)
                if not (pasta_csv / f"{ano}year.csv").is_file()]
    if faltando:
        arquivos = ", ".join(p.name for p in faltando)
        raise FileNotFoundError(
            f"Não encontrei estes CSVs em {pasta_csv}: {arquivos}. "
            "Coloque os cinco arquivos na pasta dados/. O script não baixa ARFF/ZIP."
        )

    engine = create_engine(f"sqlite:///{caminho_banco.as_posix()}")
    resumo = {}
    metadados = []
    tabelas = []
    momento = datetime.now(timezone.utc).isoformat()

    try:
        for ano in range(1, 6):
            arquivo = pasta_csv / f"{ano}year.csv"
            df = ler_csv(arquivo)
            nome_tabela = f"falencia_{ano}_ano"

            # Cada horizonte tem tabela própria. Dados faltantes ficam NULL no SQLite.
            df.to_sql(nome_tabela, con=engine, if_exists="replace", index=False, chunksize=2000)
            tabelas.append(nome_tabela)

            ausentes = int(df[FEATURES].isna().sum().sum())
            registro = {
                "horizonte_anos": ano,
                "arquivo_csv": arquivo.name,
                "tabela_sql": nome_tabela,
                "registros": int(len(df)),
                "colunas": int(len(df.columns)),
                "empresas_falidas": int((df[ALVO] == 1).sum()),
                "empresas_nao_falidas": int((df[ALVO] == 0).sum()),
                "valores_ausentes_atributos": ausentes,
                "sha256_csv": sha256_arquivo(arquivo),
                "importado_em_utc": momento,
            }
            resumo[ano] = registro
            metadados.append(registro)
            print(
                f"{arquivo.name}: {len(df):,} registros, {len(df.columns)} colunas, "
                f"{ausentes:,} valores ausentes nos atributos."
            )

        pd.DataFrame(metadados).to_sql(
            "metadados_carga", con=engine, if_exists="replace", index=False
        )

        # A view acrescenta o horizonte de previsão sem alterar os CSVs/tabelas originais.
        with engine.begin() as conexao:
            conexao.execute(text('DROP VIEW IF EXISTS "falencia_consolidada"'))
            partes = [
                f'SELECT {ano} AS ano_previsao, * FROM "falencia_{ano}_ano"'
                for ano in range(1, 6)
            ]
            conexao.execute(text(
                'CREATE VIEW "falencia_consolidada" AS ' + " UNION ALL ".join(partes)
            ))

        total = sum(item["registros"] for item in metadados)
        resumo_final = {
            "banco": str(caminho_banco),
            "tabelas": tabelas,
            "total_linhas_csv": total,
            "total_colunas_por_tabela": len(COLUNAS_ESPERADAS),
            "total_falidas": sum(item["empresas_falidas"] for item in metadados),
            "total_nao_falidas": sum(item["empresas_nao_falidas"] for item in metadados),
            "total_valores_ausentes_atributos": sum(item["valores_ausentes_atributos"] for item in metadados),
            "observacao": "Dados brutos preservados; a imputação é feita durante o treinamento.",
        }
        print("\nBanco SQLite criado/atualizado:", caminho_banco)
        print("Tabelas:", ", ".join(tabelas))
        print(f"Registros totais antes da deduplicação do treinamento: {total:,}")
        return resumo_final
    finally:
        engine.dispose()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Importa exclusivamente CSVs para SQLite.")
    parser.add_argument("--dados", default=str(PASTA_DADOS_PADRAO), help="Pasta com 1year.csv a 5year.csv")
    parser.add_argument("--banco", default=str(BANCO_PADRAO), help="Caminho do arquivo SQLite")
    args = parser.parse_args()
    criar_banco(args.dados, args.banco)


if __name__ == "__main__":
    main()
