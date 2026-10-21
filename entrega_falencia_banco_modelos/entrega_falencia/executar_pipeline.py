"""Executa o fluxo completo: CSVs locais -> SQLite -> treinamento -> métricas/modelos."""
from __future__ import annotations

import argparse
from pathlib import Path

from criar_banco import criar_banco
from treinar_modelos import main as treinar_modelos

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa os cinco CSVs para SQLite e treina Random Forest/XGBoost."
    )
    parser.add_argument("--dados", default=str(BASE_DIR / "dados"),
                        help="Pasta com 1year.csv a 5year.csv")
    parser.add_argument("--banco", default=str(BASE_DIR / "falencia_empresas.db"),
                        help="Caminho do banco SQLite")
    parser.add_argument("--saida", default=str(BASE_DIR),
                        help="Pasta para salvar modelos, métricas e atributos")
    args = parser.parse_args()

    print("=" * 72)
    print("ETAPA 1/2 — IMPORTAÇÃO DOS CSVs PARA O BANCO")
    print("=" * 72)
    resumo_banco = criar_banco(args.dados, args.banco)
    print("Resumo da importação:", resumo_banco)

    print("\n" + "=" * 72)
    print("ETAPA 2/2 — PREPARAÇÃO, TREINAMENTO E AVALIAÇÃO")
    print("=" * 72)
    resultados = treinar_modelos(
        pasta_csv=None,
        pasta_saida=args.saida,
        caminho_banco=args.banco,
    )

    print("\nFluxo concluído.")
    print("Banco:", Path(args.banco).resolve())
    print("Melhor modelo:", resultados["melhor_modelo_selecionado"])
    print("Arquivo de modelo:", Path(args.saida).resolve() / "modelo_falencia.joblib")
    print("Comparação:", Path(args.saida).resolve() / "comparacao_modelos.csv")


if __name__ == "__main__":
    main()
