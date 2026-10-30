"""
Executor Consolidado de Todos os Testes do Projeto
Executa as suítes de teste de todos os integrantes e consolida os resultados:
- Integrante 2: 26 Testes da API REST do Backend
- Integrante 3: 5 Testes Unitários da Integração com a Groq
- Integrante 4: 13 Testes Headless da Interface Streamlit (AppTest)
- Integrante 5: 12 Testes de Avaliação da IA, Entradas Extremas e Resiliência
"""

import os
import subprocess
import sys
import time

PYTHON_BIN = sys.executable

def executar_comando(nome, cmd, cwd=None):
    print("\n" + "=" * 70)
    print(f"EXECUTANDO: {nome}")
    print("=" * 70)
    inicio = time.time()
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    duracao = time.time() - inicio
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    ok = (res.returncode == 0)
    status_txt = "✅ PASSOU" if ok else "❌ FALHOU"
    print(f"[{status_txt}] {nome} (Duração: {duracao:.2f}s)")
    return ok

def main():
    print("*" * 75)
    print("  SUÍTE CONSOLIDADA DE TESTES DO PROJETO INTEGRADOR")
    print("*" * 75)

    base = os.path.dirname(os.path.abspath(__file__))

    resultados = []

    # 1. Testes do Backend (Integrante 2)
    p2 = os.path.join(base, "02_Backend_Integracao")
    ok2 = executar_comando(
        "Integrante 2 — Testes da API REST do Backend (26 Cenários)",
        [PYTHON_BIN, "run_backend_tests_verbose.py"],
        cwd=p2
    )
    resultados.append(("Integrante 2 — Backend REST API", ok2))

    # 2. Testes da Groq (Integrante 3)
    p3 = os.path.join(base, "03_Integracao_Groq_LLM")
    ok3 = executar_comando(
        "Integrante 3 — Testes Unitários da Integração Groq LLM",
        [PYTHON_BIN, "test_groq.py"],
        cwd=p3
    )
    resultados.append(("Integrante 3 — Módulo Groq LLM", ok3))

    # 3. Testes da Interface Streamlit (Integrante 4)
    p4 = os.path.join(base, "04_Frontend_Streamlit")
    ok4 = executar_comando(
        "Integrante 4 — Testes de Interface Streamlit AppTest (13 Testes)",
        [PYTHON_BIN, "-m", "pytest", "test_interface.py", "-q"],
        cwd=p4
    )
    resultados.append(("Integrante 4 — Interface Streamlit", ok4))

    # 4. Testes de Avaliação da IA (Integrante 5)
    p5 = os.path.join(base, "05_Testes_e_Avaliacao_IA")
    ok5 = executar_comando(
        "Integrante 5 — Testes de Avaliação da IA, Validação e Resiliência",
        [PYTHON_BIN, "executar_testes_integrante5.py"],
        cwd=p5
    )
    resultados.append(("Integrante 5 — Avaliação da IA e Resiliência", ok5))

    print("\n" + "*" * 75)
    print("  RESUMO FINAL DE APROVAÇÃO DAS SUÍTES")
    print("*" * 75)
    for nome, ok in resultados:
        print(f"  {'[✅ APROVADO]' if ok else '[❌ REPROVADO]'} {nome}")

    todos_ok = all(r[1] for r in resultados)
    print("*" * 75)
    if todos_ok:
        print("  🎉 TODOS OS TESTES PASSARAM COM 100% DE SUCESSO!")
    else:
        print("  ⚠️ Houve falhas em algum dos módulos. Verifique as saídas acima.")
    print("*" * 75)
    sys.exit(0 if todos_ok else 1)

if __name__ == "__main__":
    main()
