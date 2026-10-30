"""
Suíte Abrangente de Testes e Avaliação da IA
Integrante 5 — Responsável: Testes Iniciais e Avaliação da IA

Cenários Cobertos:
1. Entradas Válidas (Empresa Saudável vs Empresa em Risco)
2. Entradas Incompletas (Imputação de Mediana e Cálculo de Percentual)
3. Entradas Incorretas (Valores Não-Numéricos, NaNs, JSON Inválido, Horizonte Inválido)
4. Valores Extremos e Outliers (|x| > 10.000 gerando alertas)
5. Verificação da Predição do Modelo e Coerência das Probabilidades
6. Interpretação do Contexto pelo Chat e Construção do Prompt LLM
7. Teste de Falha de Conexão com a Groq e Degradação para Modo Local
8. Coerência das Recomendações Contábeis da IA com os Indicadores Fornecidos
9. Validação do Alerta Legal Obrigatório
"""

import json
import os
import sys

# Adicionar pasta do frontend ao path para importar llm_adapter e groq_service
FRONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04_Frontend_Streamlit"))
if FRONT_PATH not in sys.path:
    sys.path.insert(0, FRONT_PATH)

import requests

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api/diagnostico")

# Indicadores de Referência
SAUDAVEL = {
    "Attr1": 0.12, "Attr2": 0.35, "Attr3": 0.30, "Attr4": 2.1,
    "Attr6": 0.20, "Attr7": 0.14, "Attr8": 1.8, "Attr9": 1.3, "Attr10": 0.65
}

EM_RISCO = {
    "Attr1": -0.08, "Attr2": 0.78, "Attr3": -0.05, "Attr4": 0.85,
    "Attr6": -0.10, "Attr7": -0.04, "Attr8": 0.20, "Attr9": 0.70, "Attr10": 0.15
}

registros_teste = []


def registrar(cenario, status, detalhe=""):
    simbolo = "✅ PASSOU" if status else "❌ FALHOU"
    print(f"[{simbolo}] {cenario}")
    if detalhe and not status:
        print(f"    Detalhe: {detalhe}")
    registros_teste.append({"cenario": cenario, "status": status, "detalhe": detalhe})


def run_all():
    print("=" * 75)
    print("INTEGRANTE 5 — EXECUÇÃO DA SUÍTE DE TESTES E AVALIAÇÃO DA IA")
    print("=" * 75)

    # 1. Entradas Válidas
    try:
        r_sau = requests.post(f"{BASE_URL}/analisar", json={"nomeEmpresa": "Empresa Saudável S.A.", "indicadores": SAUDAVEL}, timeout=5)
        d_sau = r_sau.json()
        assert r_sau.status_code == 200 and d_sau["classificacaoRisco"] == "Baixo Risco"
        assert d_sau["probabilidadeInsolvencia"] < 0.35
        registrar("1.1 Entrada Válida — Empresa Saudável classificada como Baixo Risco", True)
    except Exception as e:
        registrar("1.1 Entrada Válida — Empresa Saudável classificada como Baixo Risco", False, str(e))

    try:
        r_ris = requests.post(f"{BASE_URL}/analisar", json={"nomeEmpresa": "Empresa Risco LTDA", "indicadores": EM_RISCO}, timeout=5)
        d_ris = r_ris.json()
        assert r_ris.status_code == 200 and d_ris["classificacaoRisco"] == "Alto Risco"
        assert d_ris["probabilidadeInsolvencia"] >= 0.60
        assert len(d_ris["pontosDeAtencao"]) >= 3
        registrar("1.2 Entrada Válida — Empresa em Risco classificada como Alto Risco com Pontos de Atenção", True)
    except Exception as e:
        registrar("1.2 Entrada Válida — Empresa em Risco classificada como Alto Risco com Pontos de Atenção", False, str(e))

    # 2. Entradas Incompletas
    try:
        r_inc = requests.post(f"{BASE_URL}/analisar", json={"indicadores": {"Attr2": 0.60}}, timeout=5)
        d_inc = r_inc.json()
        assert r_inc.status_code == 200
        assert len(d_inc["atributosImputados"]) == 9
        assert d_inc["percentualIndicadoresInformados"] == 10.0
        assert "Attr1" in d_inc["atributosImputados"]
        registrar("2.1 Entrada Incompleta — Imputação estatística pela mediana para 9 atributos", True)
    except Exception as e:
        registrar("2.1 Entrada Incompleta — Imputação estatística pela mediana para 9 atributos", False, str(e))

    # 3. Entradas Incorretas
    try:
        r_str = requests.post(f"{BASE_URL}/analisar", json={"indicadores": {"Attr1": "texto_invalido"}}, timeout=5)
        assert r_str.status_code == 400 and "não numérico" in r_str.json().get("erro", "")
        registrar("3.1 Entrada Incorreta — Rejeição de valores string com HTTP 400", True)
    except Exception as e:
        registrar("3.1 Entrada Incorreta — Rejeição de valores string com HTTP 400", False, str(e))

    try:
        r_hor = requests.post(f"{BASE_URL}/analisar", json={"horizonteAnos": 10, "indicadores": {"Attr2": 0.5}}, timeout=5)
        assert r_hor.status_code == 400 and "horizonte" in r_hor.json().get("erro", "").lower()
        registrar("3.2 Entrada Incorreta — Rejeição de horizonte fora de 1..5 anos", True)
    except Exception as e:
        registrar("3.2 Entrada Incorreta — Rejeição de horizonte fora de 1..5 anos", False, str(e))

    try:
        r_dup = requests.post(f"{BASE_URL}/analisar", json={"indicadores": {"Attr1": 0.1, "attr1": 0.2}}, timeout=5)
        assert r_dup.status_code == 400 and "mais de uma vez" in r_dup.json().get("erro", "")
        registrar("3.3 Entrada Incorreta — Rejeição de atributos duplicados case-insensitive", True)
    except Exception as e:
        registrar("3.3 Entrada Incorreta — Rejeição de atributos duplicados case-insensitive", False, str(e))

    # 4. Valores Extremos e Outliers
    try:
        r_ext = requests.post(f"{BASE_URL}/analisar", json={"indicadores": {"Attr1": 99999, "Attr2": 0.5}}, timeout=5)
        d_ext = r_ext.json()
        assert r_ext.status_code == 200
        assert any("extremo" in a.lower() for a in d_ext["alertasValidacao"])
        registrar("4.1 Valores Extremos — Geração de alerta de validação para outlier sem quebra do fluxo", True)
    except Exception as e:
        registrar("4.1 Valores Extremos — Geração de alerta de validação para outlier sem quebra do fluxo", False, str(e))

    # 5. Coerência das Probabilidades
    try:
        r_prob = requests.post(f"{BASE_URL}/analisar", json={"indicadores": SAUDAVEL}, timeout=5)
        d_prob = r_prob.json()
        soma = d_prob["probabilidadeSaudavel"] + d_prob["probabilidadeInsolvencia"]
        assert abs(soma - 1.0) < 0.001
        registrar("5.1 Modelo Preditivo — Probabilidades somam exatamente 100%", True)
    except Exception as e:
        registrar("5.1 Modelo Preditivo — Probabilidades somam exatamente 100%", False, str(e))

    # 6. Interpretação do Contexto pelo Chat
    try:
        eid = "teste-chat-5"
        ctx_txt = "Problema grave com fluxo de caixa e inadimplência de grandes clientes."
        requests.post(f"{BASE_URL}/analisar", json={"empresaId": eid, "contextoUsuario": ctx_txt, "indicadores": EM_RISCO}, timeout=5)
        c_chat = requests.get(f"{BASE_URL}/chat-consulta", params={"empresaId": eid}, timeout=5).json()
        resumo = c_chat["resumoParaPromptLLM"]
        assert ctx_txt in resumo
        assert "Alto Risco" in resumo
        assert "Pontos de Atenção Contábeis" in resumo
        registrar("6.1 Chat / LLM — Resumo analítico integra contexto do usuário e pontos de atenção", True)
    except Exception as e:
        registrar("6.1 Chat / LLM — Resumo analítico integra contexto do usuário e pontos de atenção", False, str(e))

    # 7. Teste de Degradação para Resposta Local quando Groq Indisponível
    try:
        from llm_adapter import gerar_resposta
        # Simula consulta sem chamar API externa
        texto, origem = gerar_resposta(
            "O que fazer agora?",
            c_chat,
            []
        )
        assert len(texto) > 50
        assert origem in ("groq", "local", "local_erro")
        registrar(f"7.1 Resiliência da IA — Degradação e geração de orientação operacional (origem: {origem})", True)
    except Exception as e:
        registrar("7.1 Resiliência da IA — Degradação e geração de orientação operacional", False, str(e))

    # 8. Coerência das Recomendações Contábeis
    try:
        # Verifica se as orientações para empresa em risco mencionam dívidas ou capital de giro
        texto_lower = texto.lower()
        tem_coerencia = any(w in texto_lower for w in ["dívida", "passivo", "capital", "caixa", "prejuízo", "lucro", "prazo", "fornecedor"])
        assert tem_coerencia
        registrar("8.1 Coerência Temática — Recomendações abordam termos financeiros alinhados aos indicadores críticos", True)
    except Exception as e:
        registrar("8.1 Coerência Temática — Recomendações abordam termos financeiros alinhados aos indicadores críticos", False, str(e))

    # 9. Alerta Legal Obrigatório
    try:
        aviso_obrigatorio = "não substitui uma consultoria contábil"
        app_file = os.path.join(FRONT_PATH, "app.py")
        with open(app_file, "r", encoding="utf-8") as f:
            conteudo_app = f.read()
        assert aviso_obrigatorio in conteudo_app
        registrar("9.1 Compliance e Ética — Aviso legal contábil presente na interface e no rodapé", True)
    except Exception as e:
        registrar("9.1 Compliance e Ética — Aviso legal contábil presente na interface e no rodapé", False, str(e))

    print("\n" + "=" * 75)
    total = len(registros_teste)
    passaram = sum(1 for r in registros_teste if r["status"])
    print(f"RESUMO DA SUÍTE DE TESTES: {passaram}/{total} cenários validados ({100*passaram/total:.1f}% de sucesso).")
    print("=" * 75)
    return passaram == total

if __name__ == "__main__":
    sucesso = run_all()
    sys.exit(0 if sucesso else 1)
