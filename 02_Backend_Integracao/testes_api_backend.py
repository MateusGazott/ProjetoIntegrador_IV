import os
import sys

import requests

BASE = os.getenv("BACKEND_URL", "http://localhost:5000/api/diagnostico")

SAUDAVEL = {"Attr1": 0.12, "Attr2": 0.35, "Attr3": 0.30, "Attr4": 2.1, "Attr6": 0.20,
            "Attr7": 0.14, "Attr8": 1.8, "Attr9": 1.3, "Attr10": 0.65}
EM_RISCO = {"Attr1": -0.08, "Attr2": 0.78, "Attr3": -0.05, "Attr4": 0.85, "Attr6": -0.10,
            "Attr7": -0.04, "Attr8": 0.20, "Attr9": 0.70, "Attr10": 0.15}

resultados = []


def caso(nome):
    def decorador(funcao):
        try:
            funcao()
            resultados.append((nome, True, ""))
        except AssertionError as erro:
            resultados.append((nome, False, str(erro) or "assert falhou"))
        except Exception as erro:
            resultados.append((nome, False, f"{type(erro).__name__}: {erro}"))
        return funcao
    return decorador


def analisar(indicadores, **extra):
    corpo = {"empresaId": "t-" + os.urandom(3).hex(), "nomeEmpresa": "Teste", "horizonteAnos": 1,
             "contextoUsuario": "", "indicadores": indicadores}
    corpo.update(extra)
    return requests.post(f"{BASE}/analisar", json=corpo, timeout=10)


@caso("saude retorna status ok")
def _():
    r = requests.get(f"{BASE}/saude", timeout=10)
    assert r.status_code == 200 and r.json()["status"] == "ok"


@caso("atributos-modelo lista features, descricoes e medianas")
def _():
    j = requests.get(f"{BASE}/atributos-modelo", timeout=10).json()
    assert len(j["feature_names"]) > 0
    assert set(j["feature_names"]) <= set(j["medians"])


@caso("empresa saudavel classificada como Baixo Risco")
def _():
    r = analisar(SAUDAVEL)
    assert r.status_code == 200, r.text
    assert r.json()["classificacaoRisco"] == "Baixo Risco", r.json()["classificacaoRisco"]
    assert r.json()["pontosDeAtencao"] == []


@caso("empresa em risco classificada como Alto Risco com pontos de atencao")
def _():
    j = analisar(EM_RISCO).json()
    assert j["classificacaoRisco"] == "Alto Risco", j["classificacaoRisco"]
    assert j["classePrevista"] == 1
    assert len(j["pontosDeAtencao"]) >= 4


@caso("probabilidades somam 1")
def _():
    j = analisar(EM_RISCO).json()
    assert abs(j["probabilidadeSaudavel"] + j["probabilidadeInsolvencia"] - 1) < 0.001


@caso("dados incompletos sao imputados e percentual reflete")
def _():
    j = analisar({"Attr2": 0.6}).json()
    assert len(j["atributosImputados"]) == len(j["indicadoresProcessados"]) - 1
    assert j["percentualIndicadoresInformados"] < 20


@caso("valores nulos sao imputados")
def _():
    j = analisar({"Attr1": None, "Attr2": 0.5}).json()
    assert "Attr1" in j["atributosImputados"]


@caso("nomes de atributo sao case-insensitive")
def _():
    j = analisar({"attr2": 0.5}).json()
    assert "Attr2" not in j["atributosImputados"]


@caso("atributo desconhecido gera alerta e e ignorado")
def _():
    j = analisar({"Attr2": 0.5, "Foo": 1}).json()
    assert any("Foo" in a for a in j["alertasValidacao"])


@caso("valor extremo gera alerta")
def _():
    j = analisar({"Attr1": 99999}).json()
    assert any("extremo" in a for a in j["alertasValidacao"])


@caso("valor fora do intervalo numerico retorna 400")
def _():
    r = analisar({"Attr1": 1e300})
    assert r.status_code == 400 and "erro" in r.json()


@caso("indicadores duplicados em maiusculas e minusculas retornam 400")
def _():
    r = analisar({"Attr1": 0.1, "attr1": 0.2})
    assert r.status_code == 400 and "mais de uma vez" in r.json()["erro"]


@caso("indicadores vazios retornam 400")
def _():
    assert analisar({}).status_code == 400


@caso("nenhum atributo conhecido retorna 400")
def _():
    assert analisar({"Foo": 1, "Bar": 2}).status_code == 400


@caso("horizonte invalido retorna 400")
def _():
    for h in (0, 6, -1):
        r = analisar({"Attr2": 0.5}, horizonteAnos=h)
        assert r.status_code == 400, h
        assert "horizonteAnos" in " ".join(r.json().get("detalhes", []))


@caso("empresaId invalido retorna 400")
def _():
    assert analisar({"Attr2": 0.5}, empresaId="a b/c").status_code == 400


@caso("nome muito longo retorna 400")
def _():
    assert analisar({"Attr2": 0.5}, nomeEmpresa="x" * 101).status_code == 400


@caso("contexto muito longo retorna 400")
def _():
    assert analisar({"Attr2": 0.5}, contextoUsuario="x" * 1001).status_code == 400


@caso("json malformado retorna 400 com corpo padronizado")
def _():
    r = requests.post(f"{BASE}/analisar", data="{nao e json", headers={"Content-Type": "application/json"}, timeout=10)
    assert r.status_code == 400 and "erro" in r.json()


@caso("valor nao numerico retorna 400")
def _():
    r = requests.post(f"{BASE}/analisar", json={"indicadores": {"Attr1": "abc"}}, timeout=10)
    assert r.status_code == 400 and "erro" in r.json()


@caso("empresaId e nome ausentes sao gerados")
def _():
    r = requests.post(f"{BASE}/analisar", json={"indicadores": {"Attr2": 0.5}}, timeout=10)
    assert r.status_code == 200 and r.json()["empresaId"] and r.json()["nomeEmpresa"] == "Empresa Não Identificada"


@caso("chat-consulta devolve resumo com contexto e pontos de atencao")
def _():
    j = analisar(EM_RISCO, empresaId="chat-1", contextoUsuario="queda de faturamento").json()
    c = requests.get(f"{BASE}/chat-consulta", params={"empresaId": j["empresaId"]}, timeout=10).json()
    assert "queda de faturamento" in c["resumoParaPromptLLM"]
    assert "Pontos de Atenção" in c["resumoParaPromptLLM"]
    assert c["diagnosticoCompleto"]["empresaId"] == "chat-1"


@caso("chat-consulta sem id devolve o ultimo diagnostico")
def _():
    analisar(SAUDAVEL, empresaId="ultimo-1")
    c = requests.get(f"{BASE}/chat-consulta", timeout=10).json()
    assert c["empresaId"] == "ultimo-1"


@caso("chat-consulta de id inexistente retorna 404")
def _():
    assert requests.get(f"{BASE}/chat-consulta", params={"empresaId": "nao-existe"}, timeout=10).status_code == 404


@caso("chat-consulta com id invalido retorna 400")
def _():
    assert requests.get(f"{BASE}/chat-consulta", params={"empresaId": "a b"}, timeout=10).status_code == 400


@caso("reanalise com mesmo id sobrescreve o diagnostico")
def _():
    analisar(SAUDAVEL, empresaId="mesmo-1")
    analisar(EM_RISCO, empresaId="mesmo-1")
    c = requests.get(f"{BASE}/chat-consulta", params={"empresaId": "mesmo-1"}, timeout=10).json()
    assert c["classificacaoRisco"] == "Alto Risco"


falhas = [r for r in resultados if not r[1]]
for nome, ok, detalhe in resultados:
    print(("PASSOU  " if ok else "FALHOU  ") + nome + ("" if ok else f"  -> {detalhe}"))
print(f"\n{len(resultados) - len(falhas)}/{len(resultados)} testes passaram")
sys.exit(1 if falhas else 0)
