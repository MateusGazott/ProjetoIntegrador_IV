import sys
import os
import requests

BASE = os.getenv("BACKEND_URL", "http://localhost:5000/api/diagnostico")

SAUDAVEL = {"Attr1": 0.12, "Attr2": 0.35, "Attr3": 0.30, "Attr4": 2.1, "Attr6": 0.20,
            "Attr7": 0.14, "Attr8": 1.8, "Attr9": 1.3, "Attr10": 0.65}
EM_RISCO = {"Attr1": -0.08, "Attr2": 0.78, "Attr3": -0.05, "Attr4": 0.85, "Attr6": -0.10,
            "Attr7": -0.04, "Attr8": 0.20, "Attr9": 0.70, "Attr10": 0.15}

def analisar(indicadores, **extra):
    corpo = {"empresaId": "t-" + os.urandom(3).hex(), "nomeEmpresa": "Teste", "horizonteAnos": 1,
             "contextoUsuario": "", "indicadores": indicadores}
    corpo.update(extra)
    return requests.post(f"{BASE}/analisar", json=corpo, timeout=5)

tests = []

def test(name):
    def dec(fn):
        tests.append((name, fn))
        return fn
    return dec

@test("saude retorna status ok")
def _():
    r = requests.get(f"{BASE}/saude", timeout=5)
    assert r.status_code == 200 and r.json()["status"] == "ok"

@test("atributos-modelo lista features, descricoes e medianas")
def _():
    j = requests.get(f"{BASE}/atributos-modelo", timeout=5).json()
    assert len(j["feature_names"]) > 0
    assert set(j["feature_names"]) <= set(j["medians"])

@test("empresa saudavel classificada como Baixo Risco")
def _():
    r = analisar(SAUDAVEL)
    assert r.status_code == 200, r.text
    assert r.json()["classificacaoRisco"] == "Baixo Risco", r.json()["classificacaoRisco"]
    assert r.json()["pontosDeAtencao"] == []

@test("empresa em risco classificada como Alto Risco com pontos de atencao")
def _():
    j = analisar(EM_RISCO).json()
    assert j["classificacaoRisco"] == "Alto Risco", j["classificacaoRisco"]
    assert j["classePrevista"] == 1
    assert len(j["pontosDeAtencao"]) >= 4

@test("probabilidades somam 1")
def _():
    j = analisar(EM_RISCO).json()
    assert abs(j["probabilidadeSaudavel"] + j["probabilidadeInsolvencia"] - 1) < 0.001

@test("dados incompletos sao imputados e percentual reflete")
def _():
    j = analisar({"Attr2": 0.6}).json()
    assert len(j["atributosImputados"]) == len(j["indicadoresProcessados"]) - 1
    assert j["percentualIndicadoresInformados"] < 20

@test("valores nulos sao imputados")
def _():
    j = analisar({"Attr1": None, "Attr2": 0.5}).json()
    assert "Attr1" in j["atributosImputados"]

@test("nomes de atributo sao case-insensitive")
def _():
    j = analisar({"attr2": 0.5}).json()
    assert "Attr2" not in j["atributosImputados"]

@test("atributo desconhecido gera alerta e e ignorado")
def _():
    j = analisar({"Attr2": 0.5, "Foo": 1}).json()
    assert any("Foo" in a for a in j["alertasValidacao"])

@test("valor extremo gera alerta")
def _():
    j = analisar({"Attr1": 99999}).json()
    assert any("extremo" in a for a in j["alertasValidacao"])

@test("valor fora do intervalo numerico retorna 400")
def _():
    r = analisar({"Attr1": 1e300})
    assert r.status_code == 400 and "erro" in r.json()

@test("indicadores duplicados em maiusculas e minusculas retornam 400")
def _():
    r = analisar({"Attr1": 0.1, "attr1": 0.2})
    assert r.status_code == 400 and "mais de uma vez" in r.json()["erro"]

@test("indicadores vazios retornam 400")
def _():
    assert analisar({}).status_code == 400

@test("nenhum atributo conhecido retorna 400")
def _():
    assert analisar({"Foo": 1, "Bar": 2}).status_code == 400

@test("horizonte invalido retorna 400")
def _():
    for h in (0, 6, -1):
        r = analisar({"Attr2": 0.5}, horizonteAnos=h)
        assert r.status_code == 400, h
        assert "horizonteAnos" in " ".join(r.json().get("detalhes", []))

@test("empresaId invalido retorna 400")
def _():
    assert analisar({"Attr2": 0.5}, empresaId="a b/c").status_code == 400

@test("nome muito longo retorna 400")
def _():
    assert analisar({"Attr2": 0.5}, nomeEmpresa="x" * 101).status_code == 400

@test("contexto muito longo retorna 400")
def _():
    assert analisar({"Attr2": 0.5}, contextoUsuario="x" * 1001).status_code == 400

@test("json malformado retorna 400 com corpo padronizado")
def _():
    r = requests.post(f"{BASE}/analisar", data="{nao e json", headers={"Content-Type": "application/json"}, timeout=5)
    assert r.status_code == 400 and "erro" in r.json()

@test("valor nao numerico retorna 400")
def _():
    r = requests.post(f"{BASE}/analisar", json={"indicadores": {"Attr1": "abc"}}, timeout=5)
    assert r.status_code == 400 and "erro" in r.json()

@test("empresaId e nome ausentes sao gerados")
def _():
    r = requests.post(f"{BASE}/analisar", json={"indicadores": {"Attr2": 0.5}}, timeout=5)
    assert r.status_code == 200 and r.json()["empresaId"] and r.json()["nomeEmpresa"] == "Empresa Não Identificada"

@test("chat-consulta devolve resumo com contexto e pontos de atencao")
def _():
    j = analisar(EM_RISCO, empresaId="chat-1", contextoUsuario="queda de faturamento").json()
    c = requests.get(f"{BASE}/chat-consulta", params={"empresaId": j["empresaId"]}, timeout=5).json()
    assert "queda de faturamento" in c["resumoParaPromptLLM"]
    assert "Pontos de Atenção" in c["resumoParaPromptLLM"]
    assert c["diagnosticoCompleto"]["empresaId"] == "chat-1"

@test("chat-consulta sem id devolve o ultimo diagnostico")
def _():
    analisar(SAUDAVEL, empresaId="ultimo-1")
    c = requests.get(f"{BASE}/chat-consulta", timeout=5).json()
    assert c["empresaId"] == "ultimo-1"

@test("chat-consulta de id inexistente retorna 404")
def _():
    assert requests.get(f"{BASE}/chat-consulta", params={"empresaId": "nao-existe"}, timeout=5).status_code == 404

@test("chat-consulta com id invalido retorna 400")
def _():
    assert requests.get(f"{BASE}/chat-consulta", params={"empresaId": "a b"}, timeout=5).status_code == 400

@test("reanalise com mesmo id sobrescreve o diagnostico")
def _():
    analisar(SAUDAVEL, empresaId="mesmo-1")
    analisar(EM_RISCO, empresaId="mesmo-1")
    c = requests.get(f"{BASE}/chat-consulta", params={"empresaId": "mesmo-1"}, timeout=5).json()
    assert c["classificacaoRisco"] == "Alto Risco"

print(f"Executando {len(tests)} testes...")
passou = 0
for name, fn in tests:
    sys.stdout.write(f"- {name} ... ")
    sys.stdout.flush()
    try:
        fn()
        sys.stdout.write("PASSOU\n")
        passou += 1
    except Exception as e:
        sys.stdout.write(f"FALHOU: {e}\n")
    sys.stdout.flush()

print(f"\nResultado Final: {passou}/{len(tests)} testes passaram.")
