"""Testes headless da interface. Requer o mock (ou o backend C#) em http://localhost:5000."""
import sys, types
from streamlit.testing.v1 import AppTest

def novo():
    at = AppTest.from_file("app.py", default_timeout=30); at.run(); return at

def botao(at, rotulo):
    return next(b for b in at.button if rotulo in b.label)

def test_fluxo_feliz_risco():
    at = novo(); assert not at.exception
    botao(at, "em risco").click(); at.run()
    botao(at, "Analisar").click(); at.run()
    assert not at.exception and at.session_state.diagnostico["classificacaoRisco"] == "Alto Risco"
    assert len(at.session_state.mensagens) == 1          # explicação inicial
    assert not at.chat_input[0].disabled

def test_fluxo_saudavel_e_virgula_decimal():
    at = novo(); botao(at, "saudável").click(); at.run()
    botao(at, "Analisar").click(); at.run()
    assert at.session_state.diagnostico["classificacaoRisco"] == "Baixo Risco"
    assert at.session_state.diagnostico["indicadoresProcessados"]["Attr1"] == 0.12

def test_valor_invalido():
    at = novo(); at.text_input(key="ind_Attr1").set_value("abc")
    botao(at, "Analisar").click(); at.run()
    assert any("Attr1" in e.value for e in at.error) and at.session_state.diagnostico is None

def test_nan_rejeitado():
    at = novo(); at.text_input(key="ind_Attr2").set_value("nan")
    botao(at, "Analisar").click(); at.run()
    assert any("Attr2" in e.value for e in at.error)

def test_formulario_vazio():
    at = novo(); botao(at, "Analisar").click(); at.run()
    assert any("pelo menos um" in e.value for e in at.error)

def test_valor_extremo_gera_aviso_e_backend_alerta():
    at = novo(); at.text_input(key="ind_Attr1").set_value("99999")
    botao(at, "Analisar").click(); at.run()
    assert at.session_state.diagnostico["alertasValidacao"]
    assert any("extremo" in w.value for w in at.warning)   # alerta do backend exibido no diagnóstico

def test_incompleto_imputa_mediana():
    at = novo(); at.text_input(key="ind_Attr2").set_value("0,6")
    botao(at, "Analisar").click(); at.run()
    assert len(at.session_state.diagnostico["atributosImputados"]) == 9

def test_chat_envia_e_valida():
    at = novo(); botao(at, "em risco").click(); at.run(); botao(at, "Analisar").click(); at.run()
    at.chat_input[0].set_value("Por onde começar?").run()
    assert len(at.session_state.mensagens) == 3
    at.chat_input[0].set_value("x" * 1500).run()
    assert len(at.session_state.mensagens) == 3 and any("longa" in e.value for e in at.error)

def test_chat_bloqueado_antes_da_analise():
    assert novo().chat_input[0].disabled

def test_backend_offline():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["backend_url"] = "http://localhost:59999/api/diagnostico"; at.run()
    assert not at.exception and any("offline" in e.value.lower() or "Não foi possível" in e.value for e in at.error)

def test_falha_da_groq_degrada_para_local():
    mod = types.ModuleType("groq_service")
    def boom(*a, **k): raise TimeoutError("groq fora do ar")
    mod.gerar_orientacao = boom; sys.modules["groq_service"] = mod
    try:
        at = novo(); botao(at, "em risco").click(); at.run(); botao(at, "Analisar").click(); at.run()
        assert not at.exception and at.session_state.mensagens[0]["origem"] == "local_erro"
    finally:
        del sys.modules["groq_service"]

def test_groq_ok_usada_quando_presente():
    mod = types.ModuleType("groq_service"); mod.gerar_orientacao = lambda m, r, h: "RESPOSTA GROQ"
    sys.modules["groq_service"] = mod
    try:
        at = novo(); botao(at, "saudável").click(); at.run(); botao(at, "Analisar").click(); at.run()
        assert at.session_state.mensagens[0] == {"role": "assistant", "content": "RESPOSTA GROQ", "origem": "groq"}
    finally:
        del sys.modules["groq_service"]

def test_aviso_legal_presente():
    assert any("não substitui" in w.value for w in novo().warning)
