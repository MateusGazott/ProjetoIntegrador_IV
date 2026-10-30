"""
Camada de ligação entre o chat (Streamlit) e o módulo da Groq (Integrante 3).

CONTRATO esperado do módulo do Integrante 3 — arquivo `groq_service.py` na mesma pasta:

    def gerar_orientacao(mensagem_usuario: str,
                         resumo_diagnostico: str,
                         historico: list[dict]) -> str

    - mensagem_usuario : texto digitado no chat
    - resumo_diagnostico: campo `resumoParaPromptLLM` do endpoint /chat-consulta
    - historico        : [{"role": "user"|"assistant", "content": "..."}, ...]
    - retorna          : texto da resposta (pode levantar exceção em caso de falha)

Se `groq_service` ainda não existir (ou a chamada falhar), a interface continua
funcionando com uma resposta local baseada em regras simples, claramente identificada.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

MAX_CARACTERES_MENSAGEM = 1000


def validar_mensagem(texto: str) -> Tuple[bool, str]:
    texto = (texto or "").strip()
    if not texto:
        return False, "Digite uma mensagem antes de enviar."
    if len(texto) > MAX_CARACTERES_MENSAGEM:
        return False, f"Mensagem muito longa ({len(texto)} caracteres). O limite é {MAX_CARACTERES_MENSAGEM}."
    return True, ""


def _carregar_groq():
    try:
        return importlib.import_module("groq_service")
    except ImportError:
        return None


def groq_disponivel() -> bool:
    return _carregar_groq() is not None


def gerar_resposta(
    mensagem_usuario: str,
    consulta_chat: Dict[str, Any],
    historico: List[Dict[str, str]],
) -> Tuple[str, str]:
    """
    Retorna (texto, origem) onde origem ∈ {"groq", "local", "local_erro"}.
    Nunca levanta exceção: falhas da LLM viram resposta local + origem "local_erro".
    """
    modulo = _carregar_groq()
    if modulo is not None:
        try:
            texto = modulo.gerar_orientacao(
                mensagem_usuario, consulta_chat.get("resumoParaPromptLLM", ""), historico
            )
            if texto and texto.strip():
                return texto.strip(), "groq"
            raise ValueError("resposta vazia da LLM")
        except Exception:  # noqa: BLE001 - qualquer falha da LLM deve degradar com elegância
            return _resposta_local(mensagem_usuario, consulta_chat), "local_erro"
    return _resposta_local(mensagem_usuario, consulta_chat), "local"


# ----------------------------------------------------------------------------------
# Resposta local (fallback) — regras simples sobre os indicadores efetivamente informados
# ----------------------------------------------------------------------------------
_REGRAS = [
    ("Attr1", lambda v: v < 0, "A empresa está tendo **prejuízo** em relação ao seu ativo (Attr1 negativo).",
     "Revise custos fixos e margens dos produtos/serviços menos lucrativos."),
    ("Attr2", lambda v: v > 0.7, "O **endividamento é alto**: mais de 70% do ativo é financiado por dívidas (Attr2).",
     "Renegocie prazos e juros e evite novas dívidas até recuperar o caixa."),
    ("Attr3", lambda v: v < 0, "O **capital de giro é negativo** (Attr3): as dívidas de curto prazo superam os ativos de curto prazo.",
     "Alongue o prazo das dívidas e acelere o recebimento de clientes."),
    ("Attr4", lambda v: v < 1, "A **liquidez corrente está abaixo de 1** (Attr4): pode faltar caixa para pagar contas de curto prazo.",
     "Monte um fluxo de caixa semanal e priorize pagamentos críticos."),
    ("Attr6", lambda v: v < 0, "Os **lucros retidos são negativos** (Attr6): há prejuízos acumulados.",
     "Avalie aporte de capital ou reinvestimento de lucros futuros."),
    ("Attr7", lambda v: v < 0, "O **resultado operacional (EBIT) é negativo** (Attr7).",
     "Reveja preços, volume de vendas e despesas operacionais."),
    ("Attr8", lambda v: v < 0.5, "O **patrimônio líquido é baixo frente às dívidas** (Attr8).",
     "Fortaleça o capital próprio e reduza a dependência de terceiros."),
    ("Attr10", lambda v: v < 0.2, "O **patrimônio líquido representa pouco do ativo** (Attr10).",
     "Reduza passivos ou capitalize a empresa."),
]


def _resposta_local(mensagem: str, consulta: Dict[str, Any]) -> str:
    diag = consulta.get("diagnosticoCompleto", {}) or {}
    classificacao = consulta.get("classificacaoRisco", "—")
    prob = consulta.get("probabilidadeRiscoPercentual", 0)
    nome = consulta.get("nomeEmpresa", "a empresa")
    valores = diag.get("indicadoresProcessados", {}) or {}
    imputados = set(diag.get("atributosImputados", []) or [])

    causas, sugestoes = [], []
    for attr, cond, causa, sugestao in _REGRAS:
        if attr in valores and attr not in imputados and cond(valores[attr]):
            causas.append(causa)
            sugestoes.append(sugestao)

    linhas = [f"**{nome}** foi classificada como **{classificacao}**, com **{prob:.1f}%** de probabilidade de insolvência."]
    if causas:
        linhas.append("\n**Possíveis causas observadas nos indicadores:**")
        linhas += [f"- {c}" for c in causas]
        linhas.append("\n**Sugestões preventivas:**")
        linhas += [f"- {s}" for s in sugestoes]
    else:
        linhas.append("\nNenhum dos indicadores informados cruzou os limites de alerta simples. "
                      "Continue monitorando liquidez, endividamento e resultado operacional.")
    if imputados:
        linhas.append(f"\n_Atenção: {len(imputados)} indicador(es) não informado(s) foi(ram) preenchido(s) com a mediana "
                      "do treinamento, o que reduz a precisão do diagnóstico._")
    linhas.append("\n_Resposta gerada localmente por regras simples (assistente de IA indisponível)._")
    return "\n".join(linhas)
