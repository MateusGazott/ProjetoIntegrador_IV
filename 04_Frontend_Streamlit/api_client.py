"""
Cliente HTTP do Backend C# (Integrante 2) usado pela interface Streamlit.

Converte falhas de rede / HTTP em exceções com mensagens amigáveis, para que a
interface nunca exiba um traceback para o usuário.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api/diagnostico").rstrip("/")
TIMEOUT_SEGUNDOS = 15


class BackendError(Exception):
    """Erro genérico retornado pelo backend (HTTP 4xx/5xx)."""


class BackendOffline(BackendError):
    """Backend inacessível (conexão recusada, DNS, timeout)."""


def _tratar(resp: requests.Response) -> Dict[str, Any]:
    if resp.ok:
        return resp.json()
    try:
        corpo = resp.json()
        detalhe = corpo.get("erro") or corpo.get("title") or str(corpo)
    except ValueError:
        detalhe = resp.text[:200] or resp.reason
    raise BackendError(f"{detalhe} (HTTP {resp.status_code})")


def _request(metodo: str, caminho: str, base_url: str, **kwargs) -> Dict[str, Any]:
    try:
        resp = requests.request(metodo, f"{base_url.rstrip('/')}{caminho}", timeout=TIMEOUT_SEGUNDOS, **kwargs)
    except requests.exceptions.Timeout as e:
        raise BackendOffline("O backend demorou demais para responder.") from e
    except requests.exceptions.ConnectionError as e:
        raise BackendOffline(f"Não foi possível conectar ao backend em {base_url}.") from e
    except requests.exceptions.RequestException as e:
        raise BackendOffline(f"Falha de comunicação com o backend: {e}") from e
    return _tratar(resp)


def obter_atributos_modelo(base_url: str = BACKEND_BASE_URL) -> Dict[str, Any]:
    """GET /atributos-modelo -> feature_names, feature_descriptions, medians, thresholds."""
    return _request("GET", "/atributos-modelo", base_url)


def analisar_empresa(
    base_url: str,
    empresa_id: str,
    nome_empresa: str,
    horizonte_anos: int,
    contexto_usuario: str,
    indicadores: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """POST /analisar -> RiskDiagnosisResponse."""
    payload = {
        "empresaId": empresa_id,
        "nomeEmpresa": nome_empresa,
        "horizonteAnos": horizonte_anos,
        "contextoUsuario": contexto_usuario,
        "indicadores": indicadores,
    }
    return _request("POST", "/analisar", base_url, json=payload)


def consultar_diagnostico_chat(base_url: str, empresa_id: Optional[str] = None) -> Dict[str, Any]:
    """GET /chat-consulta -> ChatDiagnosticContextResponse (inclui resumoParaPromptLLM)."""
    params = {"empresaId": empresa_id} if empresa_id else {}
    return _request("GET", "/chat-consulta", base_url, params=params)
