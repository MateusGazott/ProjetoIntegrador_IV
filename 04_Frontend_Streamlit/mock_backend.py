"""
Serviço de Backend Preditivo de Risco Financeiro (Python / REST API)
Integrante 2 — Backend e Integração do Modelo

Funcionalidades:
- Servidor HTTP multi-thread (ThreadingHTTPServer) de alta disponibilidade.
- Carrega o modelo treinado (Joblib ou Heurístico) e os metadados (model_metadata.json).
- Pré-processamento com validação de entradas, tratamento de nomes case-insensitive,
  detecção de duplicatas, detecção de valores extremos e imputação por mediana.
- Geração de pontos de atenção contábeis a partir dos indicadores.
- Classificação de risco: Baixo Risco (< 0.35), Médio Risco (0.35..0.60), Alto Risco (>= 0.60).
- Armazenamento em memória do contexto de diagnóstico para o Chat/LLM (/chat-consulta).
- Contrato 100% compatível com a API ASP.NET Core e com o Frontend Streamlit.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# Localizar model_metadata.json
META_CANDIDATOS = [
    os.path.join(HERE, "RiskAnalysisBackend", "ModelsML", "model_metadata.json"),
    os.path.join(HERE, "..", "01_Treinamento_Modelo_Tabular", "model_metadata.json"),
    os.path.join(HERE, "model_metadata.json"),
]
META_PATH = next((p for p in META_CANDIDATOS if os.path.exists(p)), META_CANDIDATOS[0])
with open(META_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

# Ajustar threshold de alto risco se necessário para 0.60
if METADATA.get("threshold_alto_risco", 0.65) > 0.60:
    METADATA["threshold_alto_risco"] = 0.60

# Localizar modelo treinado Joblib
MODEL_CANDIDATOS = [
    os.path.join(HERE, "RiskAnalysisBackend", "ModelsML", "modelo_risco.joblib"),
    os.path.join(HERE, "..", "01_Treinamento_Modelo_Tabular", "modelo_risco.joblib"),
    os.path.join(HERE, "modelo_risco.joblib"),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATOS if os.path.exists(p)), None)

TRAINED_MODEL = None
if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        import joblib
        TRAINED_MODEL = joblib.load(MODEL_PATH)
        print(f"[Backend] Modelo treinado carregado com sucesso de: {MODEL_PATH}")
    except Exception as e:
        print(f"[Backend] Aviso: Falha ao carregar modelo joblib ({e}). Usando modo heurístico.")

# Armazenamento em memória para consulta do chat
DIAGNOSTICOS_STORE: Dict[str, Dict[str, Any]] = {}
ULTIMO_EMPRESA_ID: Optional[str] = None


def validar_empresa_id(empresa_id: str) -> bool:
    """Aceita letras, números, hífen e sublinhado (sem espaços ou barras)."""
    return bool(re.match(r"^[A-Za-z0-9_\-]+$", empresa_id))


def determinar_faixa_risco(prob: float) -> str:
    if prob >= METADATA.get("threshold_alto_risco", 0.60):
        return "Alto Risco"
    if prob >= METADATA.get("threshold_medio_risco", 0.35):
        return "Médio Risco"
    return "Baixo Risco"


def gerar_pontos_atencao(processados: Dict[str, float], imputados: List[str]) -> List[str]:
    pontos = []
    imp = set(imputados)

    if "Attr1" not in imp and processados.get("Attr1", 0) < 0:
        pontos.append("Prejuízo líquido em relação aos ativos totais (Attr1 negativo).")
    if "Attr2" not in imp and processados.get("Attr2", 0) > 0.70:
        pontos.append("Alto grau de endividamento geral: dívidas superam 70% dos ativos totais (Attr2).")
    if "Attr3" not in imp and processados.get("Attr3", 0) < 0:
        pontos.append("Capital de giro negativo: passivo circulante maior que ativo circulante (Attr3).")
    if "Attr4" not in imp and processados.get("Attr4", 0) < 1.0:
        pontos.append("Índice de liquidez corrente abaixo de 1,0 indicando aperto no curto prazo (Attr4).")
    if "Attr6" not in imp and processados.get("Attr6", 0) < 0:
        pontos.append("Lucros retidos negativos indicando histórico de prejuízos acumulados (Attr6).")
    if "Attr7" not in imp and processados.get("Attr7", 0) < 0:
        pontos.append("Resultado operacional (EBIT) negativo frente aos ativos (Attr7).")
    if "Attr8" not in imp and processados.get("Attr8", 0) < 0.5:
        pontos.append("Patrimônio líquido baixo quando comparado ao passivo total (Attr8).")
    if "Attr10" not in imp and processados.get("Attr10", 0) < 0.20:
        pontos.append("Baixa participação de capital próprio na estrutura de capital (Attr10).")

    return pontos


def construir_resumo_prompt(diagnostico: Dict[str, Any], contexto_usuario: Optional[str]) -> str:
    pontos = diagnostico.get("pontosDeAtencao", [])
    texto_pontos = "\n".join(f"  - {p}" for p in pontos) if pontos else "  - Nenhum ponto crítico identificado."
    
    resumo = (
        f"[DIAGNÓSTICO DO MODELO PREDITIVO]\n"
        f"- Empresa: {diagnostico['nomeEmpresa']} (ID: {diagnostico['empresaId']})\n"
        f"- Horizonte de Projeção: {diagnostico['horizonteAnos']} ano(s)\n"
        f"- Classificação de Risco: {diagnostico['classificacaoRisco']}\n"
        f"- Probabilidade de Insolvência/Falência: {diagnostico['probabilidadeInsolvencia'] * 100:.2f}%\n"
        f"- Probabilidade Saudável: {diagnostico['probabilidadeSaudavel'] * 100:.2f}%\n"
        f"- Modo de Execução: {diagnostico['modoExecucao']}\n"
        f"- Indicadores Informados: {diagnostico['percentualIndicadoresInformados']}%\n"
    )
    if contexto_usuario:
        resumo += f"- Contexto Informado pelo Usuário: {contexto_usuario.strip()}\n"
    resumo += f"\n[Pontos de Atenção Contábeis]\n{texto_pontos}\n"
    return resumo


def executar_analise(req: Dict[str, Any]) -> Dict[str, Any]:
    global ULTIMO_EMPRESA_ID

    if not isinstance(req, dict):
        raise ValueError("A requisição deve ser um objeto JSON válido.")

    ind = req.get("indicadores")
    if not ind or not isinstance(ind, dict) or len(ind) == 0:
        raise ValueError("Os dados financeiros (indicadores) da empresa não foram fornecidos.")

    # Validação do horizonte
    h = req.get("horizonteAnos", 1)
    if not isinstance(h, (int, float)) or int(h) < 1 or int(h) > 5:
        err = ValueError("O horizonte de análise deve estar entre 1 e 5 anos.")
        err.detalhes = ["horizonteAnos deve estar entre 1 e 5 anos."]
        raise err

    # Validação do nome
    nome = req.get("nomeEmpresa")
    if nome is not None:
        if not isinstance(nome, str):
            raise ValueError("O nome da empresa deve ser um texto.")
        if len(nome) > 100:
            raise ValueError("O nome da empresa não pode exceder 100 caracteres.")
        nome = nome.strip()
    if not nome:
        nome = "Empresa Não Identificada"

    # Validação do contexto
    ctx = req.get("contextoUsuario")
    if ctx is not None:
        if not isinstance(ctx, str):
            raise ValueError("O contexto do usuário deve ser um texto.")
        if len(ctx) > 1000:
            raise ValueError("O contexto do usuário não pode exceder 1000 caracteres.")
        ctx = ctx.strip()

    # Validação do empresaId
    eid = req.get("empresaId")
    if eid is not None:
        if not isinstance(eid, str):
            raise ValueError("empresaId inválido.")
        eid = eid.strip()
        if not validar_empresa_id(eid):
            raise ValueError("empresaId contém caracteres inválidos. Use apenas letras, números e hífen.")
    if not eid:
        import uuid
        eid = f"emp-{uuid.uuid4().hex[:8]}"

    # Validação de duplicação case-insensitive de indicadores
    chaves_vistas = {}
    for k in ind.keys():
        k_lower = k.lower()
        if k_lower in chaves_vistas:
            raise ValueError(f"O indicador '{chaves_vistas[k_lower]}' foi informado mais de uma vez com variações de maiúsculas/minúsculas.")
        chaves_vistas[k_lower] = k

    # Pré-processamento de atributos
    esperados = METADATA["feature_names"]
    medianas = METADATA.get("medians", {})
    norm_ind = {k.lower(): v for k, v in ind.items()}

    processados: Dict[str, float] = {}
    imputados: List[str] = []
    alertas: List[str] = []

    # Verificar atributos desconhecidos
    esperados_lower = {f.lower(): f for f in esperados}
    for k in ind.keys():
        if k.lower() not in esperados_lower:
            alertas.append(f"O atributo '{k}' não faz parte das variáveis do modelo e foi ignorado.")

    for f in esperados:
        v = norm_ind.get(f.lower())
        if v is None:
            processados[f] = float(medianas.get(f, 0.0))
            imputados.append(f)
        else:
            if not isinstance(v, (int, float)):
                raise ValueError(f"O indicador '{f}' contém um valor não numérico: '{v}'.")
            v_float = float(v)
            if math.isnan(v_float) or math.isinf(v_float) or abs(v_float) > 1e100:
                raise ValueError(f"O indicador '{f}' contém um valor fora do intervalo numérico válido.")
            processados[f] = v_float
            if abs(v_float) > 10000:
                alertas.append(f"O indicador '{f}' apresenta valor extremo ({v_float}). Verifique a escala informada.")

    if len(imputados) == len(esperados):
        raise ValueError("Nenhum dos indicadores enviados corresponde aos atributos esperados pelo modelo.")

    # Predição com o Modelo Treinado ou Heurística
    p_saudavel, p_insolvencia, classe, modo = 0.5, 0.5, 0, "FallbackHeuristico"
    if TRAINED_MODEL is not None:
        try:
            df_in = pd.DataFrame([[processados[f] for f in esperados]], columns=esperados)
            probabilidades = TRAINED_MODEL.predict_proba(df_in)[0]
            p_saudavel = float(probabilidades[0])
            p_insolvencia = float(probabilidades[1])
            classe = 1 if p_insolvencia >= 0.5 else 0
            modo = "RandomForest"
        except Exception as e:
            modo = "FallbackHeuristico"

    if modo == "FallbackHeuristico":
        # Heurística ponderada
        score = (
            0.30
            + (processados["Attr2"] - 0.5) * 0.65
            - processados["Attr1"] * 0.80
            - processados["Attr3"] * 0.40
            + (0.20 if processados["Attr4"] < 1.0 else -0.10)
            - processados["Attr6"] * 0.50
            - processados["Attr7"] * 0.60
            - (0.15 if processados["Attr8"] > 1.0 else -0.10)
        )
        p_insolvencia = min(max(score, 0.02), 0.98)
        p_saudavel = 1.0 - p_insolvencia
        classe = 1 if p_insolvencia >= 0.5 else 0

    total_attrs = len(esperados)
    perc_informado = round((total_attrs - len(imputados)) * 100.0 / total_attrs, 1)
    classificacao = determinar_faixa_risco(p_insolvencia)
    pontos_atencao = gerar_pontos_atencao(processados, imputados)

    diagnostico = {
        "empresaId": eid,
        "nomeEmpresa": nome,
        "horizonteAnos": int(h),
        "dataAnalise": datetime.now(timezone.utc).isoformat(),
        "classePrevista": classe,
        "classificacaoRisco": classificacao,
        "probabilidadeSaudavel": round(p_saudavel, 4),
        "probabilidadeInsolvencia": round(p_insolvencia, 4),
        "modoExecucao": modo,
        "percentualIndicadoresInformados": perc_informado,
        "indicadoresProcessados": processados,
        "atributosImputados": imputados,
        "pontosDeAtencao": pontos_atencao,
        "alertasValidacao": alertas,
    }

    resumo_llm = construir_resumo_prompt(diagnostico, ctx)
    contexto_chat = {
        "empresaId": eid,
        "nomeEmpresa": nome,
        "classificacaoRisco": classificacao,
        "probabilidadeRiscoPercentual": round(p_insolvencia * 100, 2),
        "contextoInformadoPeloUsuario": ctx,
        "resumoParaPromptLLM": resumo_llm,
        "diagnosticoCompleto": diagnostico,
    }

    DIAGNOSTICOS_STORE[eid] = contexto_chat
    ULTIMO_EMPRESA_ID = eid

    return diagnostico


class BackendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status_code: int, data: Any):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.close_connection = True

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/api/diagnostico/saude", "/saude"):
            return self._send_json(200, {
                "status": "ok",
                "modo": "RandomForest" if TRAINED_MODEL else "FallbackHeuristico",
                "ehModeloReal": TRAINED_MODEL is not None,
                "totalAtributos": len(METADATA["feature_names"]),
                "diagnosticosArmazenados": len(DIAGNOSTICOS_STORE)
            })

        if path in ("/api/diagnostico/atributos-modelo", "/atributos-modelo"):
            return self._send_json(200, METADATA)

        if path in ("/api/diagnostico/chat-consulta", "/chat-consulta"):
            query = parse_qs(parsed.query)
            eid_param = query.get("empresaId", [None])[0]
            eid = eid_param or ULTIMO_EMPRESA_ID

            if not eid:
                return self._send_json(404, {"erro": "Nenhum diagnóstico registrado no histórico."})

            if not validar_empresa_id(eid):
                return self._send_json(400, {"erro": "empresaId informado contém caracteres inválidos."})

            if eid not in DIAGNOSTICOS_STORE:
                return self._send_json(404, {"erro": f"Diagnóstico não encontrado para a empresaId: '{eid}'."})

            return self._send_json(200, DIAGNOSTICOS_STORE[eid])

        self._send_json(404, {"erro": f"Rota não encontrada: {parsed.path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path not in ("/api/diagnostico/analisar", "/analisar"):
            return self._send_json(404, {"erro": f"Rota não encontrada: {parsed.path}"})

        content_len = int(self.headers.get("Content-Length", 0))
        corpo_raw = self.rfile.read(content_len) if content_len > 0 else b""

        try:
            req = json.loads(corpo_raw.decode("utf-8"))
        except Exception:
            return self._send_json(400, {"erro": "Corpo da requisição não é um JSON válido."})

        try:
            resultado = executar_analise(req)
            self._send_json(200, resultado)
        except ValueError as e:
            resp = {"erro": str(e)}
            if hasattr(e, "detalhes"):
                resp["detalhes"] = e.detalhes
            self._send_json(400, resp)
        except Exception as e:
            self._send_json(500, {"erro": f"Erro interno do servidor: {str(e)}"})

    def log_message(self, format, *args):
        pass


def run_server(port: int = 5000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, BackendHandler)
    print(f"🚀 Backend Preditivo em execução em http://localhost:{port}/api/diagnostico")
    httpd.serve_forever()


if __name__ == "__main__":
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(porta)
