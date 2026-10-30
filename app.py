"""
Interface Web (Streamlit) — Integrante 4
Formulário de Indicadores Contábeis + Diagnóstico Preditivo + Chat Textual com IA (Groq)

Execução: streamlit run app.py
"""

from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import api_client as api
import llm_adapter as llm

# Configuração da Página
st.set_page_config(
    page_title="FinGuard • Diagnóstico de Risco Financeiro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Moderna e Responsiva
CUSTOM_CSS = """
<style>
    /* Tipografia e espaçamento geral */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* Badges de Risco com Gradientes */
    .badge-baixo {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.25);
        border: 1px solid #34d399;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .badge-medio {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 6px 16px rgba(245, 158, 11, 0.25);
        border: 1px solid #fbbf24;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .badge-alto {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.3);
        border: 1px solid #f87171;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }

    /* Cartões de Métricas */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-card-title {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Sugestões rápidas de chat */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 16px 0;
    }
    .chip {
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .chip:hover {
        background: #e2e8f0;
        color: #0f172a;
    }

    /* Rodapé e Disclaimer */
    .legal-footer {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 28px;
        line-height: 1.45;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

AVISO = (
    "⚠️ **Este sistema é uma ferramenta de apoio e não substitui uma consultoria contábil ou financeira profissional.** "
    "O resultado é uma estimativa estatística baseada em indicadores informados."
)

EXEMPLOS = {
    "saudavel": {
        "Attr1": "0,12", "Attr2": "0,35", "Attr3": "0,30", "Attr4": "2,1",
        "Attr6": "0,20", "Attr7": "0,14", "Attr8": "1,8", "Attr9": "1,3", "Attr10": "0,65"
    },
    "risco": {
        "Attr1": "-0,08", "Attr2": "0,78", "Attr3": "-0,05", "Attr4": "0,85",
        "Attr6": "-0,10", "Attr7": "-0,04", "Attr8": "0,20", "Attr9": "0,70", "Attr10": "0,15"
    },
}

# ------------------------------------------------------------------ Estado da Sessão
ss = st.session_state
ss.setdefault("backend_url", api.BACKEND_BASE_URL)
ss.setdefault("metadata", None)
ss.setdefault("diagnostico", None)
ss.setdefault("consulta_chat", None)
ss.setdefault("mensagens", [])
ss.setdefault("sugestao_selecionada", None)


# ------------------------------------------------------------------ Utilidades
def parse_decimal(texto: str) -> Tuple[Optional[float], Optional[str]]:
    """Converte '0,25' / '0.25' / '1.234,5' em float. Vazio -> (None, None)."""
    t = (texto or "").strip().replace(" ", "")
    if not t:
        return None, None
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return None, "não é um número válido"
    if math.isnan(v) or math.isinf(v):
        return None, "valor inválido (NaN/infinito)"
    return v, None


def carregar_metadata(forcar: bool = False) -> Optional[dict]:
    if ss.metadata is None or forcar:
        try:
            ss.metadata = api.obter_atributos_modelo(ss.backend_url)
        except api.BackendError as e:
            ss.metadata = None
            ss.erro_metadata = str(e)
    return ss.metadata


def aplicar_exemplo(chave: str):
    for k in list(ss.keys()):
        if k.startswith("ind_"):
            ss[k] = ""
    for attr, valor in EXEMPLOS[chave].items():
        ss[f"ind_{attr}"] = valor


def badge_risco_html(classificacao: str) -> str:
    if classificacao == "Baixo Risco":
        return "<div class='badge-baixo'>🟢 Baixo Risco de Insolvência</div>"
    if classificacao == "Médio Risco":
        return "<div class='badge-medio'>🟡 Médio Risco (Atenção Operacional)</div>"
    return "<div class='badge-alto'>🔴 Alto Risco de Insolvência</div>"


def executar_analise(nome: str, horizonte: int, contexto: str, indicadores: Dict[str, Optional[float]]) -> bool:
    empresa_id = f"emp-{uuid.uuid4().hex[:8]}"
    with st.status("📊 Analisando indicadores financeiros...", expanded=True) as status:
        try:
            st.write("📡 Enviando indicadores ao modelo preditivo...")
            diag = api.analisar_empresa(ss.backend_url, empresa_id, nome, horizonte, contexto, indicadores)
            
            st.write("🧾 Consolidando contexto analítico e pontos de atenção...")
            consulta = api.consultar_diagnostico_chat(ss.backend_url, diag["empresaId"])
            
            ss.diagnostico, ss.consulta_chat, ss.mensagens = diag, consulta, []
            
            st.write("🤖 Gerando síntese executiva com o assistente inteligente...")
            texto, origem = llm.gerar_resposta(
                "Explique o diagnóstico em linguagem simples, cite as possíveis causas do risco e dê sugestões preventivas.",
                consulta,
                []
            )
            ss.mensagens.append({"role": "assistant", "content": texto, "origem": origem})
            
            status.update(label="Diagnóstico concluído com sucesso!", state="complete", expanded=False)
            return True
        except api.BackendOffline as e:
            status.update(label="Falha de conexão com o backend", state="error")
            st.error(f"🔌 {e} Verifique se o backend está em execução na porta correta.")
            return False
        except api.BackendError as e:
            status.update(label="Erro no processamento dos dados", state="error")
            st.error(f"❌ O backend recusou a requisição: {e}")
            return False


# ------------------------------------------------------------------ Barra Lateral
with st.sidebar:
    st.markdown("### ⚙️ Configurações do Sistema")
    ss.backend_url = st.text_input("URL do Backend (REST API)", value=ss.backend_url)
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("🔄 Reconectar", use_container_width=True):
        ss.metadata = None
    if col_btn2.button("🧹 Limpar Chat", use_container_width=True):
        if ss.consulta_chat:
            ss.mensagens = ss.mensagens[:1]
    
    meta = carregar_metadata()
    if meta:
        st.success(f"Backend Conectado ({len(meta['feature_names'])} indicadores contábeis)")
    else:
        st.error("Backend Offline")
        if ss.get("erro_metadata"):
            st.caption(f"Detalhe: {ss.erro_metadata}")

    groq_ok = llm.groq_disponivel()
    st.info(f"Assistente IA: {'Groq Cloud (Llama 3) ✅' if groq_ok else 'Modo Local Heurístico ℹ️'}")
    
    st.divider()
    st.markdown("""
    **Sobre o Projeto:**
    - Modelo: Random Forest & ONNX
    - Base: Polish Companies Bankruptcy
    - Módulos: ML, Backend, Groq LLM, Streamlit
    """)
    st.caption("Projeto Integrador IV — Diagnóstico de Risco Empresarial")


# ------------------------------------------------------------------ Cabeçalho Hero
st.markdown("""
<div class="hero-container">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <h1 class="hero-title">📊 FinGuard Analytics</h1>
        <span class="hero-badge">MVP Integrado • Machine Learning & IA</span>
    </div>
    <div class="hero-subtitle">
        Diagnóstico preditivo precoce de insolvência financeira para pequenas e médias empresas,
        com análise de indicadores contábeis e orientações práticas geradas por inteligência artificial.
    </div>
</div>
""", unsafe_allow_html=True)

# Alerta obrigatório
st.warning(AVISO)

if not meta:
    st.error("Não foi possível carregar os atributos do modelo. Inicie o backend e clique em **Reconectar** na barra lateral.")
    st.stop()


# ------------------------------------------------------------------ Formulário Financeiro
with st.expander("📋 Formulário de Indicadores Contábeis da Empresa", expanded=ss.diagnostico is None):
    st.markdown("Selecione um exemplo rápido para testar ou preencha manualmente os indicadores contábeis:")
    col_ex1, col_ex2 = st.columns(2)
    col_ex1.button("✨ Preencher exemplo: empresa saudável", on_click=aplicar_exemplo, args=("saudavel",), use_container_width=True)
    col_ex2.button("⚠️ Preencher exemplo: empresa em risco", on_click=aplicar_exemplo, args=("risco",), use_container_width=True)

    with st.form("form_dados"):
        col_n, col_h = st.columns([3, 1])
        nome = col_n.text_input("Razão Social / Nome da Empresa", value="Minha Empresa LTDA", max_chars=100)
        horizonte = col_h.selectbox("Horizonte de Projeção", [1, 2, 3, 4, 5], index=0, help="Anos à frente para avaliar a probabilidade de risco.")
        
        contexto = st.text_area(
            "Contexto Operacional e de Mercado (Opcional)",
            max_chars=500,
            height=75,
            placeholder="Ex.: Sofremos queda de 25% no faturamento nos últimos 6 meses, com aumento nas despesas financeiras e estoque acumulado."
        )

        st.markdown("---")
        st.markdown("**Indicadores Financeiros:** Aceita vírgula ou ponto (ex: `0,25` ou `0.25`). Campos em branco serão preenchidos pela mediana do setor.")

        cols = st.columns(2)
        for i, attr in enumerate(meta["feature_names"]):
            desc = meta["feature_descriptions"].get(attr, "")
            mediana = meta["medians"].get(attr)
            cols[i % 2].text_input(
                f"{attr} — {desc}",
                key=f"ind_{attr}",
                placeholder=f"Mediana do setor: {mediana}",
                help=desc
            )

        enviado = st.form_submit_button("🔍 Analisar risco", type="primary", use_container_width=True)

    if enviado:
        erros: List[str] = []
        indicadores: Dict[str, Optional[float]] = {}
        for attr in meta["feature_names"]:
            valor, erro = parse_decimal(ss.get(f"ind_{attr}", ""))
            if erro:
                erros.append(f"**{attr}**: {erro}.")
            indicadores[attr] = valor
        
        if not nome.strip():
            erros.append("**Nome da empresa** é obrigatório.")
        if not erros and all(v is None for v in indicadores.values()):
            erros.append("Informe pelo menos um indicador financeiro.")
        
        if erros:
            st.error("Corrija os campos abaixo para prosseguir:\n\n" + "\n".join(f"- {e}" for e in erros))
        else:
            if executar_analise(nome.strip(), int(horizonte), contexto.strip(), indicadores):
                st.rerun()


# ------------------------------------------------------------------ Painel de Diagnóstico
diag = ss.diagnostico
if diag:
    st.markdown(f"### 📈 Diagnóstico Financeiro — **{diag['nomeEmpresa']}**")
    
    col_card1, col_card2, col_card3 = st.columns([1.5, 1, 1])
    with col_card1:
        st.markdown(badge_risco_html(diag["classificacaoRisco"]), unsafe_allow_html=True)
    
    p_ins = diag["probabilidadeInsolvencia"]
    p_sau = diag["probabilidadeSaudavel"]
    
    with col_card2:
        st.metric(label="Probabilidade de Insolvência", value=f"{p_ins * 100:.1f}%")
    with col_card3:
        st.metric(label="Probabilidade de Estabilidade", value=f"{p_sau * 100:.1f}%")

    st.progress(min(max(p_ins, 0.0), 1.0), text=f"Termômetro de Risco: {p_ins * 100:.1f}% de chance estimada de insolvência")

    classe_txt = "Alerta de Insolvência" if diag["classePrevista"] == 1 else "Saudável / Baixo Risco"
    modo_txt = "Random Forest (Modelo Treinado)" if "RandomForest" in diag.get("modoExecucao", "") else diag.get("modoExecucao", "Heurístico")
    
    st.caption(
        f"**Situação:** {classe_txt} · **Horizonte:** {diag['horizonteAnos']} ano(s) · "
        f"**Motor Preditivo:** {modo_txt} · **Indicadores Informados:** {diag.get('percentualIndicadoresInformados', 100)}%"
    )

    # Exibição de alertas e avisos da validação
    for alerta in diag.get("alertasValidacao", []):
        st.warning(alerta)

    if diag.get("atributosImputados"):
        st.info(
            f"ℹ️ {len(diag['atributosImputados'])} indicador(es) não preenchido(s) receberam a mediana estatística da base histórica: "
            f"`{', '.join(diag['atributosImputados'])}`."
        )

    # Pontos de Atenção Contábeis
    if diag.get("pontosDeAtencao"):
        st.markdown("#### 🚨 Pontos de Atenção Críticos")
        for p in diag["pontosDeAtencao"]:
            st.markdown(f"- ⚠️ {p}")

    # Tabela detalhada de indicadores com comparativo
    with st.expander("📊 Tabela Comparativa de Indicadores vs Mediana Histórica"):
        imput = set(diag.get("atributosImputados", []))
        tabela_dados = []
        for k, v in diag["indicadoresProcessados"].items():
            med_ref = meta["medians"].get(k, 0.0)
            origem_txt = "Mediana (Imputado)" if k in imput else "Informado pelo Usuário"
            tabela_dados.append({
                "Atributo": k,
                "Descrição Contábil": meta["feature_descriptions"].get(k, ""),
                "Valor Utilizado": round(v, 4),
                "Mediana de Referência": round(med_ref, 4),
                "Origem": origem_txt
            })
        st.dataframe(pd.DataFrame(tabela_dados), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ Chat com IA
st.markdown("---")
st.markdown("### 💬 Chat com o Consultor IA (FinGuard)")

if not diag:
    st.info("💡 Realize a primeira análise financeira acima para habilitar o chat interativo com o assistente inteligente.")

ORIGENS = {
    "local": "💡 Resposta gerada por regras locais especializadas.",
    "local_erro": "⚠️ O assistente em nuvem está indisponível; exibindo resposta analítica local por regras contábeis.",
    "groq": "🤖 Resposta gerada via Groq Cloud (Llama 3)."
}

for m in ss.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("origem") in ORIGENS:
            st.caption(ORIGENS[m["origem"]])

prompt = st.chat_input("Pergunte sobre o diagnóstico (ex.: por onde começar a reduzir o risco?)", disabled=not diag)

if prompt:
    ok, msg_erro = llm.validar_mensagem(prompt)
    if not ok:
        st.error(msg_erro)
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        historico = [{"role": m["role"], "content": m["content"]} for m in ss.mensagens]
        with st.chat_message("assistant"):
            with st.spinner("O consultor inteligente está elaborando as orientações..."):
                texto, origem = llm.gerar_resposta(prompt.strip(), ss.consulta_chat, historico)
            st.markdown(texto)
            if origem in ORIGENS:
                st.caption(ORIGENS[origem])
        
        ss.mensagens.append({"role": "user", "content": prompt.strip()})
        ss.mensagens.append({"role": "assistant", "content": texto, "origem": origem})

# Rodapé com aviso legal
st.markdown(f"<div class='legal-footer'>{AVISO}</div>", unsafe_allow_html=True)
