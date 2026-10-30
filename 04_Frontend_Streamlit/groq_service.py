"""
Módulo de Integração com a API da Groq (LLM)
Integrante 3 — Responsável pela Comunicação com a Groq e Engenharia de Prompts

Funcionalidades:
- Conexão segura com a API da Groq utilizando o modelo Llama 3 (llama-3.3-70b-versatile ou llama-3.1-8b-instant).
- Proteção da chave de API via variável de ambiente GROQ_API_KEY.
- Engenharia de Prompt especializada em finanças corporativas e turnaround de PMEs:
  - Explicação do diagnóstico em linguagem simples e acessível a empresários.
  - Identificação de possíveis causas raiz dos indicadores críticos.
  - Recomendações práticas, preventivas e cronograma de ações sugeridas.
  - Respeito às diretrizes éticas e aviso contábil.
- Suporte a histórico de conversação multi-turn.
- Validação rigorosa de mensagens de entrada.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Modelo padrão de alto desempenho na Groq
MODELO_GROQ_PADRAO = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODELO_GROQ_RAPIDO = "llama-3.1-8b-instant"

# Prompt de Sistema do Especialista Financeiro
SYSTEM_PROMPT = """Você é o FinGuard AI, um consultor sênior em diagnóstico de risco financeiro e recuperação de empresas, especializado em auxiliar gestores de pequenas e médias empresas (PMEs).

Sua missão é interpretar o diagnóstico preditivo gerado pelo modelo estatístico/Machine Learning e transformá-lo em orientações claras, práticas e empáticas, evitando termos contábeis herméticos sem explicação prévia.

Diretrizes obrigatórias:
1. Clareza e Linguagem Simples: Explique os termos técnicos de forma direta (ex: explique 'Capital de Giro' como o dinheiro necessário para a empresa girar o dia a dia e pagar fornecedores antes de receber das vendas).
2. Estrutura da Resposta:
   - Resumo da Situação: Diga claramente o nível de risco e o significado da estimativa em tom realista, porém construtivo.
   - Possíveis Causas Raiz: Aponte quais indicadores mais pesaram contra a empresa (ex.: dívida excessiva de curto prazo, margem operacional negativa, estoque parado).
   - Plano de Ação Preventivo: Dê de 3 a 5 passos práticos prioritários (ex.: fluxo de caixa diário, negociação de prazos com fornecedores, revisão da precificação).
3. Contexto do Usuário: Leve em consideração qualquer contexto fornecido (ex: queda nas vendas, problemas com inadimplência).
4. Ética e Alerta Legal: Ao final, reforce sempre com cordialidade que a análise é um apoio preventivo e deve ser complementada por um profissional de contabilidade ou consultoria contábil devidamente habilitado.
"""


def obter_cliente_groq():
    """Inicializa o cliente da Groq com a chave da variável de ambiente."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "A chave de API da Groq não foi encontrada. "
            "Defina a variável de ambiente GROQ_API_KEY para habilitar as respostas do modelo de linguagem."
        )

    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError(
            "O pacote 'groq' não está instalado no ambiente Python. "
            "Execute: pip install groq"
        )


def validar_entradas(mensagem_usuario: str, resumo_diagnostico: str) -> None:
    if not mensagem_usuario or not mensagem_usuario.strip():
        raise ValueError("A mensagem do usuário não pode estar vazia.")
    if len(mensagem_usuario) > 2000:
        raise ValueError("A mensagem do usuário excede o limite máximo de 2.000 caracteres.")
    if not resumo_diagnostico or not resumo_diagnostico.strip():
        raise ValueError("O resumo do diagnóstico preditivo é necessário para orientar a IA.")


def montar_mensagens(
    mensagem_usuario: str,
    resumo_diagnostico: str,
    historico: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Constrói o array de mensagens formatado para a API de chat completions."""
    mensagens: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Contexto financeiro do diagnóstico injetado como mensagem de apoio
    bloco_diagnostico = (
        f"--- DADOS DO DIAGNÓSTICO ATUAL ---\n"
        f"{resumo_diagnostico.strip()}\n"
        f"----------------------------------"
    )
    mensagens.append({"role": "system", "content": bloco_diagnostico})

    # Adicionar histórico prévio (se houver)
    if historico:
        for m in historico:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content and role in ("user", "assistant"):
                mensagens.append({"role": role, "content": content})

    # Mensagem atual do usuário
    mensagens.append({"role": "user", "content": mensagem_usuario.strip()})
    return mensagens


def gerar_orientacao(
    mensagem_usuario: str,
    resumo_diagnostico: str,
    historico: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Função principal chamada pela interface Streamlit / llm_adapter.
    Gera a orientação contextualizada via API da Groq.
    """
    validar_entradas(mensagem_usuario, resumo_diagnostico)
    client = obter_cliente_groq()

    mensagens = montar_mensagens(mensagem_usuario, resumo_diagnostico, historico)

    try:
        # Tenta com o modelo principal de 70B parâmetros
        chat_completion = client.chat.completions.create(
            model=MODELO_GROQ_PADRAO,
            messages=mensagens,
            temperature=0.6,
            max_tokens=1024,
            top_p=0.9,
        )
        resposta = chat_completion.choices[0].message.content
        if resposta and resposta.strip():
            return resposta.strip()
        raise ValueError("Resposta vazia retornada pela Groq.")
    except Exception as e_primaria:
        # Fallback para o modelo rápido de 8B se o modelo de 70B sofrer rate limit ou erro
        try:
            chat_completion = client.chat.completions.create(
                model=MODELO_GROQ_RAPIDO,
                messages=mensagens,
                temperature=0.6,
                max_tokens=1024,
                top_p=0.9,
            )
            resposta = chat_completion.choices[0].message.content
            if resposta and resposta.strip():
                return resposta.strip()
        except Exception:
            pass
        raise RuntimeError(f"Falha na comunicação com a API da Groq: {e_primaria}")


# Teste local via linha de comando
if __name__ == "__main__":
    print("Módulo Groq Service carregado com sucesso.")
    print("Modelo configurado:", MODELO_GROQ_PADRAO)
    chave = os.getenv("GROQ_API_KEY")
    if chave:
        print(f"Chave detectada: {chave[:6]}...{chave[-4:]}")
    else:
        print("Chave GROQ_API_KEY não definida (modo de demonstração).")
