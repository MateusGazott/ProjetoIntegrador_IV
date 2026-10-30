# Integrante 3 — Integração da API da Groq (LLM)

## Visão Geral
Este módulo implementa a comunicação com a API da **Groq**, fornecendo inteligência contextualizada para explicar o diagnóstico financeiro gerado pelo modelo preditivo em linguagem acessível e propor planos de ação preventivos.

---

## 1. Configuração e Modelos Utilizados
- **Provedor:** Groq Cloud API (`https://api.groq.com`)
- **Modelos Homologados:**
  - `llama-3.3-70b-versatile`: Modelo principal de alta precisão analítica e raciocínio financeiro.
  - `llama-3.1-8b-instant`: Modelo secundário de fallback com resposta ultrarrápida.
- **Variável de Ambiente:** `GROQ_API_KEY` (chave protegida, nunca versionada no código).

---

## 2. Engenharia de Prompt Especializada
O sistema utiliza um System Prompt configurado como um Consultor Sênior em Turnaround e Gestão de Riscos (FinGuard AI):
- Traduz indicadores contábeis complexos para a realidade do empresário de PME.
- Identifica causas-raiz com base nos pontos de atenção do diagnóstico (`resumoParaPromptLLM`).
- Sugere planos de mitigação imediatos: recomposição de capital de giro, escalonamento de passivos de curto prazo e estancamento de sangrias operacionais.
- Adiciona avisos éticos e legais reforçando o papel de apoio preventivo sem substituir a consultoria contábil habilitada.

---

## 3. Contrato da Função
```python
def gerar_orientacao(
    mensagem_usuario: str,
    resumo_diagnostico: str,
    historico: list[dict]
) -> str:
    ...
```

---

## 4. Testes do Módulo
Execute os testes unitários com:
```bash
python test_groq.py
```
**Resultado:** 5/5 testes aprovados (validações, prompts, tratamento de erros e fallback).
