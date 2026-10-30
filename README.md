# FinGuard Analytics — Diagnóstico Preditivo de Risco de Falência com IA

Sistema inteligente para detecção precoce de risco de insolvência e falência empresarial voltado a pequenas e médias empresas (PMEs). O projeto integra modelos tabulares de Machine Learning treinados sobre o *Polish Companies Bankruptcy Dataset* com serviços de inferência de alta performance, módulo conversacional com a API da Groq (Llama 3) e uma interface web moderna em Streamlit.

---

## 👥 Divisão de Responsabilidades e Entregas dos Integrantes

### 🔹 Integrante 1 — Treinamento do Modelo Tabular 
- **Localização:** `01_Treinamento_Modelo_Tabular/`
- **Atividades:**
  - Carregamento e consolidação dos dados de 1 a 5 anos (43.405 registros).
  - Tratamento de dados ausentes e infinitos por imputação com medianas amostrais.
  - Seleção dos 10 atributos contábeis críticos (`Attr1` a `Attr10`).
  - Treinamento comparativo entre **Random Forest** (82,29% acurácia, ROC-AUC 0,7902), **XGBoost** (ROC-AUC 0,8048) e **Regressão Logística** (Baseline).
  - Seleção do Random Forest para produção e cálculo de importância das variáveis (`Attr6`, `Attr1` e `Attr7` como principais).
  - Exportação do modelo para **ONNX** (`modelo_risco.onnx`), **Joblib** (`modelo_risco.joblib`) e metadados (`model_metadata.json`).
  - Notebook completo documentado: `Treinamento_e_Comparacao_Modelos.ipynb`.

### 🔹 Integrante 2 — Backend e Integração do Modelo
- **Localização:** `02_Backend_Integracao/`
- **Atividades:**
  - Implementação da API REST com suporte dual:
    1. **Solução ASP.NET Core (C# / .NET 8)** em `RiskAnalysisBackend/` usando `Microsoft.ML.OnnxRuntime`.
    2. **Serviço Python Nativo Multi-thread (`backend_service.py`)** para execução autônoma imediata.
  - Pré-processamento e normalização de atributos com tratamento case-insensitive e imputação automática.
  - Detecção de valores extremos (|x| > 10.000) e geração de alertas contábeis.
  - Algoritmo de geração de **Pontos de Atenção** específicos (prejuízo, liquidez < 1,0, capital de giro negativo, alto endividamento).
  - Endpoint de consulta para a LLM (`GET /api/diagnostico/chat-consulta`) com histórico por empresa.
  - Suíte de 26 testes de API (`run_backend_tests_verbose.py`) com 100% de aprovação.

### 🔹 Integrante 3 — Integração da API da Groq
- **Localização:** `03_Integracao_Groq_LLM/`
- **Atividades:**
  - Integração com a API da Groq usando os modelos `llama-3.3-70b-versatile` e `llama-3.1-8b-instant`.
  - Proteção de credenciais via variável de ambiente `GROQ_API_KEY`.
  - Engenharia de Prompts (FinGuard AI) para traduzir o diagnóstico técnico em orientações práticas, didáticas e acessíveis para empresários.
  - Estruturação de respostas com causas-raiz e planos de ação prioritários em 3 a 5 etapas.
  - Mecanismo de degradação graciosa para fallback local quando offline.
  - Suíte de testes unitários (`test_groq.py`) com 100% de aprovação.

### 🔹 Integrante 4 — Interface Web e Chat Textual
- **Localização:** `04_Frontend_Streamlit/`
- **Atividades:**
  - Interface web responsiva em **Streamlit** com design corporativo refinado (tipografia Plus Jakarta Sans, Hero Banner, cartões de métricas e badges de risco com gradientes).
  - Formulário financeiro dinâmico aceitando notação com vírgula ou ponto (ex.: `0,25` ou `0.25`).
  - Botões de preenchimento rápido para demonstração (*Empresa Saudável* e *Empresa em Risco*).
  - Painel de diagnóstico visual com termômetro de risco, probabilidades, pontos de atenção e tabela comparativa contra a mediana do setor.
  - Chat textual conversacional com histórico multi-turn e identificação de procedência (Groq vs Local).
  - Alerta legal em conformidade legal informando que o sistema é uma ferramenta de apoio e não substitui consultoria contábil.
  - 13 testes automatizados via `pytest` e `Streamlit AppTest` (`test_interface.py`) com 100% de aprovação.

### 🔹 Integrante 5 — Testes Iniciais e Avaliação da IA
- **Localização:** `05_Testes_e_Avaliacao_IA/`
- **Atividades:**
  - Execução de testes de estresse: entradas válidas, dados incompletos, entradas incorretas (strings, NaNs, duplicidades), horizontes inválidos e outliers.
  - Verificação de resiliência e degradação suave na indisponibilidade da Groq.
  - Avaliação de coerência temática das recomendações da IA em relação aos indicadores informados.
  - Registro de incidentes e plano de ação corretivo implementado.
  - Relatório formal consolidado: `Relatorio_Testes_e_Avaliacao_IA.md` com 12/12 cenários aprovados.

---

## 🚀 Como Executar o Projeto Completo

### Método 1: Início Rápido com 1 Clique (Windows)
Dê um duplo clique no arquivo:
```cmd
iniciar_sistema.bat
```
O script iniciará automaticamente o **Backend** na porta 5000 e a **Interface Streamlit** na porta 8501.

### Método 2: Início Manual por Terminais

**Terminal 1 — Iniciar o Backend:**
```bash
python 02_Backend_Integracao/backend_service.py 5000
```

**Terminal 2 — Iniciar o Frontend:**
```bash
streamlit run 04_Frontend_Streamlit/app.py
```
Acesse no navegador: `http://localhost:8501`

---

## 🧪 Como Executar Todos os Testes

Para validar a integridade completa de todos os módulos de uma vez só:
```bash
python executar_todos_os_testes.py
```

### Resultados dos Testes Automatizados:
- **Backend REST API (Integrante 2):** 26/26 testes aprovados (100%)
- **Módulo Groq LLM (Integrante 3):** 5/5 testes aprovados (100%)
- **Interface Streamlit (Integrante 4):** 13/13 testes aprovados (100%)
- **Avaliação da IA e Resiliência (Integrante 5):** 12/12 testes aprovados (100%)
- **Total Geral:** **56/56 testes com 100% de aprovação!**

---

## 📁 Estrutura de Pastas do Repositório

```
├── 01_Treinamento_Modelo_Tabular/        # Integrante 1 (Mateus - ML)
│   ├── dados/                           # Arquivos 1year.csv a 5year.csv
│   ├── treinar_modelos.py               # Pipeline completo de treino e comparação
│   ├── Treinamento_e_Comparacao_Modelos.ipynb # Notebook executado com gráficos
│   ├── modelo_risco.onnx                # Modelo exportado em formato ONNX
│   ├── modelo_risco.joblib              # Modelo serializado em Joblib
│   ├── model_metadata.json              # Metadados, descrições e medianas
│   └── README.md                        # Documentação técnica do Integrante 1
│
├── 02_Backend_Integracao/                # Integrante 2 (Backend)
│   ├── backend_service.py               # Servidor REST Python multi-thread
│   ├── RiskAnalysisBackend/             # Projeto ASP.NET Core C# (.NET 8)
│   ├── run_backend_tests_verbose.py     # Suíte de 26 testes de API
│   ├── testes_api_backend.py            # Testes originais do backend
│   └── README.md                        # Documentação da API REST
│
├── 03_Integracao_Groq_LLM/               # Integrante 3 (Groq API)
│   ├── groq_service.py                  # Cliente Groq Cloud com fallback
│   ├── test_groq.py                     # Suíte de testes unitários da Groq
│   └── README.md                        # Engenharia de Prompts e configurações
│
├── 04_Frontend_Streamlit/                # Integrante 4 (Frontend)
│   ├── app.py                           # Aplicação web Streamlit aprimorada
│   ├── api_client.py                    # Cliente HTTP para comunicação com backend
│   ├── llm_adapter.py                   # Adaptador de chat e resposta local
│   ├── test_interface.py                # 13 testes automatizados (AppTest)
│   ├── .streamlit/config.toml           # Configurações de tema e UI
│   └── README.md                        # Instruções e documentação da interface
│
├── 05_Testes_e_Avaliacao_IA/             # Integrante 5 (Testes e IA)
│   ├── executar_testes_integrante5.py   # Suíte de validação e resiliência
│   ├── Relatorio_Testes_e_Avaliacao_IA.md # Relatório técnico formal de testes
│   └── README.md                        # Resumo das rotinas de homologação
│
├── executar_todos_os_testes.py           # Runner global consolidado
├── iniciar_sistema.bat                   # Inicializador rápido do sistema
└── README.md                             # Documento mestre do projeto
```

---

## ⚖️ Aviso Legal
*Este sistema é uma ferramenta de apoio analítico baseada em modelagem probabilística e inteligência artificial generativa. Os resultados e recomendações fornecidos não substituem, em nenhuma circunstância, a avaliação contábil, tributária, financeira e jurídica realizada por profissionais certificados e habilitados.*
