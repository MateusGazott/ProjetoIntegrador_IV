# Integrante 2 — Backend e Integração do Modelo

## Visão Geral
Este módulo é responsável por disponibilizar o serviço REST de predição de risco de insolvência, integrando o modelo treinado (ONNX / Joblib) com pré-processamento de indicadores contábeis, cálculo de pontos de atenção e provimento de contexto para a LLM (Groq) e a Interface Streamlit.

---

## 1. Arquitetura do Backend
O projeto oferece duas implementações intercambiáveis que compartilham 100% do mesmo contrato de API:
1. **Solução ASP.NET Core (.NET 8 / C#)**: localizada em `RiskAnalysisBackend/`, utiliza `Microsoft.ML.OnnxRuntime` para inferência de alto desempenho em C#.
2. **Serviço Python Nativo Multi-thread (`backend_service.py`)**: servidor REST pronto para execução imediata em qualquer ambiente com Python, carregando o modelo treinado `modelo_risco.joblib` ou inferência heurística.

---

## 2. Endpoints da API

### `GET /api/diagnostico/saude`
Retorna o status operacional do backend.
```json
{
  "status": "ok",
  "modo": "RandomForest",
  "ehModeloReal": true,
  "totalAtributos": 10,
  "diagnosticosArmazenados": 1
}
```

### `GET /api/diagnostico/atributos-modelo`
Retorna a lista de indicadores requeridos, suas descrições contábeis e as medianas calculadas no treinamento.

### `POST /api/diagnostico/analisar`
Recebe os dados da empresa e retorna a classificação de risco, probabilidades, pontos de atenção e alertas.
- **Corpo da Requisição:**
```json
{
  "empresaId": "emp-001",
  "nomeEmpresa": "Indústria Alfa LTDA",
  "horizonteAnos": 1,
  "contextoUsuario": "Queda recente de receitas e acúmulo de duplicatas.",
  "indicadores": {
    "Attr1": -0.08,
    "Attr2": 0.78,
    "Attr3": -0.05,
    "Attr4": 0.85,
    "Attr6": -0.10,
    "Attr7": -0.04,
    "Attr8": 0.20,
    "Attr9": 0.70,
    "Attr10": 0.15
  }
}
```
- **Resposta:**
```json
{
  "empresaId": "emp-001",
  "nomeEmpresa": "Indústria Alfa LTDA",
  "horizonteAnos": 1,
  "classificacaoRisco": "Alto Risco",
  "probabilidadeSaudavel": 0.3894,
  "probabilidadeInsolvencia": 0.6106,
  "classePrevista": 1,
  "modoExecucao": "RandomForest",
  "percentualIndicadoresInformados": 90.0,
  "pontosDeAtencao": [
    "Prejuízo líquido em relação aos ativos totais (Attr1 negativo).",
    "Alto grau de endividamento geral: dívidas superam 70% dos ativos totais (Attr2).",
    "Capital de giro negativo: passivo circulante maior que ativo circulante (Attr3)."
  ]
}
```

### `GET /api/diagnostico/chat-consulta?empresaId={id}`
Permite ao chat consultar o diagnóstico mais recente ou de uma empresa específica, entregando o campo `resumoParaPromptLLM` formatado para consumo da Groq.

---

## 3. Como Executar

### Opção A — Serviço Python (Recomendado):
```bash
python backend_service.py 5000
```

### Opção B — Backend C# (.NET SDK):
```bash
cd RiskAnalysisBackend
dotnet run --urls "http://localhost:5000"
```

---

## 4. Testes do Backend
Execute a suíte de testes com 26 cenários:
```bash
python run_backend_tests_verbose.py
```
**Resultado:** 26/26 testes aprovados (100% de cobertura de validações, casos extremos, imputação e histórico).
