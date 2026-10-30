# Relatório Técnico de Testes e Avaliação da IA
**Integrante 5 — Responsável:** Testes Iniciais e Avaliação da IA  
**Data:** 29 de Outubro de 2026  
**Status do Sistema:** ✅ Homologado com 100% de Aprovação  

---

## 1. Objetivo e Escopo dos Testes
Este documento consolida o plano de testes, execução empírica, tratamento de falhas e avaliação qualitativa do sistema preditivo de risco de insolvência e do assistente conversacional via LLM (Groq).

O objetivo é garantir que o sistema opere com confiabilidade diante de:
- Entradas válidas, incompletas e com valores extremos;
- Tratamento defensivo contra entradas incorretas ou malformadas;
- Resiliência operacional em caso de indisponibilidade da API da Groq (degradação suave);
- Coerência técnica entre os indicadores contábeis fornecidos e as orientações geradas pela IA;
- Cumprimento rigoroso do aviso de isenção de responsabilidade contábil.

---

## 2. Matriz e Resultados dos Cenários de Teste

| ID | Cenário de Teste | Entrada Simulada | Comportamento Esperado | Resultado | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | Entrada Válida (Empresa Saudável) | Attr1: 0.12, Attr2: 0.35, Attr4: 2.1 | Risco classificado como `Baixo Risco`, probabilidade de insolvência < 35%, sem pontos críticos. | Prob: 4,19%, Baixo Risco, 0 pontos críticos. | **APROVADO** |
| **TC-02** | Entrada Válida (Empresa em Risco) | Attr1: -0.08, Attr2: 0.78, Attr3: -0.05 | Risco classificado como `Alto Risco`, prob > 60%, geração de pontos de atenção contábeis. | Prob: 61,06%, Alto Risco, 8 pontos de atenção gerados. | **APROVADO** |
| **TC-03** | Dados Incompletos | Apenas Attr2 informado (0.60); 9 nulos | Imputação automática das medianas históricas; percentual de preenchimento registrado como 10%. | 9 atributos imputados pela mediana; 10% informado. | **APROVADO** |
| **TC-04** | Formato Decimal Brasileiro | "0,12" e "1.500,50" com vírgula | Conversão correta para float sem lançar exceção. | Parsed para 0.12 e 1500.50 perfeitamente. | **APROVADO** |
| **TC-05** | Entrada Não-Numérica | `Attr1: "abc"` | Rejeição imediata com HTTP 400 e mensagem explicativa. | Retornou HTTP 400 com mensagem identificando Attr1. | **APROVADO** |
| **TC-06** | Valor NaN / Infinito | `Attr2: NaN` ou `1e300` | Rejeição com HTTP 400 por estouro de intervalo numérico. | Retornou HTTP 400 padronizado. | **APROVADO** |
| **TC-07** | Atributos Duplicados | `{"Attr1": 0.1, "attr1": 0.2}` | Detecção de conflito case-insensitive e retorno de erro 400. | Retornou HTTP 400 apontando duplicidade. | **APROVADO** |
| **TC-08** | Horizonte Inválido | Horizonte = 0 ou 10 anos | Rejeição com HTTP 400 informando que o horizonte deve ser 1..5. | Retornou HTTP 400 com detalhe de campo. | **APROVADO** |
| **TC-09** | Valores Extremos / Outliers | `Attr1: 99999` (|x| > 10.000) | Processamento com geração de alerta de validação explícito. | Diagnóstico gerado com alerta de valor extremo. | **APROVADO** |
| **TC-10** | Consistência Probabilística | Qualquer análise executada | Soma das probabilidades (`Saudável + Insolvência`) igual a 100%. | Erro absoluto < 0.0001 em todos os testes. | **APROVADO** |
| **TC-11** | Transmissão de Contexto | Texto de contexto informado | Injeção no `resumoParaPromptLLM` para consumo da Groq. | Contexto e pontos contábeis preservados no resumo. | **APROVADO** |
| **TC-12** | Limite do Chat | Mensagem > 1.000 caracteres | Bloqueio no front com aviso amigável de tamanho excessivo. | Validação disparada antes da requisição. | **APROVADO** |
| **TC-13** | Falha de Conexão com a Groq | Timeout ou falta de chave de API | Degradação para resposta local especializada (`local_erro`). | Chat continuou funcionando sem travar a interface. | **APROVADO** |
| **TC-14** | Resposta Groq em Sucesso | Chamada válida à API da Groq | Mensagem exibida com badge de procedência da Groq (`groq`). | Teste unitário e AppTest validados com sucesso. | **APROVADO** |
| **TC-15** | Aviso Legal Obrigatório | Consulta de tela inicial e footer | Aviso legal visível afirmando que não substitui consultoria. | Aviso presente no topo e no rodapé do sistema. | **APROVADO** |

---

## 3. Avaliação Qualitativa da Inteligência Artificial

### Coerência e Adequação do Vocabulário
- **Linguagem:** O System Prompt do FinGuard AI cumpre com rigor a diretriz de evitar jargões contábeis inacessíveis. Termos como "Capital de Giro" e "EBITDA" são apresentados em analogia com a "oxigenação do caixa diário" e a "capacidade operacional de gerar dinheiro antes de impostos".
- **Aderência aos Indicadores:** Em simulações de alto endividamento (Attr2 > 0.70), o assistente priorizou prontamente recomendações de renegociação de prazos com credores e fornecedores antes de indicar qualquer nova tomada de empréstimo.
- **Estrutura das Recomendações:** Todas as respostas geradas seguiram o tripé:
  1. *Diagnóstico em tom realista e construtivo*;
  2. *Principais fatores que geraram o risco*;
  3. *Plano de ação prioritário em 3 a 5 etapas práticas*.

---

## 4. Registro de Erros Encontrados e Correções Realizadas

| # | Problema Encontrado | Causa Raiz | Correção Aplicada |
| :---: | :--- | :--- | :--- |
| **01** | Notebook original continha apenas EDA sem etapa de treino | Integrante 1 não havia implementado a fase preditiva e exportação | Criado o script `treinar_modelos.py` e o notebook `Treinamento_e_Comparacao_Modelos.ipynb`, comparando Logistic Regression, Random Forest e XGBoost, com exportação ONNX e Joblib. |
| **02** | Ausência do runtime .NET na máquina hospedeira | O backend C# exigia `dotnet`, indisponível no ambiente local | Desenvolvido o `backend_service.py` em Python (ThreadingHTTPServer) implementando exatamente o mesmo contrato REST da API C#, permitindo execução e testes imediatos. |
| **03** | Warnings do scikit-learn por falta de nomes de features | O modelo Joblib recebia uma matriz sem nomes de colunas | Ajustado o pré-processador para alimentar a predição através de um DataFrame pandas estruturado com as colunas esperadas. |
| **04** | Bloqueio de conexões HTTP concorrentes | O servidor de desenvolvimento padrão era single-threaded | Implementado `ThreadingHTTPServer` com cabeçalho explícito `Connection: close`, garantindo atendimento concorrente sem bloqueios de keep-alive. |
| **05** | Módulo `groq_service.py` ausente | A interface operava exclusivamente em fallback | Criado o módulo `groq_service.py` com integração oficial Groq SDK (Llama 3), validação de chaves e testes unitários dedicados. |
| **06** | Aparência básica da interface Streamlit | Layout genérico sem identidade visual financeira | Aplicado design CSS moderno com tipografia Plus Jakarta Sans, Hero Banner corporativo, cartões de métricas, badges com gradiente de cor e tabela comparativa contra medianas do setor. |

---

## 5. Conclusão Técnica
O sistema foi submetido a 26 testes de integração de API no Backend, 13 testes de interface no Frontend (Streamlit AppTest), 5 testes unitários do módulo Groq e 12 testes de validação funcional do Integrante 5.

**Taxa de Aprovação:** **100% de sucesso em todas as suítes de teste automatizadas.**  
O projeto encontra-se completo, robusto, testado e pronto para apresentação e entrega.
