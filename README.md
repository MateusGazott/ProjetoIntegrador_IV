# ProjetoIntegrador_IV
# Tema: Previsão de Falencia de Pequenas Empresas

A plataforma opera em duas frentes complementares:

1. **Atuação Direta (Modelo Preditivo Tabular):** Um algoritmo de Machine Learning supervisionado processa os indicadores financeiros quantitativos do usuário (faturamento, despesas, passivos, capital de giro) para calcular uma probabilidade estatística exata e objetiva do risco de insolvência.
2. **Atuação Indireta (Copiloto de Consultoria via LLM):** Uma LLM integrando técnicas de RAG (via API do GroqCloud) recebe o diagnóstico numérico e o contexto do usuário para traduzir a análise técnica em linguagem clara, gerando planos de ação preventivos e orientações estratégicas acessíveis.

### IA 
- **Privacidade (LGPD):** O processamento de dados é realizado de forma anonimizada, sem necessidade de envio de dados identificáveis (como CNPJ, Razão Social ou nomes) para a LLM.
- **Segurança:** Conta com sanitização de entradas contra ataques de *Prompt Injection*.
- **Transparência:** O sistema atua como uma ferramenta de apoio à decisão, sem substituir auditorias ou consultorias contábeis formais.
