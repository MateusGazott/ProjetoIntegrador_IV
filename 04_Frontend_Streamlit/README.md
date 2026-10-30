# Integrante 4 — Interface Web e Chat com IA (Streamlit)

A ideia é criar a interface do MVP, onde o usuário preenche os indicadores financeiros, recebe um diagnóstico de risco e pode conversar com um assistente de IA para tirar dúvidas e receber orientações.

A interface vai consumir o backend em C# do Integrante 2 e utilizar o módulo da Groq desenvolvido pelo Integrante 3 para responder às mensagens do chat.

## Estrutura do projeto

A pasta `Integrante4_Frontend_Streamlit` deve conter os seguintes arquivos:

- `app.py`: interface principal, com formulário, diagnóstico e chat.
- `api_client.py`: responsável pela comunicação com o backend C# e pelo tratamento de erros.
- `llm_adapter.py`: faz a integração com o módulo da Groq e disponibiliza uma resposta local caso a IA falhe.
- `mock_backend.py`: simula o backend para permitir testes sem precisar do .NET.
- `test_interface.py`: contém os 13 testes automatizados da interface.
- `model_metadata.json`: cópia dos metadados utilizada apenas pelo mock.
- `requirements.txt`: lista das dependências do projeto.
- `.streamlit/config.toml`: configurações do Streamlit.

## Como executar

Primeiro, instale as dependências:

```bash
pip install -r requirements.txt
```

Depois, abra um terminal para iniciar o backend real do Integrante 2:

```bash
cd RiskAnalysisBackend
dotnet run --urls "http://localhost:5000"
```

Se não tiver o .NET disponível, é possível utilizar o `mock_backend.py` para testar a interface sem o backend real.

Em outro terminal, dentro da pasta da interface, execute:

```bash
streamlit run app.py
```

A URL do backend pode ser alterada na barra lateral da interface ou pela variável de ambiente `BACKEND_URL`. Por padrão, será utilizada `http://localhost:5000/api/diagnostico`.

## O que a interface precisa ter

- **Formulário financeiro:** os campos devem ser carregados automaticamente a partir do endpoint `GET /atributos-modelo`. O sistema deve aceitar números com vírgula ou ponto decimal, como `0,25` e `0.25`. Se um campo ficar vazio, o backend ficará responsável por preencher o valor com a mediana.
- **Diagnóstico de risco:** após o envio do formulário, a interface deve mostrar a classificação (baixo, médio ou alto risco), a porcentagem de insolvência, uma barra visual, uma tabela com os indicadores e a identificação dos valores que foram preenchidos automaticamente. Também deve exibir os alertas enviados pelo backend e uma explicação inicial do resultado.
- **Chat com IA:** o usuário poderá conversar com o assistente depois de realizar a primeira análise. O chat deve manter o histórico das mensagens e validar os textos enviados.
- **Tratamento de erros:** a interface precisa lidar com situações como backend offline, erros HTTP 400, 404 e 500, campos inválidos, mensagens vazias ou muito longas e falhas na integração com a Groq.
- **Interface responsiva:** a tela deve funcionar bem tanto em computadores quanto em dispositivos com largura de 390 px.
- **Carregamento e aviso legal:** utilizar os indicadores de carregamento do Streamlit e deixar o aviso legal visível no topo e no rodapé da página.

## Integração com o Integrante 3 (Groq)

O arquivo `groq_service.py` deve ser criado nesta mesma pasta, com a função:

```python
def gerar_orientacao(
    mensagem_usuario: str,
    resumo_diagnostico: str,
    historico: list[dict]
) -> str:
    ...
```

Essa função será responsável por gerar as respostas do assistente utilizando a Groq.

- O `resumo_diagnostico` deve receber o campo `resumoParaPromptLLM` do endpoint `/chat-consulta`.
- O `historico` deve conter as mensagens anteriores no formato `role` e `content`, indicando quem enviou cada mensagem (`user` ou `assistant`).
- Se a chamada à Groq falhar, a interface deve avisar o usuário e utilizar uma resposta local como alternativa, sem interromper o funcionamento do chat.
- A chave da API deve ficar na variável de ambiente `GROQ_API_KEY`, nunca diretamente no código.

Enquanto o módulo da Groq não estiver disponível, o chat deve continuar funcionando com respostas locais baseadas em regras simples, deixando claro na tela que está utilizando esse modo alternativo.

## Testes automatizados

Para executar os testes, inicie o `mock_backend.py` ou deixe o backend C# rodando e execute:

```bash
python -m pytest test_interface.py -q
```

Os 13 testes devem verificar os principais cenários, incluindo diagnósticos de baixo e alto risco, números com vírgula decimal, valores inválidos ou `NaN`, formulário vazio, valores extremos, dados incompletos e validação do chat.

Também devem ser testados o limite de 1.000 caracteres por mensagem, o bloqueio do chat antes da primeira análise, a indisponibilidade do backend, as falhas e o funcionamento da Groq e a presença do aviso legal.

## Alguns pontos importantes para o grupo

- Sem o arquivo `modelo_risco.onnx`, o backend funciona em modo heurístico. A interface deve avisar o usuário e mostrar o valor de `modoExecucao`, para deixar claro quando o resultado não vem do modelo treinado.
- O `Attr5` está com descrições diferentes no `exportar_modelo_onnx.py` e no `model_metadata.json`. Precisamos alinhar isso com o Integrante 1 para manter a descrição correta.
- Como o endpoint `/chat-consulta` guarda o diagnóstico em memória, cada análise deve gerar um `empresaId` único. Isso evita que usuários diferentes acabem acessando o diagnóstico uns dos outros.
- As pastas `.vs/`, geradas pelo Visual Studio, estão incluídas no ZIP. É recomendável adicioná-las ao `.gitignore` para evitar arquivos desnecessários no repositório.