# JARVIS - Assistente Pessoal Inteligente

JARVIS é um assistente pessoal inteligente que utiliza o Groq para inferência de modelo de linguagem grande e OpenAI para embeddings. O serviço fornece uma API para responder perguntas com base nos dados pessoais do usuário armazenados no Supabase.

## Características

- Assistente pessoal "JARVIS" que busca dados do usuário no Supabase
- Personalização com base nos dados e preferências do usuário
- Tratamento personalizado usando nome e pronome de tratamento do usuário
- Armazenamento vetorial em memória (sem necessidade de servidor Qdrant externo)
- Interface web para testes do assistente JARVIS
- Suporte para tarefas recorrentes com cálculo inteligente de próximas ocorrências

## Configuração

### Pré-requisitos

- Python 3.8+
- Chave de API Groq
- Chave de API OpenAI (para embeddings)
- Acesso ao Supabase com tabelas de projetos, tarefas e preferências

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/yourusername/jarvis-assistant.git
cd jarvis-assistant
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Crie um arquivo `.env` com suas chaves de API:
```
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Groq Configuration
GROQ_API_KEY=your_groq_api_key

# Collection Names
QDRANT_USER_COLLECTION_NAME=user_data

# Model Configuration
GROQ_MODEL_NAME=llama-3.1-70b-versatile
EMBEDDING_MODEL_NAME=text-embedding-3-small

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### Executando a API

1. Inicie o servidor API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Após iniciar o servidor, você pode acessar a interface web navegando para:
```
http://localhost:8000
```

## Interface Web

A interface web fornece uma maneira fácil de testar o assistente personalizado JARVIS:

- Permite inserir o UUID e nome do usuário
- Carrega os dados do usuário
- Mostra informações sobre o usuário carregado (projetos, tarefas)
- Interface de chat para conversar com o JARVIS
- Opção para visualizar os prompts gerados

## Como funciona

Esta aplicação utiliza o Qdrant em memória, o que elimina a necessidade de configurar um servidor Qdrant separado. Todos os dados vetoriais são armazenados na memória durante a execução da aplicação, tornando a instalação e execução muito mais simples.

Quando você carrega os dados do usuário, eles são automaticamente convertidos em embeddings e armazenados em um banco de dados vetorial em memória, que é usado para realizar buscas semânticas durante as consultas.

## Endpoints da API

### Interface Web
```
GET /
```
Retorna a interface web para testes interativos.

### Status da API
```
GET /api
```
Retorna uma mensagem de status indicando que a API está funcionando.

### Carregar Dados do Usuário
```
POST /api/load-user-data
```
Carrega os dados do usuário do Supabase para o armazenamento vetorial em memória.

Corpo da requisição:
```json
{
  "user_id": "uuid-do-usuario-no-supabase",
  "user_name": "Nome do Usuário"
}
```

### Consulta do Usuário (JARVIS)
```
POST /api/user-query
```
Responde a uma consulta do usuário com contexto personalizado.

Corpo da requisição:
```json
{
  "user_id": "uuid-do-usuario-no-supabase",
  "user_name": "Nome do Usuário",
  "query": "Quais são meus projetos em andamento?",
  "conversation_history": [
    {"role": "user", "content": "Olá, JARVIS!"},
    {"role": "assistant", "content": "Olá, Sr. Silva! Como posso ajudá-lo hoje?"}
  ],
  "k": 3
}
```

### Busca nos Dados do Usuário
```
POST /api/user-search
```
Busca dados relevantes do usuário com base em uma consulta.

Corpo da requisição:
```json
{
  "user_id": "uuid-do-usuario-no-supabase",
  "user_name": "Nome do Usuário",
  "query": "projetos com prazo esta semana",
  "k": 3
}
```

### Prompt Aumentado do Usuário
```
POST /api/user-augmented-prompt
```
Gera um prompt personalizado com contexto do usuário.

Corpo da requisição:
```json
{
  "user_id": "uuid-do-usuario-no-supabase",
  "user_name": "Nome do Usuário",
  "query": "Liste minhas tarefas pendentes",
  "k": 3
}
```

## Exemplo de Uso

```python
import requests
import json

# Consulta ao JARVIS
response = requests.post(
    "http://localhost:8000/api/user-query",
    json={
        "user_id": "uuid-do-usuario-no-supabase",
        "user_name": "Nome do Usuário",
        "query": "Quais são meus projetos e tarefas prioritárias?"
    }
)

# Mostrar resposta
result = response.json()
print(result["answer"])
```

## Estrutura de Dados do Supabase

O sistema utiliza as seguintes tabelas no Supabase:

### Projects
Armazena informações sobre projetos do usuário
- `id`: UUID do projeto
- `name`: Nome do projeto
- `color`: Cor associada ao projeto
- `user_id`: UUID do usuário
- `created_at`, `updated_at`: Timestamps

### Tasks
Armazena informações sobre tarefas do usuário
- `id`: UUID da tarefa
- `title`: Título da tarefa
- `description`: Descrição da tarefa
- `completed`: Status de conclusão
- `due_date`: Data de vencimento
- `project_id`: Referência ao projeto
- `user_id`: UUID do usuário
- `recurrence_type`: Tipo de recorrência (daily, weekly, monthly, yearly)
- `recurrence_interval`: Intervalo de recorrência

### User Preferences
Armazena preferências do usuário
- `id`: UUID da preferência
- `user_id`: UUID do usuário
- `form_of_address`: Forma de tratamento preferida
- `phone_number`: Número de telefone
- `allow_notifications`: Preferência de notificações

## Tarefas Recorrentes

O JARVIS possui um recurso avançado para lidar com tarefas recorrentes. Quando o usuário pergunta sobre tarefas para um determinado período ou data, o assistente é capaz de calcular automaticamente as ocorrências de tarefas recorrentes pendentes que caem dentro desse período.

### Funcionamento das Tarefas Recorrentes

Cada tarefa pode ter as seguintes propriedades de recorrência:
- `recurrence_type`: Define o tipo de recorrência ("daily", "weekly", "monthly", "yearly")
- `recurrence_interval`: Define o intervalo de recorrência (ex: a cada 2 dias, a cada 3 semanas)

O JARVIS aplica a seguinte lógica:
1. Identifica tarefas recorrentes não concluídas (completed = false)
2. Calcula as próximas ocorrências baseadas no tipo e intervalo de recorrência
3. Filtra as ocorrências que caem dentro do período solicitado pelo usuário
4. Inclui essas tarefas recorrentes nas respostas

Exemplos de consultas que utilizam este recurso:
- "Quais são minhas tarefas para amanhã?"
- "O que tenho planejado para esta semana?"
- "Mostre minhas tarefas para o mês de julho"
- "Tenho alguma tarefa recorrente pendente?"

Esta funcionalidade torna o JARVIS particularmente útil para gerenciamento de agendas e planejamento de atividades, especialmente para usuários com muitas tarefas regulares.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes. 