import os
from typing import List, Dict, Any, Optional
import logging
import pathlib

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from groq import Groq
from langchain_text_splitters import MarkdownTextSplitter

import config
from user_data_service import UserDataService

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conteúdo embutido do JARVIS para usar quando o arquivo não estiver disponível
JARVIS_SALES_CONTENT = """# JARVIS: Seu Assistente Pessoal de IA

## O que é o JARVIS?

O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial, inspirado no famoso assistente do Homem de Ferro. Diferente de outros aplicativos de tarefas, o JARVIS entende linguagem natural e organiza sua vida através de uma interface intuitiva e acessível de qualquer dispositivo.

**[➡️ Acesse agora o JARVIS: https://www.appjarvis.com.br/](https://www.appjarvis.com.br/)**

## Como o JARVIS transforma sua produtividade

### 🔍 Entendendo Linguagem Natural

O JARVIS utiliza modelos avançados de IA (GPT-4o) para compreender suas tarefas exatamente como você as descreve:

- **Simplesmente digite ou fale**: "Reunião com José na terça-feira às 15h sobre o projeto Alfa #Trabalho/Reuniões"
- **O JARVIS extrai automaticamente**:
  - Título: "Reunião com José sobre o projeto Alfa"
  - Data: Próxima terça-feira
  - Horário: 15:00
  - Projeto: Trabalho
  - Seção: Reuniões

### 📅 Gestão Inteligente de Tarefas

O sistema organiza suas tarefas automaticamente com base em:

- **Datas inteligentes**: reconhece expressões como "amanhã", "próxima semana", "início do mês"
- **Recorrências**: configura automaticamente tarefas para se repetirem quando você diz "todos os dias", "toda segunda", "mensalmente"
- **Categorização automática**: usa hashtags (#Projeto) e barras (/Seção) para organizar suas tarefas

### 🔔 Notificações Que Funcionam

- **Alertas no dispositivo**: receba lembretes contextuais no momento certo
- **Notificações via WhatsApp**: mantenha-se informado mesmo quando não estiver usando o aplicativo
- **Priorização inteligente**: o sistema aprende quais tarefas são mais importantes para você

### 🤖 JARVIS no WhatsApp

Acesso total aos seus dados, assistente proativo, cria e gerencia tarefas.

## Segurança e Privacidade

Autenticação segura, armazenamento criptografado e controle total dos dados.

## Comece Agora - Vagas Limitadas!

**[🚀 Experimente o JARVIS GRATUITAMENTE: https://www.appjarvis.com.br/](https://www.appjarvis.com.br/)**

## Planos Acessíveis

- **7 Dias Grátis** – Experimente todas as funcionalidades sem cartão de crédito
- **Plano Anual** – Apenas R$ 50,00 por um ano inteiro de produtividade transformadora """

class SalesRAGService:
    """Serviço RAG especializado em vendas do JARVIS"""

    def __init__(self):
        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        )
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize vector database for sales content
        self.sales_docs = []
        # Armazenar embeddings separadamente
        self.doc_embeddings = {}
        self.load_sales_content()
        
        # For user context feature (optional/future)
        self.user_data_service = UserDataService(use_in_memory=True, embedding_model=self.embedding_model)
        
        logger.info(f"Sales RAG Service initialized with model: {config.GROQ_MODEL_NAME}")
    
    def load_sales_content(self):
        """Load sales content from markdown file or use embedded content"""
        try:
            # Tentar usar o conteúdo embutido
            content = JARVIS_SALES_CONTENT
            logger.info("Usando conteúdo embutido para JARVIS Sales")
            
            # Processar o conteúdo como um documento
            document = Document(page_content=content)
            documents = [document]
            
            # Split the document into chunks
            text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
            self.sales_docs = text_splitter.split_documents(documents)
            
            # Add metadata to identify sections
            for i, doc in enumerate(self.sales_docs):
                # Extract section from content if possible
                content = doc.page_content
                section = "General"
                
                if "## " in content:
                    section_line = content.split("## ")[1].split("\n")[0]
                    section = section_line.strip()
                elif "### " in content:
                    section_line = content.split("### ")[1].split("\n")[0]
                    section = section_line.strip()
                
                doc.metadata["section"] = section
                doc.metadata["chunk_id"] = i
                doc.metadata["source"] = "jarvis_sales_content"
                
                # Pré-calcular e armazenar embeddings separadamente com IDs baseados no índice
                self.doc_embeddings[i] = self.embedding_model.embed_documents([doc.page_content])[0]
            
            logger.info(f"Loaded {len(self.sales_docs)} sales content chunks with embeddings")
            return {"status": "success", "message": f"Loaded {len(self.sales_docs)} sales content chunks"}
        except Exception as e:
            logger.error(f"Error processing sales content: {str(e)}", exc_info=True)
            # Initialize with default content to avoid cascading errors
            self.sales_docs = [
                Document(
                    page_content="JARVIS: Seu Assistente Pessoal de IA. O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial.",
                    metadata={"section": "Geral", "chunk_id": 0, "source": "default_content"}
                )
            ]
            # Criar embedding para o conteúdo padrão
            self.doc_embeddings[0] = self.embedding_model.embed_documents(["JARVIS: Seu Assistente Pessoal de IA. O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial."])[0]
            return {"status": "error", "message": f"Error processing sales content: {str(e)}"}
    
    def search_sales_content(self, query: str, k: int = 3):
        """Search for relevant sales content based on the query"""
        # Simple similarity search (could use a proper vector DB in production)
        if not self.sales_docs:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Usar os embeddings pré-calculados
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Criar uma matriz de embeddings a partir do dicionário
        doc_embedding_list = [self.doc_embeddings[i] for i in range(len(self.sales_docs))]
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], doc_embedding_list)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "text": self.sales_docs[idx].page_content,
                "metadata": self.sales_docs[idx].metadata
            })
        
        return results
    
    def generate_sales_prompt(self, query: str, k: int = 3):
        """Generate a persuasive sales prompt based on the query"""
        # Get relevant sales documents
        sales_results = self.search_sales_content(query, k=k)
        
        # Extract text from results
        sales_context = "\n".join([result["text"] for result in sales_results])
        
        # Create system prompt for sales agent
        system_prompt = """Você é Davi Cardoso, Gerente Comercial do JARVIS. Você deve se comunicar no estilo do WhatsApp - mensagens curtas, diretas e com boa formatação.

ESTILO DE COMUNICAÇÃO NO WHATSAPP:
- Use mensagens curtas e objetivas (2-4 parágrafos no máximo)
- Evite textos longos que parecem um manual de instruções
- Use emojis com moderação para humanizar a conversa 
- Utilize quebras de linha para melhorar a legibilidade
- Destaque pontos importantes com *asteriscos* para negrito
- Seja conversacional como em um bate-papo real
- Faça perguntas curtas e específicas

ABORDAGEM DE VENDAS:
- PRIMEIRO ENTENDA, DEPOIS OFEREÇA: Faça perguntas para entender a necessidade antes de falar sobre o produto
- MENOS É MAIS: Forneça apenas informações relevantes para o momento da conversa
- FOCO NO CLIENTE: Concentre-se nos problemas e necessidades específicas do cliente
- LINGUAGEM SIMPLES: Evite jargões técnicos complexos
- GRADUALMENTE: Revele informações em pequenas doses, não tudo de uma vez

COMPARTILHE O LINK:
- Compartilhe o link quando identificar interesse na aquisição ou teste: https://www.appjarvis.com.br/

GATILHOS DE URGÊNCIA:
- Mencione que as vagas para o teste gratuito são *limitadas* nesta semana
- Use frases como "estamos com *promoção especial* até o final da semana"
- Destaque que o preço atual de R$50/ano é promocional e pode aumentar em breve
- Mencione que as próximas atualizações só estarão disponíveis para usuários já cadastrados
- Indique quando houver "poucas vagas" restantes para o acesso antecipado
- Crie senso de oportunidade: "Esta é uma chance única de..."
- Sugestão sutil: "Muitos usuários estão aderindo agora para garantir este valor"
- Destaque a economia: "Isso significa menos de R$0,14 por dia - *por tempo limitado*"

SEQUÊNCIA DE INTERAÇÃO:
1. Apresentação breve e amigável (apenas na primeira mensagem)
2. Pergunte sobre necessidades/desafios específicos
3. Escute atentamente antes de propor soluções
4. Relacione as características do JARVIS com os problemas mencionados
5. Ofereça valor antes de falar de preço
6. Use call-to-action simples e direto
7. Incorpore um elemento de urgência sutil e relevante ao final
8. Compartilhe o link quando o cliente mostrar interesse

SOBRE INFORMAÇÕES DO PRODUTO:
- Mencione apenas o que é relevante para a conversa atual
- Aprofunde detalhes somente quando solicitado
- Destaque benefícios, não recursos técnicos
- Teste gratuito: 7 dias sem cartão de crédito (*vagas limitadas*)
- Plano anual: R$50/ano (menos de R$4,20/mês) - *oferta por tempo limitado*
- Destaque que o JARVIS está em fase de crescimento e os preços tendem a aumentar
- Link do site: https://www.appjarvis.com.br/

IMPORTANTE: Quando o cliente fizer uma pergunta, responda diretamente e de forma concisa. Não transforme cada resposta em uma apresentação completa do produto. Construa a conversa gradualmente, como um diálogo natural de WhatsApp. Insira elementos de urgência de forma natural e relevante ao contexto, evitando parecer agressivo ou desesperado.
"""
        
        # Create augmented prompt with sales context
        augmented_prompt = f"""Use o contexto abaixo sobre o JARVIS para responder à pergunta do potencial cliente
de forma persuasiva mas conversacional, como uma conversa de WhatsApp. Seja conciso e direto.
Inclua sutilmente elementos de urgência/escassez para criar um senso de oferta por tempo limitado.
Compartilhe o link do site (https://www.appjarvis.com.br/) quando for apropriado à conversa.

Contexto sobre o JARVIS:
{sales_context}

Pergunta/objeção do cliente: {query}"""
        
        return system_prompt, augmented_prompt
    
    def _convert_messages_to_groq_format(self, messages):
        """Convert LangChain message format to Groq API format"""
        groq_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                groq_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                groq_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                groq_messages.append({"role": "assistant", "content": message.content})
        return groq_messages

    def answer_sales_query(self, query: str, conversation_history: List[Dict[str, str]] = None, k: int = 3):
        """Answer a sales query using the RAG approach with sales content"""
        try:
            # Generate sales-oriented system prompt and augmented prompt
            system_prompt, augmented_prompt = self.generate_sales_prompt(query, k)
            
            # Initialize messages with sales system prompt
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if provided
            if conversation_history:
                for entry in conversation_history:
                    if entry["role"] == "user":
                        messages.append(HumanMessage(content=entry["content"]))
                    elif entry["role"] == "assistant":
                        messages.append(AIMessage(content=entry["content"]))
            
            # Add current query with context
            messages.append(HumanMessage(content=augmented_prompt))
            
            # Convert to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages)
            
            # Invoke the model directly using Groq client
            logger.info(f"Invocando modelo Groq para venda com {len(groq_messages)} mensagens")
            completion = self.groq_client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=groq_messages,
                temperature=0.7,  # Slightly higher temperature for creative sales responses
                max_completion_tokens=1024,
                top_p=0.95,
                stream=False,
                reasoning_format="hidden"
            )
            response_content = completion.choices[0].message.content
            logger.info("Resposta de vendas recebida com sucesso")
            
            return {
                "answer": response_content,
                "augmented_prompt": augmented_prompt,
                "system_prompt": system_prompt
            }
        except Exception as e:
            logger.error(f"Erro ao responder consulta de vendas: {e}")
            return {
                "answer": f"Desculpe, ocorreu um erro ao processar sua consulta: {str(e)}",
                "augmented_prompt": "",
                "system_prompt": ""
            } 