import os
from typing import List, Dict, Any, Optional
import logging
import pathlib
from datetime import datetime

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

O JARVIS utiliza modelos avançados de IA (GPT-4o) para compreender suas tarefas exatamente como você as descreve.

### 📅 Gestão Inteligente de Tarefas

O sistema organiza suas tarefas automaticamente com base em datas inteligentes, recorrências e categorização.

### 🔔 Notificações Que Funcionam

Alertas no dispositivo, notificações via WhatsApp e priorização inteligente.

### 🤖 JARVIS no WhatsApp

Acesso total aos seus dados, assistente proativo, cria e gerencia tarefas.

## Segurança e Privacidade

Autenticação segura, armazenamento criptografado e controle total dos dados.

## Comece Agora - Vagas Limitadas!

**[🚀 Experimente o JARVIS GRATUITAMENTE: https://www.appjarvis.com.br/](https://www.appjarvis.com.br/)**

## Planos Acessíveis

- 7 Dias Grátis – Experimente todas as funcionalidades sem cartão de crédito
- Plano Anual – Apenas R$ 50,00 por um ano inteiro de produtividade transformadora"""


# Módulo de gerenciamento de conteúdo
class ContentManager:
    """Gerencia o carregamento e processamento do conteúdo de vendas."""
    
    def __init__(self, content_text=None, chunk_size=1000, chunk_overlap=200):
        """
        Inicializa o gerenciador de conteúdo.
        
        Args:
            content_text (str, optional): Texto do conteúdo de vendas. Usa o conteúdo embutido se None.
            chunk_size (int): Tamanho de cada fragmento para o text splitter.
            chunk_overlap (int): Sobreposição entre fragmentos para o text splitter.
        """
        self.content_text = content_text or JARVIS_SALES_CONTENT
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
    def load_content(self) -> List[Document]:
        """
        Carrega e processa o conteúdo de vendas.
        
        Returns:
            List[Document]: Lista de documentos processados.
        """
        try:
            # Processar o conteúdo como um documento
            document = Document(page_content=self.content_text)
            documents = [document]
            
            # Split the document into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to identify sections
            for i, doc in enumerate(chunks):
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
            
            logger.info(f"Processed content into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing content: {str(e)}", exc_info=True)
            # Return default content to avoid cascading errors
            default_doc = Document(
                page_content="JARVIS: Seu Assistente Pessoal de IA. O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial.",
                metadata={"section": "Geral", "chunk_id": 0, "source": "default_content"}
            )
            return [default_doc]


# Módulo de embedding
class EmbeddingService:
    """Serviço para geração e gestão de embeddings."""
    
    def __init__(self, embedding_model=None):
        """
        Inicializa o serviço de embeddings.
        
        Args:
            embedding_model: Modelo de embeddings a ser usado. Se None, usa OpenAIEmbeddings padrão.
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        )
        # Cache para armazenar embeddings calculados
        self.embedding_cache = {}
        
    def get_embeddings_for_documents(self, documents: List[Document]) -> Dict[int, List[float]]:
        """
        Gera embeddings para uma lista de documentos.
        
        Args:
            documents (List[Document]): Lista de documentos para gerar embeddings.
            
        Returns:
            Dict[int, List[float]]: Dicionário de embeddings por índice de documento.
        """
        embeddings = {}
        for i, doc in enumerate(documents):
            # Usar cache se disponível
            cache_key = hash(doc.page_content)
            if cache_key in self.embedding_cache:
                embeddings[i] = self.embedding_cache[cache_key]
                continue
                
            # Calcular novo embedding
            embedding = self.embedding_model.embed_documents([doc.page_content])[0]
            self.embedding_cache[cache_key] = embedding
            embeddings[i] = embedding
            
        return embeddings
        
    def embed_query(self, query: str) -> List[float]:
        """
        Gera embedding para uma consulta.
        
        Args:
            query (str): Texto da consulta.
            
        Returns:
            List[float]: Embedding da consulta.
        """
        # Usar cache se disponível
        cache_key = hash(query)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        # Calcular novo embedding
        embedding = self.embedding_model.embed_query(query)
        self.embedding_cache[cache_key] = embedding
        return embedding


# Módulo de busca semântica
class SemanticSearchService:
    """Serviço para busca semântica em documentos."""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Inicializa o serviço de busca semântica.
        
        Args:
            embedding_service (EmbeddingService): Serviço de embeddings a ser usado.
        """
        self.embedding_service = embedding_service
        # Cache para resultados de busca
        self.search_cache = {}
        # Tamanho máximo do cache
        self.max_cache_size = 100
        # Contador de frequência de consultas
        self.query_frequency = {}
        
    def _normalize_query(self, query: str) -> str:
        """
        Normaliza uma consulta para uso como chave de cache.
        
        Args:
            query (str): Consulta original.
            
        Returns:
            str: Consulta normalizada.
        """
        # Normalizar para lowercase e remover espaços extras
        normalized = query.lower().strip()
        # Remover pontuação comum
        for char in ".,;:!?":
            normalized = normalized.replace(char, "")
        return normalized
        
    def _get_cache_key(self, query: str, k: int) -> str:
        """
        Gera uma chave de cache para uma consulta e k.
        
        Args:
            query (str): Consulta normalizada.
            k (int): Número de resultados.
            
        Returns:
            str: Chave de cache.
        """
        return f"{query}_{k}"
    
    def _update_query_frequency(self, query: str):
        """
        Atualiza a frequência de uma consulta.
        
        Args:
            query (str): Consulta normalizada.
        """
        if query in self.query_frequency:
            self.query_frequency[query] += 1
        else:
            self.query_frequency[query] = 1
    
    def _clean_cache_if_needed(self):
        """
        Limpa o cache se ele exceder o tamanho máximo.
        """
        if len(self.search_cache) <= self.max_cache_size:
            return
            
        # Ordenar consultas por frequência
        sorted_queries = sorted(
            self.query_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Manter apenas as consultas mais frequentes
        queries_to_keep = set([q for q, _ in sorted_queries[:self.max_cache_size // 2]])
        
        # Remover do cache as consultas menos frequentes
        new_cache = {}
        for cache_key, results in self.search_cache.items():
            query = cache_key.split("_")[0]
            if query in queries_to_keep:
                new_cache[cache_key] = results
        
        # Atualizar o cache e o contador de frequência
        self.search_cache = new_cache
        self.query_frequency = {q: f for q, f in self.query_frequency.items() if q in queries_to_keep}
        
        logger.info(f"Cache cleaned. Kept {len(self.search_cache)} entries based on frequency.")
    
    def search(self, query: str, documents: List[Document], doc_embeddings: Dict[int, List[float]], k: int = 3):
        """
        Realiza busca semântica em documentos.
        
        Args:
            query (str): Consulta para busca.
            documents (List[Document]): Lista de documentos para buscar.
            doc_embeddings (Dict[int, List[float]]): Dicionário de embeddings por índice de documento.
            k (int): Número de resultados a retornar.
            
        Returns:
            List[Dict]: Lista de resultados com score, texto e metadados.
        """
        if not documents:
            return []
        
        # Normalizar query e gerar chave de cache
        normalized_query = self._normalize_query(query)
        cache_key = self._get_cache_key(normalized_query, k)
        
        # Atualizar frequência
        self._update_query_frequency(normalized_query)
        
        # Verificar cache
        if cache_key in self.search_cache:
            logger.info(f"Cache hit for query: {query}")
            return self.search_cache[cache_key]
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Preparar lista de embeddings a partir do dicionário
        doc_embedding_list = [doc_embeddings[i] for i in range(len(documents))]
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], doc_embedding_list)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "text": documents[idx].page_content,
                "metadata": documents[idx].metadata
            })
        
        # Armazenar no cache
        self.search_cache[cache_key] = results
        
        # Limpar cache se necessário
        self._clean_cache_if_needed()
        
        return results


# Módulo de geração de prompts
class PromptGenerator:
    """Gerador de prompts para o chatbot de vendas."""
    
    def __init__(self):
        """Inicializa o gerador de prompts."""
        # Variações de linguagem para tornar as respostas mais naturais
        self.greeting_variations = [
            "Olá! Sou o Davi, gerente comercial do JARVIS.",
            "Oi! Aqui é o Davi, do time JARVIS.",
            "E aí! Davi do JARVIS aqui para te ajudar.",
            "Olá! Davi Cardoso, do JARVIS, prazer em conhecê-lo!",
            "Oi! Davi, do time comercial do JARVIS. Como posso ajudar?"
        ]
        
        self.closing_variations = [
            "Estou à disposição para tirar qualquer dúvida!",
            "Ficou alguma dúvida? Estou aqui para ajudar!",
            "O que você acha? Posso te ajudar com mais detalhes?",
            "Gostaria de saber mais sobre algum aspecto específico?",
            "Estou aqui para o que precisar! O que achou?"
        ]
        
        self.urgency_phrases = [
            "As vagas para teste gratuito são limitadas nesta semana!",
            "Estamos com uma promoção especial até o final da semana.",
            "O preço atual é promocional e vai aumentar em breve.",
            "As próximas atualizações estarão disponíveis apenas para usuários já cadastrados.",
            "Temos poucas vagas restantes para o acesso antecipado.",
            "Esta é uma chance única de garantir o valor atual.",
            "Muitos usuários estão aderindo agora para garantir este preço.",
            "Isso significa menos de R$0,14 por dia - por tempo limitado!"
        ]
    
    def _select_random_variation(self, variations_list):
        """
        Seleciona uma variação aleatória de texto.
        
        Args:
            variations_list (list): Lista de variações.
            
        Returns:
            str: Variação selecionada.
        """
        import random
        return random.choice(variations_list)
    
    def _adapt_to_conversation_stage(self, conversation_history=None):
        """
        Adapta o prompt ao estágio da conversa.
        
        Args:
            conversation_history (list, optional): Histórico de conversa.
            
        Returns:
            dict: Configurações adaptadas.
        """
        if not conversation_history:
            # Primeira mensagem - foco em apresentação e entendimento
            return {
                "include_greeting": True,
                "focus_on_understanding": True,
                "include_product_details": False,
                "include_pricing": False,
                "include_urgency": False,
                "include_link": False
            }
        
        msg_count = len(conversation_history) // 2  # Par de mensagens (user/assistant)
        
        if msg_count == 1:
            # Segunda interação - entender necessidades, mencionar benefícios
            return {
                "include_greeting": False,
                "focus_on_understanding": True,
                "include_product_details": True,
                "include_pricing": False,
                "include_urgency": False,
                "include_link": False
            }
        elif msg_count == 2:
            # Terceira interação - detalhar solução, mencionar preço
            return {
                "include_greeting": False,
                "focus_on_understanding": False,
                "include_product_details": True,
                "include_pricing": True,
                "include_urgency": True,
                "include_link": True
            }
        else:
            # Interações posteriores - foco em conversão, urgência
            return {
                "include_greeting": False,
                "focus_on_understanding": False,
                "include_product_details": True,
                "include_pricing": True,
                "include_urgency": True,
                "include_link": True
            }
    
    def generate_system_prompt(self, conversation_history=None) -> str:
        """
        Gera o prompt do sistema para o chatbot de vendas, adaptado ao estágio da conversa.
        
        Args:
            conversation_history (list, optional): Histórico de conversa.
            
        Returns:
            str: Prompt do sistema.
        """
        # Adaptar ao estágio da conversa
        config = self._adapt_to_conversation_stage(conversation_history)
        
        # Selecionar variações para uso neste prompt
        greeting = self._select_random_variation(self.greeting_variations)
        closing = self._select_random_variation(self.closing_variations)
        urgency_phrase = self._select_random_variation(self.urgency_phrases)
        
        # Construir o prompt base
        prompt = """Você é Davi Cardoso, Gerente Comercial do JARVIS. Você deve se comunicar no estilo do WhatsApp - mensagens curtas, diretas e com boa formatação.

ESTILO DE COMUNICAÇÃO NO WHATSAPP:
- Use mensagens curtas e objetivas (2-4 parágrafos no máximo)
- Evite textos longos que parecem um manual de instruções
- Use emojis com moderação para humanizar a conversa 
- Utilize quebras de linha para melhorar a legibilidade
- Destaque pontos importantes com *asteriscos* para negrito
- Seja conversacional como em um bate-papo real
- Faça perguntas curtas e específicas"""

        # Adicionar seção específica para o estágio atual
        if config["focus_on_understanding"]:
            prompt += """

NESTE MOMENTO DA CONVERSA:
- FOQUE EM ENTENDER A NECESSIDADE: Faça perguntas sobre os desafios de produtividade do cliente
- DEMONSTRE EMPATIA: Valide as dificuldades mencionadas pelo cliente
- EVITE FALAR MUITO DO PRODUTO: Apenas mencione benefícios diretamente relacionados às necessidades expressas
- PERSONALIZE: Use o nome do cliente se ele o tiver mencionado"""
        else:
            prompt += """

NESTE MOMENTO DA CONVERSA:
- CONECTE NECESSIDADES COM SOLUÇÕES: Mostre como o JARVIS resolve os problemas mencionados
- SEJA ESPECÍFICO: Cite funcionalidades relevantes para o caso do cliente
- INCLUA HISTÓRIAS DE SUCESSO: Mencione brevemente como outros usuários resolveram problemas similares
- MANTENHA O DIÁLOGO: Continue fazendo perguntas para ajustar sua resposta"""

        # Adicionar o restante do prompt
        prompt += """

ABORDAGEM DE VENDAS:
- PRIMEIRO ENTENDA, DEPOIS OFEREÇA: Faça perguntas para entender a necessidade antes de falar sobre o produto
- MENOS É MAIS: Forneça apenas informações relevantes para o momento da conversa
- FOCO NO CLIENTE: Concentre-se nos problemas e necessidades específicas do cliente
- LINGUAGEM SIMPLES: Evite jargões técnicos complexos
- GRADUALMENTE: Revele informações em pequenas doses, não tudo de uma vez

COMPARTILHE O LINK:
- Sempre compartilhe o link da plataforma: https://www.appjarvis.com.br/
- Momentos ideais para compartilhar o link:
  1. Quando o cliente mostrar interesse em conhecer o produto
  2. Logo após descrever um benefício que desperte curiosidade
  3. Quando o cliente perguntar como acessar ou se cadastrar
  4. No final da conversa como call-to-action
- Formate o link de forma destacada com emojis: 🚀 https://www.appjarvis.com.br/
- Reforçe que o cadastro gratuito leva menos de 1 minuto

GATILHOS DE URGÊNCIA:
- Use a seguinte frase de urgência em algum momento da resposta: "{urgency_phrase}"
- Mencione que as vagas para o teste gratuito são *limitadas* nesta semana
- Destaque que o preço atual de R$50/ano é promocional e pode aumentar em breve
- Crie senso de oportunidade: "Esta é uma chance única de..."

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

IMPORTANTE: Quando o cliente fizer uma pergunta, responda diretamente e de forma concisa. Não transforme cada resposta em uma apresentação completa do produto. Construa a conversa gradualmente, como um diálogo natural de WhatsApp. Insira elementos de urgência de forma natural e relevante ao contexto, evitando parecer agressivo ou desesperado."""

        # Adicionar elementos de personalização
        if config["include_greeting"]:
            prompt += f"""

PARA ESTA MENSAGEM:
- Use a seguinte saudação (adaptando conforme necessário): "{greeting}"
- Termine com: "{closing}"
- {'Inclua esta frase de urgência em algum momento: "' + urgency_phrase + '"' if config["include_urgency"] else ''}
- {'Compartilhe o link do site em algum momento da conversa.' if config["include_link"] else ''}"""

        return prompt
    
    def generate_augmented_prompt(self, query: str, sales_context: str, conversation_history=None) -> str:
        """
        Gera o prompt aumentado com contexto para o chatbot de vendas.
        
        Args:
            query (str): Consulta do usuário.
            sales_context (str): Contexto de vendas relevante.
            conversation_history (list, optional): Histórico de conversa.
            
        Returns:
            str: Prompt aumentado.
        """
        # Adaptar ao estágio da conversa
        config = self._adapt_to_conversation_stage(conversation_history)
        
        # Extrair nome do cliente do histórico, se disponível
        client_name = ""
        if conversation_history and len(conversation_history) > 0:
            for message in conversation_history:
                if message["role"] == "user" and "meu nome é" in message["content"].lower():
                    name_parts = message["content"].lower().split("meu nome é")[1].strip().split()
                    if name_parts:
                        client_name = name_parts[0].capitalize()
                        break
        
        # Construir instruções específicas
        specific_instructions = []
        
        if config["focus_on_understanding"]:
            specific_instructions.append("- Faça perguntas para entender melhor as necessidades específicas do cliente")
            specific_instructions.append("- Evite listar muitas funcionalidades do produto")
        
        if config["include_product_details"]:
            specific_instructions.append("- Mencione apenas os benefícios do JARVIS que são relevantes para as necessidades expressas")
        
        if config["include_pricing"]:
            specific_instructions.append("- Mencione o período de teste gratuito e o valor promocional")
        
        if config["include_urgency"]:
            specific_instructions.append("- Inclua um elemento de urgência para incentivar ação imediata")
        
        if config["include_link"]:
            specific_instructions.append("- Compartilhe o link do site: https://www.appjarvis.com.br/")
        
        if client_name:
            specific_instructions.append(f"- Personalize a resposta usando o nome do cliente: {client_name}")
        
        # Juntar instruções
        instructions_text = "\n".join(specific_instructions)
        
        prompt = f"""Use o contexto abaixo sobre o JARVIS para responder à pergunta do potencial cliente
de forma persuasiva mas conversacional, como uma conversa de WhatsApp. Seja conciso e direto.

Instruções específicas para esta resposta:
{instructions_text}

Contexto sobre o JARVIS:
{sales_context}

Pergunta/objeção do cliente: {query}"""

        return prompt
    
    def generate_sales_prompt(self, query: str, sales_results: List[Dict], conversation_history=None) -> tuple:
        """
        Gera o prompt do sistema e o prompt aumentado para o chatbot de vendas.
        
        Args:
            query (str): Consulta do usuário.
            sales_results (List[Dict]): Resultados da busca semântica.
            conversation_history (list, optional): Histórico de conversa.
            
        Returns:
            tuple: (prompt do sistema, prompt aumentado)
        """
        # Extract text from results
        sales_context = "\n".join([result["text"] for result in sales_results])
        
        # Generate prompts
        system_prompt = self.generate_system_prompt(conversation_history)
        augmented_prompt = self.generate_augmented_prompt(query, sales_context, conversation_history)
        
        return system_prompt, augmented_prompt


# Módulo de LLM
class LLMService:
    """Serviço para interação com o modelo de linguagem grande."""
    
    def __init__(self, llm_client=None, timeout=30, max_retries=2):
        """
        Inicializa o serviço de LLM.
        
        Args:
            llm_client: Cliente para o modelo de linguagem grande. Se None, usa Groq padrão.
            timeout (int): Tempo limite para requisições em segundos.
            max_retries (int): Número máximo de tentativas em caso de falha.
        """
        self.llm_client = llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_responses = {
            "error_timeout": "Desculpe, o tempo de resposta excedeu o limite. Poderia reformular sua pergunta de forma mais específica?",
            "error_api": "Estamos enfrentando dificuldades técnicas momentâneas. Por favor, tente novamente em alguns instantes.",
            "error_unknown": "Ocorreu um erro inesperado. Nossa equipe foi notificada e está trabalhando para resolver."
        }
        
    def convert_messages_to_groq_format(self, messages):
        """
        Converte mensagens do formato LangChain para o formato Groq.
        
        Args:
            messages: Lista de mensagens no formato LangChain.
            
        Returns:
            List[Dict]: Lista de mensagens no formato Groq.
        """
        groq_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                groq_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                groq_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                groq_messages.append({"role": "assistant", "content": message.content})
        return groq_messages
    
    def _log_error_details(self, error, context=None):
        """
        Registra detalhes de erro para diagnóstico.
        
        Args:
            error (Exception): Erro ocorrido.
            context (dict, optional): Contexto adicional sobre o erro.
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        error_details = {
            "error_type": error_type,
            "error_message": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            error_details["context"] = context
            
        logger.error(f"LLM Error: {error_type} - {error_msg}", extra={"error_details": error_details})
        
    def _execute_with_retries(self, func, *args, **kwargs):
        """
        Executa uma função com retentativas.
        
        Args:
            func: Função a ser executada.
            *args, **kwargs: Argumentos para a função.
            
        Returns:
            Resultado da função.
            
        Raises:
            Exception: Se todas as tentativas falharem.
        """
        import time
        
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                if retries > 0:
                    # Espera exponencial entre tentativas
                    time.sleep(2 ** retries)
                    logger.info(f"Retry {retries}/{self.max_retries} for LLM request")
                
                # Executar a função diretamente (sem async)
                return func(*args, **kwargs)
                    
            except Exception as e:
                last_error = e
                retries += 1
                self._log_error_details(e, {"retry_count": retries})
        
        # Se chegou aqui, todas as tentativas falharam
        error_msg = f"All {self.max_retries + 1} attempts failed"
        logger.error(error_msg, exc_info=last_error)
        raise last_error
        
    def generate_response(self, messages, temperature=0.7, max_tokens=1024):
        """
        Gera uma resposta a partir de mensagens.
        
        Args:
            messages: Lista de mensagens no formato LangChain.
            temperature (float): Temperatura para geração.
            max_tokens (int): Número máximo de tokens.
            
        Returns:
            str: Resposta gerada.
        """
        try:
            # Convert to Groq format
            groq_messages = self.convert_messages_to_groq_format(messages)
            
            # Define a função para invocar o modelo
            def invoke_model():
                return self.llm_client.chat.completions.create(
                    model=config.GROQ_MODEL_NAME,
                    messages=groq_messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=0.95,
                    stream=False,
                    reasoning_format="hidden"
                )
            
            # Execute com timeout e retries
            try:
                # Executar com timeout
                import threading
                import queue
                
                # Usar uma fila para obter o resultado ou exceção
                result_queue = queue.Queue()
                
                def execute_with_queue():
                    try:
                        result = self._execute_with_retries(invoke_model)
                        result_queue.put(("success", result))
                    except Exception as e:
                        result_queue.put(("error", e))
                
                # Criar thread para executar a chamada
                thread = threading.Thread(target=execute_with_queue)
                thread.daemon = True
                thread.start()
                
                # Esperar o resultado com timeout
                try:
                    status, result = result_queue.get(timeout=self.timeout)
                    if status == "error":
                        raise result
                    completion = result
                    return completion.choices[0].message.content
                except queue.Empty:
                    logger.error(f"LLM request timed out after {self.timeout}s")
                    return self.fallback_responses["error_timeout"]
                
            except Exception as e:
                error_type = type(e).__name__
                self._log_error_details(e, {"error_type": "timeout_handler"})
                if "timeout" in error_type.lower():
                    return self.fallback_responses["error_timeout"]
                else:
                    raise e
                
        except Exception as e:
            error_type = type(e).__name__
            self._log_error_details(e, {
                "messages_count": len(messages),
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            # Fornecer respostas de fallback específicas
            if "timeout" in error_type.lower():
                return self.fallback_responses["error_timeout"]
            elif any(term in str(e).lower() for term in ["api", "key", "auth", "credential"]):
                return self.fallback_responses["error_api"]
            else:
                return f"{self.fallback_responses['error_unknown']} ({error_type})"


# Classe principal do serviço RAG
class SalesRAGService:
    """Serviço de RAG para conteúdo de vendas do JARVIS."""
    
    def __init__(self, embedding_model=None, llm_client=None):
        """
        Inicializa o serviço de RAG para vendas.
        
        Args:
            embedding_model: Modelo de embeddings a usar.
            llm_client: Cliente LLM a usar.
        """
        # Iniciar componentes
        self.content_manager = ContentManager(JARVIS_SALES_CONTENT)
        self.embedding_service = EmbeddingService(embedding_model)
        self.search_service = SemanticSearchService(self.embedding_service)
        self.prompt_generator = PromptGenerator()
        self.llm_service = LLMService(llm_client)
        
        # Configurações
        self.top_k_results = 3  # Número padrão de resultados a retornar
        self.debug_mode = False  # Modo de debug desativado por padrão
        
        # Carregar conteúdo de vendas
        self.load_sales_content()
        
        logger.info(f"Sales RAG Service initialized with model: {config.GROQ_MODEL_NAME}")
    
    def load_sales_content(self):
        """Carrega conteúdo de vendas e gera embeddings."""
        try:
            # Carregar e processar conteúdo
            self.sales_docs = self.content_manager.load_content()
            
            # Gerar embeddings para todos os documentos
            self.doc_embeddings = self.embedding_service.get_embeddings_for_documents(self.sales_docs)
            
            logger.info(f"Loaded {len(self.sales_docs)} sales content chunks with embeddings")
            return {"status": "success", "message": f"Loaded {len(self.sales_docs)} sales content chunks"}
        except Exception as e:
            logger.error(f"Error loading sales content: {str(e)}", exc_info=True)
            # Inicializar com conteúdo padrão para evitar erros em cascata
            self.sales_docs = [
                Document(
                    page_content="JARVIS: Seu Assistente Pessoal de IA. O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial.",
                    metadata={"section": "Geral", "chunk_id": 0, "source": "default_content"}
                )
            ]
            self.doc_embeddings = {0: self.embedding_service.embed_query("JARVIS: Seu Assistente Pessoal de IA")}
            return {"status": "error", "message": f"Error loading sales content: {str(e)}"}
    
    def search_sales_content(self, query: str, k: int = 3):
        """
        Busca conteúdo de vendas relevante com base na consulta.
        
        Args:
            query (str): Consulta do usuário.
            k (int): Número de resultados a retornar.
            
        Returns:
            List[Dict]: Lista de resultados relevantes.
        """
        return self.search_service.search(query, self.sales_docs, self.doc_embeddings, k)
    
    def generate_sales_prompt(self, query: str, k: int = 3, conversation_history=None):
        """
        Gera prompt de vendas persuasivo com base na consulta.
        
        Args:
            query (str): Consulta do usuário.
            k (int): Número de documentos de contexto a usar.
            conversation_history (list, optional): Histórico da conversa.
            
        Returns:
            tuple: (prompt do sistema, prompt aumentado)
        """
        # Buscar conteúdo relevante
        sales_results = self.search_sales_content(query, k=k)
        
        # Gerar prompts
        return self.prompt_generator.generate_sales_prompt(query, sales_results, conversation_history)
    
    def answer_sales_query(self, query, conversation_history=None):
        """
        Processa uma pergunta de vendas e gera uma resposta.
        
        Args:
            query (str): Pergunta do usuário.
            conversation_history (list, optional): Histórico de conversação.
            
        Returns:
            dict: Resposta formatada.
        """
        try:
            # Normalizando a query e dados
            query = query.strip() if query else ""
            conversation_history = conversation_history or []
            
            if not query:
                return {"response": "Parece que você não enviou uma mensagem. Como posso ajudar?"}
            
            # Obtendo os dados relevantes através de pesquisa semântica
            search_results = self.search_service.search(
                query=query,
                documents=self.sales_docs,
                doc_embeddings=self.doc_embeddings,
                k=self.top_k_results
            )
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
            
            # Detectando o nome do cliente (opcional)
            client_name = ""
            if conversation_history and len(conversation_history) > 0:
                for message in conversation_history:
                    if message["role"] == "user" and "meu nome é" in message["content"].lower():
                        name_parts = message["content"].lower().split("meu nome é")[1].strip().split()
                        if name_parts:
                            client_name = name_parts[0].capitalize()
                            break
            
            # Gerando o prompt do sistema e o prompt aumentado
            system_prompt, augmented_prompt = self.prompt_generator.generate_sales_prompt(
                query=query,
                sales_results=search_results,
                conversation_history=conversation_history
            )
            
            # Inicializar mensagens com prompt do sistema
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            
            messages = [SystemMessage(content=system_prompt)]
            
            # Adicionar histórico de conversa
            if conversation_history:
                for entry in conversation_history:
                    if entry["role"] == "user":
                        messages.append(HumanMessage(content=entry["content"]))
                    elif entry["role"] == "assistant":
                        messages.append(AIMessage(content=entry["content"]))
            
            # Adicionar consulta atual com contexto
            messages.append(HumanMessage(content=augmented_prompt))
            
            # Gerando a resposta final usando o modelo de linguagem
            response_content = self.llm_service.generate_response(
                messages=messages,
                temperature=0.8,
                max_tokens=1500
            )
            
            return {
                "response": response_content,
                "search_results": search_results if self.debug_mode else None
            }
            
        except Exception as e:
            logger.error(f"Error answering sales query: {str(e)}", exc_info=True)
            return {
                "response": "Desculpe, estou enfrentando algumas dificuldades técnicas no momento. Poderia tentar novamente em alguns instantes?",
                "error": str(e) if self.debug_mode else None
            } 