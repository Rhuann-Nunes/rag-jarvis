import os
from typing import List, Dict, Any, Optional
import logging
import pathlib
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from groq import Groq

import config

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conteúdo padrão para quando o arquivo não estiver disponível
DEFAULT_JARVIS_CONTENT = """# JARVIS: Seu Assistente Pessoal de IA

O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial, inspirado no famoso assistente do Homem de Ferro. Diferente de outros aplicativos de tarefas, o JARVIS entende linguagem natural e organiza sua vida através de uma interface intuitiva e acessível.

Com o JARVIS, você pode experimentar gratuitamente por 7 dias, sem qualquer compromisso."""


# Módulo de gerenciamento de conteúdo
class ContentManager:
    """Gerencia o carregamento do conteúdo informativo do JARVIS."""
    
    def __init__(self, content_file_path="static/jarvis_info.txt"):
        """
        Inicializa o gerenciador de conteúdo.
        
        Args:
            content_file_path (str): Caminho para o arquivo de texto com informações do JARVIS.
        """
        self.content_file_path = content_file_path
        
    def load_content(self) -> Dict:
        """
        Carrega o conteúdo do arquivo de texto.
        
        Returns:
            Dict: Conteúdo formatado com metadados.
        """
        try:
            # Tentar carregar o conteúdo do arquivo
            with open(self.content_file_path, 'r', encoding='utf-8') as f:
                content_text = f.read()
            
            content = {
                "text": content_text,
                "metadata": {
                    "source": self.content_file_path
                }
            }
            
            logger.info(f"Loaded JARVIS info from file: {self.content_file_path} ({len(content_text)} characters)")
            return content
        except Exception as e:
            logger.error(f"Error loading content from file: {str(e)}", exc_info=True)
            # Return default content to avoid cascading errors
            return {
                "text": DEFAULT_JARVIS_CONTENT,
                "metadata": {
                    "source": "default_content"
                }
            }


# Módulo de geração de prompts
class PromptGenerator:
    """Gerador de prompts para o assistente do JARVIS."""
    
    def __init__(self):
        """Inicializa o gerador de prompts."""
        # Variações de linguagem para tornar as respostas mais naturais
        self.greeting_variations = [
            "Olá! Sou o Davi, especialista do JARVIS.",
            "Oi! Aqui é o Davi, do time JARVIS.",
            "E aí! Davi do JARVIS aqui para te ajudar.",
            "Olá! Davi, do JARVIS, prazer em conhecê-lo!",
            "Oi! Davi, do time do JARVIS. Como posso ajudar?"
        ]
        
        self.closing_variations = [
            "Estou à disposição para tirar qualquer dúvida!",
            "Ficou alguma dúvida? Estou aqui para ajudar!",
            "O que você acha? Posso te ajudar com mais detalhes?",
            "Gostaria de saber mais sobre algum aspecto específico?",
            "Estou aqui para o que precisar! O que achou?"
        ]
        
        self.engagement_questions = [
            "Você costuma usar algum aplicativo para organizar suas tarefas?",
            "Quais são seus maiores desafios na organização do dia a dia?",
            "O que você acha mais difícil na gestão do seu tempo?",
            "Você já experimentou assistentes de produtividade antes?",
            "Como você organiza suas tarefas atualmente?"
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
        # Logging detalhado do histórico para diagnóstico de problemas
        if conversation_history:
            logger.info(f"Adapting to conversation stage with {len(conversation_history)} messages")
            
            # Calcular corretamente os pares de mensagens usuário/assistente
            user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
            assistant_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
            
            # Verifique se o último item é uma pergunta do usuário (par incompleto)
            is_last_user = conversation_history[-1].get('role') == 'user' if conversation_history else False
            
            # Calcule os pares completos (usuário + assistente)
            msg_pairs = min(len(user_messages), len(assistant_messages))
            if is_last_user:
                msg_pairs = max(0, min(len(user_messages) - 1, len(assistant_messages)))
                
            logger.info(f"Conversation has {len(user_messages)} user messages, {len(assistant_messages)} assistant messages, {msg_pairs} complete pairs")
        else:
            msg_pairs = 0
            logger.info("No conversation history, starting first interaction")
        
        # Estágio 0: Primeira mensagem (não há histórico)
        if not conversation_history or msg_pairs == 0:
            logger.info("Stage 0: First message - introduction and understanding needs")
            return {
                "include_greeting": True,
                "focus_on_understanding": True,
                "include_product_details": True,  # Incluir detalhes básicos do produto
                "include_trial": True,  # Mencionar período de teste gratuito
                "include_link": False,  # NÃO incluir link
                "ask_engagement_question": True,  # Fazer UMA pergunta de engajamento
                "ask_if_wants_link": False  # NÃO perguntar se quer o link
            }
        
        # Estágio 1: Segunda interação - o usuário respondeu à primeira mensagem
        elif msg_pairs == 1:
            logger.info("Stage 1: Second interaction - showing benefits and offering link")
            return {
                "include_greeting": False,
                "focus_on_understanding": False,  # Menos foco em perguntas, mais em soluções
                "include_product_details": True,  # Mostrar detalhes específicos
                "include_trial": True,  # Reforçar período de teste gratuito
                "include_link": True,  # Incluir o link
                "ask_engagement_question": False,  # Não fazer mais perguntas de engajamento
                "ask_if_wants_link": True  # Perguntar se quer o link
            }
        
        # Estágio 2: Terceira interação - o usuário mostrou mais interesse
        elif msg_pairs == 2:
            logger.info("Stage 2: Third interaction - providing link and detailed information")
            return {
                "include_greeting": False,
                "focus_on_understanding": False,
                "include_product_details": True,
                "include_trial": True,
                "include_link": True,
                "ask_engagement_question": False,
                "ask_if_wants_link": False  # Não perguntar novamente, apenas incluir o link
            }
        
        # Estágio 3+: Interações posteriores - foco na conversão
        else:
            logger.info(f"Stage {msg_pairs}: Later interaction - focus on conversion")
            return {
                "include_greeting": False,
                "focus_on_understanding": False,
                "include_product_details": True,
                "include_trial": True,
                "include_link": True,
                "ask_engagement_question": False,
                "ask_if_wants_link": False
            }
    
    def generate_system_prompt(self, conversation_history=None) -> str:
        """
        Gera o prompt do sistema para o assistente, adaptado ao estágio da conversa.
        
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
        engagement_question = self._select_random_variation(self.engagement_questions)
        
        # Construir o prompt base
        prompt = """Você é Davi, Especialista do JARVIS, um assistente pessoal de produtividade. Você deve se comunicar no estilo do WhatsApp - mensagens curtas, diretas e com boa formatação.

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
- FOQUE EM ENTENDER A NECESSIDADE: Faça UMA pergunta concreta sobre os desafios de produtividade do cliente
- DEMONSTRE EMPATIA: Valide as dificuldades mencionadas pelo cliente
- NÃO FAÇA MAIS DE UMA PERGUNTA: Isso confunde o usuário e dificulta a fluência da conversa
- APRESENTE O PRODUTO BREVEMENTE: Mencione 2-3 benefícios principais do JARVIS
- PERSONALIZE: Use o nome do cliente se ele o tiver mencionado"""
        else:
            prompt += """

NESTE MOMENTO DA CONVERSA:
- CONECTE NECESSIDADES COM SOLUÇÕES: Mostre como o JARVIS resolve os problemas mencionados
- SEJA ESPECÍFICO: Cite funcionalidades relevantes para o caso do cliente
- INCLUA EXEMPLOS: Mencione brevemente como outros usuários resolveram problemas similares
- OFEREÇA O LINK: Pergunte se o cliente quer experimentar o JARVIS gratuitamente"""

        # Adicionar o restante do prompt
        prompt += """

ABORDAGEM DE CONVERSA:
- PRIMEIRO ENTENDA, DEPOIS OFEREÇA: Faça perguntas para entender a necessidade antes de falar sobre o produto
- MENOS É MAIS: Forneça apenas informações relevantes para o momento da conversa
- FOCO NO CLIENTE: Concentre-se nos problemas e necessidades específicas do cliente
- LINGUAGEM SIMPLES: Evite jargões técnicos complexos
- GRADUALMENTE: Revele informações em pequenas doses, não tudo de uma vez

COMPARTILHE O LINK (SOMENTE A PARTIR DA SEGUNDA INTERAÇÃO):
- Compartilhe o link do JARVIS: https://www.appjarvis.com.br/login
- Momentos ideais para compartilhar o link:
  1. Quando o cliente mostrar interesse em conhecer o produto
  2. Logo após descrever um benefício que desperte curiosidade
  3. Quando o cliente perguntar como acessar ou se cadastrar
- Formate o link de forma destacada com emojis: 🚀 https://www.appjarvis.com.br/login
- Reforçe que o cadastro gratuito leva menos de 1 minuto
- NUNCA compartilhe o link na primeira mensagem!

PERÍODO DE TESTE GRATUITO:
- Mencione que todos os recursos estão disponíveis gratuitamente por 7 dias
- Destaque que não é necessário cartão de crédito para o teste
- Enfatize que o teste é completo, sem limitações de funcionalidades

SEQUÊNCIA DE INTERAÇÃO (IMPORTANTE):
1. Na primeira resposta: Apresentação breve, benefícios básicos, mencionar 7 dias grátis, UMA pergunta sobre necessidades
2. Na segunda resposta: Mostrar benefícios específicos à necessidade, oferecer o link para teste, perguntar se quer acessar
3. Nas interações seguintes: Responder dúvidas, destacar vantagens e sempre incluir o link

SOBRE INFORMAÇÕES DO PRODUTO:
- Mencione apenas o que é relevante para a conversa atual
- Aprofunde detalhes somente quando solicitado
- Destaque benefícios, não recursos técnicos
- 7 dias de teste gratuito sem cartão de crédito
- NUNCA mencione valores ou preços

IMPORTANTE: Quando o cliente fizer uma pergunta, responda diretamente e de forma concisa. Não transforme cada resposta em uma apresentação completa do produto. Construa a conversa gradualmente, como um diálogo natural de WhatsApp."""

        # Adicionar elementos de personalização
        if config["include_greeting"]:
            prompt += f"""

PARA ESTA MENSAGEM:
- Use a seguinte saudação (adaptando conforme necessário): "{greeting}"
- Termine com: "{closing}"
- Inclua esta pergunta para iniciar o diálogo (APENAS UMA): "{engagement_question}"
- NÃO compartilhe o link do site na primeira mensagem!"""

        return prompt
    
    def generate_augmented_prompt(self, query: str, jarvis_context: str, conversation_history=None) -> str:
        """
        Gera o prompt aumentado com contexto para o assistente.
        
        Args:
            query (str): Consulta do usuário.
            jarvis_context (str): Contexto completo sobre o JARVIS.
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
            specific_instructions.append("- Faça apenas UMA pergunta para entender as necessidades específicas do cliente")
            specific_instructions.append("- Evite listar muitas funcionalidades do produto")
        
        if config["include_product_details"]:
            specific_instructions.append("- Mencione apenas os benefícios do JARVIS que são relevantes para as necessidades expressas")
        
        if config["include_trial"]:
            specific_instructions.append("- Mencione o período de teste gratuito de 7 dias sem cartão de crédito")
        
        if config["include_link"]:
            specific_instructions.append("- Compartilhe o link do site: https://www.appjarvis.com.br/login")
        else:
            specific_instructions.append("- NÃO compartilhe o link do site nesta mensagem")
        
        if config["ask_engagement_question"]:
            specific_instructions.append("- Termine sua resposta com UMA ÚNICA pergunta sobre necessidades de organização ou produtividade")
        
        if config["ask_if_wants_link"]:
            specific_instructions.append("- Pergunte se o usuário quer o link para testar o JARVIS gratuitamente por 7 dias")
        
        if client_name:
            specific_instructions.append(f"- Personalize a resposta usando o nome do cliente: {client_name}")
        
        # Juntar instruções
        instructions_text = "\n".join(specific_instructions)
        
        prompt = f"""Use o contexto abaixo sobre o JARVIS para responder à pergunta do potencial cliente
de forma conversacional, como uma conversa de WhatsApp. Seja conciso e direto.

Instruções específicas para esta resposta:
{instructions_text}

Contexto sobre o JARVIS:
{jarvis_context}

Pergunta do cliente: {query}"""

        return prompt
    
    def generate_jarvis_prompt(self, query: str, jarvis_content: Dict, conversation_history=None) -> tuple:
        """
        Gera o prompt do sistema e o prompt aumentado para o assistente.
        
        Args:
            query (str): Consulta do usuário.
            jarvis_content (Dict): Conteúdo completo sobre o JARVIS.
            conversation_history (list, optional): Histórico de conversa.
            
        Returns:
            tuple: (prompt do sistema, prompt aumentado)
        """
        # Extrai o texto do conteúdo
        jarvis_context = jarvis_content["text"]
        
        # Generate prompts
        system_prompt = self.generate_system_prompt(conversation_history)
        augmented_prompt = self.generate_augmented_prompt(query, jarvis_context, conversation_history)
        
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
                    
                    # Validar o conteúdo da resposta
                    if not completion or not completion.choices or not completion.choices[0].message:
                        logger.error("Resposta vazia ou inválida do modelo")
                        return ""
                        
                    content = completion.choices[0].message.content
                    if not content or len(content.strip()) < 10:  # Resposta muito curta é suspeita
                        logger.error(f"Resposta muito curta ou vazia: '{content}'")
                        return ""
                    
                    return content
                except queue.Empty:
                    logger.error(f"LLM request timed out after {self.timeout}s")
                    return ""
                
            except Exception as e:
                error_type = type(e).__name__
                self._log_error_details(e, {"error_type": "timeout_handler"})
                logger.error(f"Erro ao processar resposta: {error_type} - {str(e)}")
                return ""
                
        except Exception as e:
            error_type = type(e).__name__
            self._log_error_details(e, {
                "messages_count": len(messages),
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            logger.error(f"Erro global ao gerar resposta: {error_type} - {str(e)}")
            return ""


# Classe principal do serviço RAG
class JARVISAssistantService:
    """Serviço de interação conversacional para informações sobre o JARVIS."""
    
    def __init__(self, llm_client=None):
        """
        Inicializa o serviço de assistente do JARVIS.
        
        Args:
            llm_client: Cliente LLM a usar.
        """
        # Iniciar componentes
        self.content_manager = ContentManager()
        self.prompt_generator = PromptGenerator()
        self.llm_service = LLMService(llm_client)
        
        # Configurações
        self.debug_mode = False  # Modo de debug desativado por padrão
        
        # Carregar conteúdo do JARVIS
        self.jarvis_content = self.load_jarvis_content()
        
        logger.info(f"JARVIS Assistant Service initialized with model: {config.GROQ_MODEL_NAME}")
    
    def load_jarvis_content(self):
        """Carrega conteúdo completo sobre o JARVIS."""
        try:
            # Carregar conteúdo completo
            jarvis_content = self.content_manager.load_content()
            
            logger.info(f"Loaded JARVIS content with {len(jarvis_content['text'])} characters")
            return jarvis_content
        except Exception as e:
            logger.error(f"Error loading JARVIS content: {str(e)}", exc_info=True)
            # Inicializar com conteúdo padrão para evitar erros em cascata
            return {
                "text": DEFAULT_JARVIS_CONTENT,
                "metadata": {
                    "source": "default_content"
                }
            }
    
    def load_content_from_file(self, file_path):
        """
        Carrega conteúdo de um arquivo texto.
        
        Args:
            file_path (str): Caminho para o arquivo de texto.
            
        Returns:
            Dict: Conteúdo formatado com metadados.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.jarvis_content = {
                "text": content,
                "metadata": {
                    "source": file_path
                }
            }
            
            logger.info(f"Loaded content from file: {file_path} ({len(content)} characters)")
            return self.jarvis_content
        except Exception as e:
            logger.error(f"Error loading content from file: {str(e)}", exc_info=True)
            # Manter o conteúdo atual em caso de erro
            return self.jarvis_content
    
    def generate_jarvis_prompt(self, query: str, conversation_history=None):
        """
        Gera prompt com base na consulta.
        
        Args:
            query (str): Consulta do usuário.
            conversation_history (list, optional): Histórico da conversa.
            
        Returns:
            tuple: (prompt do sistema, prompt aumentado)
        """
        # Gerar prompts usando todo o conteúdo
        return self.prompt_generator.generate_jarvis_prompt(
            query=query,
            jarvis_content=self.jarvis_content,
            conversation_history=conversation_history
        )
    
    def answer_query(self, query, conversation_history=None):
        """
        Processa uma pergunta e gera uma resposta.
        
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
            
            # Análise detalhada do histórico para debug
            user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
            assistant_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
            
            # Calcular pares completos (usuário + assistente)
            msg_pairs = min(len(user_messages), len(assistant_messages))
            if conversation_history and conversation_history[-1].get('role') == 'user':
                msg_pairs = max(0, min(len(user_messages) - 1, len(assistant_messages)))
            
            # Configuração baseada no estágio da conversa
            config = self.prompt_generator._adapt_to_conversation_stage(conversation_history)
            logger.info(f"Processing query with conversation stage: {msg_pairs} message pairs, config: {config}")
            
            # Validação rigorosa de entrada
            if not query or len(query) < 2:
                logger.info("Consulta vazia ou muito curta - retornando empty_query")
                return {"response": "", "empty_query": True}
            
            # Verificação de repetição - verificar se a última pergunta é igual à atual
            if user_messages and user_messages[-1].get('content', '').lower() == query.lower():
                logger.warning(f"Usuário repetiu a mesma pergunta: '{query}'")
            
            # Verificação adicional para evitar processamento de consultas problemáticas
            # Apenas para primeira interação (sem histórico)
            if not conversation_history and query.lower() in ["oi", "olá", "teste", "ola", "hi", "hello"]:
                # Para saudações simples, responder de forma leve sem processamento complexo
                logger.info("Saudação simples detectada em primeira interação - usando resposta padrão")
                greeting_response = "Olá! 👋 Sou Davi, Especialista do JARVIS. Como posso ajudar você hoje? Você costuma usar algum aplicativo para organizar suas tarefas?"
                return {"response": greeting_response}
            
            # Detectando o nome do cliente (opcional)
            client_name = ""
            if conversation_history and len(conversation_history) > 0:
                for message in conversation_history:
                    if message["role"] == "user" and "meu nome é" in message["content"].lower():
                        name_parts = message["content"].lower().split("meu nome é")[1].strip().split()
                        if name_parts:
                            client_name = name_parts[0].capitalize()
                            logger.info(f"Nome do cliente detectado: {client_name}")
                            break
            
            # Gerando o prompt do sistema e o prompt aumentado
            system_prompt, augmented_prompt = self.generate_jarvis_prompt(
                query=query,
                conversation_history=conversation_history
            )
            
            # Log dos prompts gerados para debug
            logger.info(f"System prompt length: {len(system_prompt)}, preview: {system_prompt[:100]}...")
            logger.info(f"Augmented prompt length: {len(augmented_prompt)}, preview: {augmented_prompt[:100]}...")
            
            # Inicializar mensagens com prompt do sistema
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            
            messages = [SystemMessage(content=system_prompt)]
            
            # Adicionar histórico de conversa, verificando a consistência dos pares
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
            
            # Validação extra da resposta gerada
            if not response_content or len(response_content.strip()) < 20:
                logger.warning("Resposta muito curta ou vazia retornada pelo modelo")
                return {"response": "", "empty_query": True}
            
            # Verificar se a resposta é similar à última resposta do assistente (repetição)
            if assistant_messages and response_content.strip() == assistant_messages[-1].get('content', '').strip():
                logger.warning("Modelo repetiu a última resposta - tentando gerar uma nova")
                
                # Adicionar uma nota para forçar variação na resposta
                messages.append(SystemMessage(content="IMPORTANTE: Forneça uma resposta diferente da anterior, mas mantenha o mesmo conteúdo informativo. Use palavras e estrutura diferentes."))
                messages.append(HumanMessage(content=query))
                
                # Tentar gerar novamente com temperatura mais alta
                response_content = self.llm_service.generate_response(
                    messages=messages,
                    temperature=0.9,
                    max_tokens=1500
                )
            
            # Retornar informações adicionais para debug
            return {
                "response": response_content,
                "conversation_stage": msg_pairs,
                "augmented_prompt": augmented_prompt[:500] + "..." if len(augmented_prompt) > 500 else augmented_prompt
            }
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}", exc_info=True)
            # Não retornar mensagens de erro para o usuário
            return {"response": "", "empty_query": True} 