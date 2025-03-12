import os
import shutil
import sys

def ensure_sales_content_file():
    """
    Garante que o arquivo jarvis_sales.md existe no diretório correto.
    """
    # Obtém o diretório atual
    current_dir = os.getcwd()
    print(f"Diretório atual: {current_dir}")
    
    # Verifica se o diretório static existe
    static_dir = os.path.join(current_dir, "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        print(f"Diretório 'static' criado em {static_dir}")
    
    # Caminho completo para o arquivo de destino
    target_file = os.path.join(static_dir, "jarvis_sales.md")
    
    # Verifica se o arquivo já existe
    if os.path.exists(target_file):
        file_size = os.path.getsize(target_file)
        print(f"Arquivo já existe: {target_file} (Tamanho: {file_size} bytes)")
        
        # Verifica se o arquivo tem conteúdo
        if file_size == 0:
            print("Arquivo está vazio. Será recriado.")
        else:
            print("Arquivo parece válido.")
            return True
    
    # Conteúdo padrão para o arquivo
    content = """# JARVIS: Seu Assistente Pessoal de IA

## O que é o JARVIS?

O JARVIS é um assistente pessoal de produtividade alimentado por inteligência artificial, inspirado no famoso assistente do Homem de Ferro. Diferente de outros aplicativos de tarefas, o JARVIS entende linguagem natural e organiza sua vida através de uma interface intuitiva e acessível de qualquer dispositivo.

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
"""
    
    # Escreve o conteúdo no arquivo
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Arquivo criado com sucesso: {target_file}")
        return True
    except Exception as e:
        print(f"Erro ao criar arquivo: {str(e)}")
        return False

if __name__ == "__main__":
    success = ensure_sales_content_file()
    sys.exit(0 if success else 1) 