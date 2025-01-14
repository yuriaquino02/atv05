# Construção de um Sistema de Recuperação de Informações usando VectorDB e RetrievalQA

Este projeto demonstra um pipeline simples para extrair texto de uma página web, processá-lo em embeddings vetoriais usando FAISS e responder a perguntas utilizando um sistema de perguntas e respostas com recuperação, alimentado pelo Ollama LLM.

## Pré-requisitos

Certifique-se de ter os seguintes itens instalados:
- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

## Instalação

1. Clone o repositório ou baixe os arquivos do projeto.
2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows
   ```
3. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```

### Dependências Necessárias
O arquivo `requirements.txt` deve conter o seguinte:
```plaintext
langchain-community
langchain-ollama
langchain
faiss-cpu
beautifulsoup4
requests
```


## Explicação do Workflow

1. **Web Scraping:**
   - O script obtém o conteúdo HTML da URL fornecida.
   - A biblioteca `BeautifulSoup` é usada para extrair o texto do HTML.

2. **Divisão do Texto:**
   - O texto extraído é dividido em chunks menores usando `RecursiveCharacterTextSplitter` para facilitar o processamento.

3. **Embeddings Vetoriais:**
   - Os chunks são convertidos em embeddings vetoriais usando FAISS com o modelo `mxbai-embed-large`.

4. **Sistema de Perguntas e Respostas:**
   - Os embeddings são armazenados em um banco de vetores FAISS.
   - Uma pergunta é processada e combinada com os embeddings armazenados para encontrar contextos similares.
   - O `OllamaLLM` é usado para gerar uma resposta detalhada com base no contexto recuperado.

