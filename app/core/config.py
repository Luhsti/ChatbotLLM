from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Define o diretório base do projeto (duas pastas acima deste arquivo)
BASE_DIR = Path(__file__).resolve().parents[2]

# Garante que o .env da raiz do projeto seja carregado
load_dotenv(BASE_DIR / ".env")

class Settings(BaseModel):
    """Configurações da aplicação lidas de variáveis de ambiente."""

    app_name: str = "Chatbot de FAQ com RAG"
    api_prefix: str = "/api/v1"
    
    # Caminho para o banco de dados vetorial
    vector_db_path: str = str(BASE_DIR / "vectorstore" / "db_faiss")

    # Tenta ler o token de duas variáveis de ambiente comuns
    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Flag para decidir qual LLM usar
    use_local_llm: bool = True  # Mude para False se quiser usar um endpoint do Hugging Face

# 🔹 Cria a instância única das configurações para ser usada em toda a aplicação
settings = Settings()

# 🔹 Padroniza o nome da variável de ambiente esperada pelo Hugging Face
if settings.hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.hf_token
