from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from app.core.loggin import setup_logging
from app.core.config import settings
from app.api.v1.routers.chat import router as chat_router

# Configura o logging assim que a aplicação inicia
setup_logging()

# Cria a instância principal da aplicação
app = FastAPI(title=settings.app_name, version="1.0.0")

# Configura o CORS para permitir requisições de qualquer origem (ajustar em produção real)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Inclui as rotas da nossa API de chat, com um prefixo de versão
app.include_router(chat_router, prefix=settings.api_prefix)

# Define o caminho para o nosso arquivo de interface de usuário
UI_PATH = Path(__file__).parent / "ui" / "index.html"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """
    Serve a interface de usuário web (chat).
    """
    return UI_PATH.read_text(encoding="utf-8")
    
@app.get("/health", summary="Verificar a saúde da aplicação")
def health_check():
    """
    Endpoint simples para verificar se a API está no ar.
    """
    return {"status": "ok"}
