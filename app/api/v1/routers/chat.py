from fastapi import APIRouter, HTTPException
from app.models.schemas import Pergunta, Resposta
from app.services.rag import build_chain
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# A cadeia RAG é construída uma única vez quando a aplicação inicia
qa_chain = build_chain()

@router.post("/perguntar", response_model=Resposta, summary="Fazer uma pergunta ao chatbot")
def perguntar(pergunta: Pergunta):
    """
    Recebe uma pergunta do usuário e retorna a resposta gerada pelo sistema RAG.
    """
    if not pergunta.texto or not pergunta.texto.strip():
        raise HTTPException(status_code=400, detail="O texto da pergunta não pode estar vazio.")
    
    try:
        logger.info("Recebida pergunta: '%s'", pergunta.texto)
        result = qa_chain.invoke({"query": pergunta.texto})
        logger.info("Resposta gerada: '%s'", result["result"])
        return Resposta(resposta=result["result"])
    except Exception as e:
        logger.error("Erro ao processar a pergunta: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar sua pergunta.")

