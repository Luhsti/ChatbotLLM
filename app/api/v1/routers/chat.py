from fastapi import APIRouter, HTTPException
from app.models.schemas import Pergunta, Resposta
from app.services.rag import build_chain
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Constrói a cadeia e a função wrapper

qa_chain, perguntar_ao_rag = build_chain()

@router.post("/perguntar", response_model=Resposta)
def perguntar(pergunta: Pergunta):
    if not pergunta.texto.strip():
        raise HTTPException(status_code=400, detail="O texto da pergunta não pode estar vazio.")

    try:
        result = perguntar_ao_rag(pergunta.texto)
        return Resposta(resposta=result)
    except Exception as e:
        logger.error("Erro ao processar a pergunta: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar sua pergunta.")








