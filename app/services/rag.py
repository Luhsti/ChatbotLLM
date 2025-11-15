from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.services import rag_llm
from app.core.config import settings
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler("app_logs.txt", mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.7  # Limiar mínimo de similaridade

# ---------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE CONSTRUÇÃO DO RAG
# ---------------------------------------------------------------------------
def build_chain():
    """
    Constrói o pipeline RAG completo:
    - Recupera contexto do FAISS
    - Gera resposta via LLM (rag_llm)
    - Refina o texto final
    """
    logger.info("Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"}
    )

    logger.info("Carregando base vetorial FAISS: %s", settings.vector_db_path)
    db = FAISS.load_local(
        settings.vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logger.info("✅ Pipeline de embeddings e FAISS inicializados com sucesso.")

    # -----------------------------------------------------------------------
    # FUNÇÃO INTERNA: PERGUNTAR AO RAG
    # -----------------------------------------------------------------------
    def perguntar_ao_rag(pergunta: str):
        logger.info("📥 Pergunta recebida: %s", pergunta)

        # Recupera documentos mais similares
        docs = db.similarity_search(pergunta, k=3)
        if not docs:
            logger.warning("Nenhum documento encontrado.")
            return "Desculpe, não encontrei informações relevantes na base."

        # Calcula similaridade coseno
        pergunta_emb = embeddings.embed_query(pergunta)
        doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])
        sims = cosine_similarity([pergunta_emb], doc_embeddings)[0]
        relevant_docs = [doc for doc, sim in zip(docs, sims) if sim >= SIMILARITY_THRESHOLD]

        if not relevant_docs:
            logger.warning("Nenhum documento passou no limiar de similaridade (%.2f).", SIMILARITY_THRESHOLD)
            return "Desculpe, não encontrei informações relevantes na base."

        # Monta o contexto final
        contexto = "\n\n".join([doc.page_content.strip() for doc in relevant_docs])
        logger.info("📚 Contexto final usado para LLM: %s", contexto)

        try:
            # Marca tempo de execução
            start_time = time.time()

            # Gera e formata resposta automaticamente
            resposta_final = rag_llm.responder_pergunta(pergunta, contexto)

            elapsed = time.time() - start_time
            logger.info(f"⏱️ Tempo total de resposta: {elapsed:.2f}s")
            logger.info(f"💬 Resposta final do RAG: {resposta_final}")

            return resposta_final

        except Exception as e:
            logger.error(f"❌ Erro ao gerar resposta: {e}")
            return "Ocorreu um erro interno ao gerar a resposta."

    return perguntar_ao_rag
