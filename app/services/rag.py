from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.services import rag_llm  # Importa o módulo que fornece o LLM
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Define o prompt que guiará o LLM, instruindo-o a usar apenas o contexto
PROMPT = PromptTemplate(
    template=(
        "Use somente o contexto abaixo para responder à pergunta.\n"
        "Se você não sabe a resposta com base no contexto, apenas diga "
        "\"Desculpe, não encontrei a informação necessária nos meus documentos.\", "
        "não tente inventar uma resposta.\n\n"
        "Contexto: {context}\n\nPergunta: {question}\n\nResposta:"
    ),
    input_variables=["context", "question"]
)

def build_chain():
    """
    Constrói a cadeia RAG completa.
    """
    logger.info("Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    logger.info("Carregando banco de dados vetorial FAISS de %s", settings.vector_db_path)
    db = FAISS.load_local(
        settings.vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logger.info(
        "Preparando LLM (%s)...",
        "local" if settings.use_local_llm else "endpoint"
    )
    llm = rag_llm.get_llm()

    logger.info("Montando a cadeia RetrievalQA...")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),  # Busca os 2 chunks mais relevantes
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False,
    )

    return chain
