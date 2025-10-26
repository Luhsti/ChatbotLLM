from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.services import rag_llm
from app.core.config import settings
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Logging para terminal e arquivo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler("app_logs.txt", mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(console_handler)

SIMILARITY_THRESHOLD = 0.7  # Limite mínimo de similaridade real

# Prompt em português para respostas claras, completas e naturais
PROMPT = PromptTemplate(
    template=(
        "Responda à pergunta usando apenas o contexto abaixo.\n"
        "Se não houver informação suficiente no contexto, diga:\n"
        "\"Desculpe, não encontrei a informação na minha base.\"\n\n"
        "Reescreva a resposta em **frases completas e claras**, "
        "usando exatamente as palavras do contexto e mantendo o significado original.\n\n"
        "Contexto: {context}\n\nPergunta: {question}\n\nResposta:"
    ),
    input_variables=["context", "question"]
)

def build_chain():
    """
    Constrói a cadeia RAG e retorna função wrapper para consultas.
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

    logger.info("Preparando LLM...")
    llm = rag_llm.get_llm()

    logger.info("Montando a cadeia RetrievalQA...")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    def perguntar_ao_rag(pergunta: str):
        """
        Retorna resposta natural em português usando filtro de similaridade.
        """
        logger.info("Pergunta recebida: %s", pergunta)

        # Busca top-k documentos
        docs = db.similarity_search(pergunta, k=3)
        logger.info("Documentos retornados pelo FAISS: %d", len(docs))

        if not docs:
            logger.info("Nenhum documento encontrado.")
            return "Desculpe, não encontrei a informação na minha base."

        # Calcula embedding da pergunta
        pergunta_emb = embeddings.embed_query(pergunta)
        doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])

        # Calcula similaridade coseno
        sims = cosine_similarity([pergunta_emb], doc_embeddings)[0]
        logger.info("Similaridades calculadas: %s", sims)

        # Seleciona documentos acima do limiar
        relevant_docs = [doc for doc, sim in zip(docs, sims) if sim >= SIMILARITY_THRESHOLD]
        logger.info("Documentos relevantes após filtro: %d", len(relevant_docs))

        if not relevant_docs:
            logger.info("Nenhum documento passou no limiar de similaridade.")
            return "Desculpe, não encontrei a informação na minha base."

        # Monta contexto concatenado
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        logger.info("Contexto final usado para LLM: %s", context)

        # Gera prompt final
        final_prompt = PROMPT.format(context=context, question=pergunta)
        logger.info("Prompt enviado ao LLM: %s", final_prompt)

        # Gera resposta
        resposta = llm.invoke(final_prompt)
        logger.info("Resposta final: %s", resposta)

        return resposta

    return chain, perguntar_ao_rag
