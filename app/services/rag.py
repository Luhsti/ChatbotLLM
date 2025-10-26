from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.services import rag_llm
from app.core.config import settings
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

# Limiar de similaridade cosine real para considerar documento relevante
SIMILARITY_THRESHOLD = 0.7

# Prompt que guiará o LLM
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
    Constrói a cadeia RAG e retorna função wrapper.
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
        retriever=db.as_retriever(search_kwargs={'k': 5}),  # pega mais docs para comparação
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    def perguntar_ao_rag(pergunta: str):
        """
        Retorna a resposta do RAG usando filtro de similaridade real.
        """
        # Busca top-k documentos
        docs = db.similarity_search(pergunta, k=5)

        if not docs:
            return "Desculpe, não possuo a informação na minha base."

        # Calcula embedding da pergunta
        pergunta_embedding = embeddings.embed_query(pergunta)

        # Calcula embeddings dos documentos
        doc_embeddings = np.array([embeddings.embed_query(doc.page_content) for doc in docs])

        # Calcula cosine similarity
        sims = cosine_similarity([pergunta_embedding], doc_embeddings)[0]

        # Seleciona documentos acima do limiar
        relevant_docs = [doc for doc, sim in zip(docs, sims) if sim >= SIMILARITY_THRESHOLD]

        if not relevant_docs:
            return "Desculpe, não possuo a informação na minha base."

        # Monta o contexto concatenando os docs relevantes
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Gera prompt final
        final_prompt = PROMPT.format(context=context, question=pergunta)

        # Chama o LLM
        return llm.invoke(final_prompt)

    return chain, perguntar_ao_rag
