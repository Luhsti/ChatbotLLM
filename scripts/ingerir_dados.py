from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Configuração básica de logging para o script
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Define os caminhos base
BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_PATH = BASE_DIR / "data" / "documentos"
VECTOR_DB_PATH = BASE_DIR / "vectorstore" / "db_faiss"

def main():
    """
    Função principal para carregar documentos, gerar embeddings e criar o índice FAISS.
    """
    if not DOCS_PATH.exists() or not any(DOCS_PATH.iterdir()):
        raise SystemExit(f"ERRO: A pasta de documentos '{DOCS_PATH}' não existe ou está vazia.")

    # Carrega todos os arquivos .txt da pasta
    logging.info("Carregando documentos de %s...", DOCS_PATH)
    loader = DirectoryLoader(
        str(DOCS_PATH),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    if not documents:
        raise SystemExit(f"Nenhum documento .txt encontrado em {DOCS_PATH}")

    logging.info("Total de %d documentos carregados.", len(documents))

    # Carrega o modelo de embeddings
    logging.info("Carregando modelo de embeddings (intfloat/e5-base-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"}
    )

    # Cria o banco de dados vetorial e o salva localmente
    logging.info("Criando e salvando o índice FAISS em %s...", VECTOR_DB_PATH)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)  # Garante que a pasta de destino exista
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(str(VECTOR_DB_PATH))

    logging.info("OK: Índice FAISS criado e salvo com sucesso!")

# 🔹 Chamada principal do script
if __name__ == "__main__":
    main()
