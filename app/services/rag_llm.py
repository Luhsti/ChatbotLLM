# Imports necessários
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def get_llm():
    """
    Carrega e retorna a instância do LLM (flan-t5-base).
    """

    # Carrega o tokenizer e o modelo
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # Cria o pipeline de geração de texto
    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        repetition_penalty=1.3,
        do_sample=True
    )

    # Retorna o pipeline encapsulado pelo LangChain
    return HuggingFacePipeline(pipeline=pipe)
