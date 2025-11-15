# rag_llm.py
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# CONFIG DO LOG
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# DISPOSITIVO (CPU / GPU)
# ============================================================
device = 0 if torch.cuda.is_available() else -1
_device_name = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Dispositivo detectado: {_device_name}")

# ============================================================
# MODELO PRINCIPAL (RAG) - FLAN-T5 (mantido)
# ============================================================
rag_model_name = "google/flan-t5-base"

# pipeline para RAG (determinístico, gera respostas com beam search)
pipe_rag = pipeline(
    task="text2text-generation",
    model=rag_model_name,
    tokenizer=rag_model_name,
    device=device,
    max_new_tokens=200,
    do_sample=False,
    num_beams=4,
    early_stopping=True
)

# ============================================================
# MODELO DE REFINAMENTO — PTT5 (UNICAMP)
# ============================================================
refine_model_name = "unicamp-dl/ptt5-base-portuguese-vocab"

logger.info("Carregando modelo de refinamento (Unicamp PTT5)...")
tokenizer_refino = AutoTokenizer.from_pretrained(refine_model_name)
modelo_refino = AutoModelForSeq2SeqLM.from_pretrained(refine_model_name)

# move modelo para cuda se disponível
if torch.cuda.is_available():
    try:
        modelo_refino = modelo_refino.to("cuda")
    except Exception:
        # em alguns ambientes device_map / accelerate é necessário; fallback para cpu
        logger.warning("Não foi possível mover o modelo para CUDA — usando CPU.")
        modelo_refino = modelo_refino.to("cpu")
else:
    modelo_refino = modelo_refino.to("cpu")

logger.info("Modelo de refinamento pronto.")


# ============================================================
# FUNÇÃO AUX — refinar_texto usando PTT5 (Unicamp)
# ============================================================
def refinar_texto(texto_bruto: str, max_length: int = 180) -> str:
    """
    Parafraseia / reescreve o texto em português usando o PTT5 da Unicamp.
    Retorna o texto refinado (sem incluir o prompt).
    """

    if not texto_bruto or not texto_bruto.strip():
        return texto_bruto

    # usar comando compacto que T5 entende bem
    prompt = f"parafrasear: {texto_bruto.strip()}"

    try:
        inputs = tokenizer_refino(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # move tensores pro mesmo device do modelo
        input_ids = inputs["input_ids"].to(modelo_refino.device)
        attention_mask = inputs["attention_mask"].to(modelo_refino.device)

        outputs = modelo_refino.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )

        resposta = tokenizer_refino.decode(outputs[0], skip_special_tokens=True).strip()

        # limpeza simples: remover possíveis ecoamentos do comando
        for prefix in ["parafrasear:", "parafrasear", "texto:", "retranscrição:"]:
            if resposta.lower().startswith(prefix):
                resposta = resposta[len(prefix):].strip()

        # garantir pontuação final
        if resposta and not resposta.endswith("."):
            resposta += "."

        return resposta

    except Exception as e:
        logger.error(f"Erro no refinamento (PTT5): {e}", exc_info=True)
        return texto_bruto


# ============================================================
# FUNÇÃO 1 — Gerar resposta bruta via RAG (FLAN-T5)
# ============================================================
def gerar_resposta_rag(pergunta: str, contexto: str) -> str:
    """
    Gera a resposta baseada no contexto usando FLAN-T5.
    Extrai e limpa a resposta; aplica fallback seguro quando necessário.
    """
    prompt = (
        "Use somente o contexto abaixo para responder à pergunta.\n"
        "Retorne apenas a resposta em português simples, sem repetir instruções.\n"
        "Se a informação não estiver no contexto, responda exatamente:\n"
        "\"Desculpe, não encontrei a informação necessária nos meus documentos.\"\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pergunta:\n{pergunta}\n\n"
        "Resposta:"
    )

    logger.info("🔎 Gerando resposta bruta via RAG (FLAN-T5)...")

    try:
        gen = pipe_rag(prompt)[0]["generated_text"].strip()
        logger.debug(f"[RAW RAG] {gen}")

        # tenta remover eco do prompt
        cleaned = gen.replace(prompt, "").strip()

        # se existir marcador "Resposta:" pega o que vem depois
        if "Resposta:" in cleaned:
            resposta = cleaned.split("Resposta:", 1)[1].strip()
        elif "resposta:" in cleaned:
            resposta = cleaned.split("resposta:", 1)[1].strip()
        else:
            resposta = cleaned

        # validações simples
        lower = (resposta or "").lower()
        invalid_starts = ["use somente", "você é um", "contexto:", "pergunta:", "responda"]

        if not resposta or len(resposta) < 3 or any(lower.startswith(s) for s in invalid_starts):
            logger.warning("Resposta RAG inválida ou repetição do prompt detectada. Aplicando fallback.")
            # fallback: extrai primeira frase útil do contexto, se houver
            if contexto and "." in contexto:
                first = contexto.split(".")[0].strip()
                if len(first) > 8:
                    return first + "."
            return "Desculpe, não encontrei a informação necessária nos meus documentos."

        logger.info(f"🧠 Resposta bruta extraída: {resposta}")
        return resposta

    except Exception as e:
        logger.error(f"Erro ao gerar resposta RAG: {e}", exc_info=True)
        return "Não foi possível gerar a resposta no momento."


# ============================================================
# FUNÇÃO 2 — Formatador (usa refinar_texto)
# ============================================================
def formatar_resposta(texto: str) -> str:
    """
    Se o texto já for o fallback, retorna tal; senão, aplica refinar_texto().
    """
    if not texto or texto.lower().startswith("desculpe, não encontrei"):
        return texto

    logger.info("✨ Refinando resposta com PTT5 (Unicamp)...")

    refinado = refinar_texto(texto)

    # validação pós-refinamento
    if not refinado or len(refinado) < 3:
        logger.warning("Refinamento retornou resultado curto — usando versão bruta.")
        return texto

    logger.info(f"🔧 Texto refinado final: {refinado}")
    return refinado


# ============================================================
# FUNÇÃO 3 — Pipeline final
# ============================================================
def responder_pergunta(pergunta: str, contexto: str) -> str:
    resposta_bruta = gerar_resposta_rag(pergunta, contexto)
    resposta_final = formatar_resposta(resposta_bruta)
    return resposta_final
