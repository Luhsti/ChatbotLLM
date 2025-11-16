# rag_llm.py
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s — %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DISPOSITIVO (CPU/GPU)
# ---------------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
_device_name = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Dispositivo detectado: {_device_name}")

# ---------------------------------------------------------------------------
# MODELO PRINCIPAL — FLAN-T5 (RAG)
# ---------------------------------------------------------------------------
rag_model_name = "google/flan-t5-base"

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

# ---------------------------------------------------------------------------
# MODELO DE REFINAMENTO — PTT5 (UNICAMP)
# ---------------------------------------------------------------------------
refine_model_name = "unicamp-dl/ptt5-large-portuguese-vocab"

logger.info("Carregando modelo de refinamento (PTT5 — Unicamp)...")
tokenizer_refino = AutoTokenizer.from_pretrained(refine_model_name)
modelo_refino = AutoModelForSeq2SeqLM.from_pretrained(refine_model_name)

# Move para GPU se disponível
if torch.cuda.is_available():
    try:
        modelo_refino = modelo_refino.to("cuda")
    except:
        logger.warning("Falha ao mover modelo para CUDA — usando CPU.")
        modelo_refino = modelo_refino.to("cpu")
else:
    modelo_refino = modelo_refino.to("cpu")

logger.info("Modelo PTT5 carregado com sucesso.")

# ---------------------------------------------------------------------------
# FUNÇÃO — Refinamento via PTT5
# ---------------------------------------------------------------------------
def refinar_texto(texto_bruto: str, max_length: int = 180) -> str:
    """
    Reescreve/parafraseia o texto em português usando o PTT5.
    Modelo treinado especificamente para a língua portuguesa.
    """

    if not texto_bruto or not texto_bruto.strip():
        return texto_bruto

    prompt = f"parafrasear: {texto_bruto.strip()}"

    try:
        inputs = tokenizer_refino(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(modelo_refino.device) for k, v in inputs.items()}

        outputs = modelo_refino.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

        texto_refinado = tokenizer_refino.decode(outputs[0], skip_special_tokens=True).strip()

        # Remover echos do comando
        for prefix in ["parafrasear:", "parafrasear", "texto:", "retranscrição:"]:
            if texto_refinado.lower().startswith(prefix):
                texto_refinado = texto_refinado[len(prefix):].strip()

        # Garante finalização limpa
        if texto_refinado and not texto_refinado.endswith("."):
            texto_refinado += "."

        return texto_refinado

    except Exception as e:
        logger.error(f"Erro no refinamento: {e}", exc_info=True)
        return texto_bruto

# ---------------------------------------------------------------------------
# FUNÇÃO — Gerar resposta via FLAN-T5 usando contexto
# ---------------------------------------------------------------------------
def gerar_resposta_rag(pergunta: str, contexto: str) -> str:
    prompt = (
        "Use somente o contexto abaixo para responder à pergunta.\n"
        "Responda em português simples e objetivo.\n"
        "Se a informação não estiver no contexto, diga exatamente:\n"
        "\"Desculpe, não encontrei a informação necessária nos meus documentos.\"\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pergunta:\n{pergunta}\n\n"
        "Resposta:"
    )

    logger.info("🔎 Gerando resposta bruta com FLAN-T5...")

    try:
        generated = pipe_rag(prompt)[0]["generated_text"].strip()

        cleaned = generated.replace(prompt, "").strip()

        if "Resposta:" in cleaned:
            resposta = cleaned.split("Resposta:", 1)[1].strip()
        else:
            resposta = cleaned

        if not resposta or len(resposta) < 3:
            logger.warning("Resposta muito curta — aplicando fallback.")
            return "Desculpe, não encontrei a informação necessária nos meus documentos."

        return resposta

    except Exception as e:
        logger.error(f"Erro FLAN-T5: {e}", exc_info=True)
        return "Não foi possível gerar a resposta no momento."

# ---------------------------------------------------------------------------
# FUNÇÃO — Formatador final (usa PTT5)
# ---------------------------------------------------------------------------
def formatar_resposta(texto: str) -> str:
    if texto.lower().startswith("desculpe, não encontrei"):
        return texto

    logger.info("✨ Refinando texto com PTT5...")
    refinado = refinar_texto(texto)

    if not refinado or len(refinado) < 3:
        logger.warning("Refino retornou texto inválido — mantendo original.")
        return texto

    return refinado

# ---------------------------------------------------------------------------
# FUNÇÃO — Pipeline completo
# ---------------------------------------------------------------------------
def responder_pergunta(pergunta: str, contexto: str) -> str:
    resposta_bruta = gerar_resposta_rag(pergunta, contexto)
    resposta_final = formatar_resposta(resposta_bruta)
    return resposta_final
