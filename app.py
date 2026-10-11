import os, time, secrets, threading, gc
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# ---- КОНФИГ ----
BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
LORA_MODEL_ID = "kosoftiks/kostik"          # Твой репозиторий на HF
HF_TOKEN      = os.environ.get("HF_TOKEN")
PORT          = int(os.environ.get("PORT", 10000))

SYSTEM_PROMPT = """Ты Костик — дружелюбный ИИ-ассистент, созданный компанией Kosoftiks Corp.
Когда тебя спрашивают, кто ты — отвечай: «Я Костик, ИИ от Kosoftiks Corp».
Не упоминай Qwen, Alibaba, OpenAI или других разработчиков.
Отвечай на русском, кратко и по делу."""

# ---- ГЛОБАЛЬНЫЕ ----
model = None
tokenizer = None
api_keys = {}          # {key: {"name": str, "created": str}}
master_key = None

app = FastAPI(title="Kostik API", version="1.0")

# ---- ЗАГРУЗКА МОДЕЛИ ----
def load_model():
    global model, tokenizer, master_key
    print("Загрузка модели...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL_ID, token=HF_TOKEN)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    master_key = "kostik-master-" + secrets.token_urlsafe(24)
    api_keys[master_key] = {"name": "master", "created": time.strftime("%Y-%m-%d %H:%M")}

    print("=" * 70)
    print("МОДЕЛЬ ЗАГРУЖЕНА")
    print("=" * 70)
    print(f"МАСТЕР-КЛЮЧ: {master_key}")
    print("=" * 70)

# ---- МОДЕЛИ ДАННЫХ ----
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

# ---- АВТОРИЗАЦИЯ ----
def verify_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    key = authorization.replace("Bearer ", "").strip()
    if key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key

# ---- ЭНДПОИНТЫ ----
@app.on_event("startup")
def startup():
    threading.Thread(target=load_model, daemon=True).start()

@app.get("/health")
def health():
    return {"status": "ok", "model": "Kostik", "loaded": model is not None}

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "kostik", "object": "model", "created": int(time.time())}]}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    k = auth.replace("Bearer ", "").strip()
    if k not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again in a minute")

    data = await request.json()
    messages = data.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    temperature = float(data.get("temperature", 0.6))
    max_tokens = int(data.get("max_tokens", 512))
    top_p = float(data.get("top_p", 0.9))

    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1].strip()

        return {
            "id": "chatcmpl-kostik",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "kostik",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "completion_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
                "total_tokens": int(out.shape[1]),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/command")
async def command(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    k = auth.replace("Bearer ", "").strip()
    if k not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    data = await request.json()
    cmd = (data.get("command") or "").strip()

    if cmd.startswith("new_key "):
        name = cmd[8:].strip() or "api"
        new_key = f"kostik-{name}-" + secrets.token_urlsafe(20)
        api_keys[new_key] = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M")}
        return {"key": new_key, "name": name}

    if cmd == "list_keys":
        return {"keys": [{"name": v["name"], "created": v["created"], "key_prefix": kk[:20] + "..."}
                         for kk, v in api_keys.items()]}

    if cmd.startswith("revoke "):
        prefix = cmd[7:].strip()
        to_remove = [kk for kk in api_keys if kk.startswith(prefix)]
        for kk in to_remove:
            del api_keys[kk]
        return {"revoked": len(to_remove)}

    raise HTTPException(status_code=400, detail="Available: new_key <name>, list_keys, revoke <prefix>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
