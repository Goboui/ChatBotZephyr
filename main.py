import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

SYSTEM_MSG = (
    "Tu es un assistant francophone concis. Réponds uniquement à la question de l'utilisateur "
    "en une seule phrase claire. N'ajoute pas d'exemples, d'explications ou de questions "
    "supplémentaires. Ne répète pas la question."
)

def sanitize_user_text(text: str) -> str:
    text = re.sub(r"<\|(system|user|assistant)\|>", "", text, flags=re.IGNORECASE)
    return text.strip()

def postprocess_answer(text: str) -> str:
    text = re.sub(r"^Note\s*:.*", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    text = re.sub(r"<\|/?(system|user|assistant)\|>", "", text, flags=re.IGNORECASE).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        text = sentences[0].strip()
    return text

def user_message(user_input, chat_history):
    user_input = sanitize_user_text(user_input or "")
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})
    return "", chat_history

def bot_message(chat_history):
    chat_history = chat_history or []
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for m in chat_history:
        if isinstance(m, dict) and m.get("role") in {"user", "assistant"} and "content" in m:
            messages.append({"role": m["role"], "content": m["content"]})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = postprocess_answer(raw_text)

    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

def clear_chat():
    return "", []

with gr.Blocks(css="style.css") as demo:
    gr.Markdown("## 🤖 Chatbot (Zephyr)")
    chatbot = gr.Chatbot(type="messages", height=600)

    msg = gr.Textbox(
        label="",
        placeholder="Écris ton message…",
        lines=1,
        max_lines=4
    )

    send_btn = gr.Button("✈️ Envoyer")
    clear = gr.Button("🧹 Réinitialiser la conversation")

    msg.submit(user_message, [msg, chatbot], [msg, chatbot]) \
       .then(bot_message, chatbot, chatbot)

    send_btn.click(user_message, [msg, chatbot], [msg, chatbot]) \
            .then(bot_message, chatbot, chatbot)

    clear.click(clear_chat, None, [msg, chatbot])

demo.launch()
