import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Choix du modèle
model_id = "HuggingFaceH4/zephyr-7b-beta"

# Chargement du tokenizer et du modèle
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Historique de la conversation avec une instruction stricte
history = (
    "<|system|>\n"
    "Tu es un assistant francophone concis. Réponds uniquement à la question de l'utilisateur, avec une seule phrase claire."
    "N'ajoute pas de suggestions, d'exemples, d'explications ou de questions supplémentaires. "
    "Ne répète pas la question, ne propose pas d'autres informations.\n"
)

# Fonction de génération de réponse
def chat(user_input, chat_history):
    global history

    # Ajout de l'entrée utilisateur dans le prompt
    prompt = history + f"<|user|>\n{user_input}\n<|assistant|>\n"

    # Tokenisation et génération
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7
    )

    # Décodage de la réponse
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response[len(prompt):].strip()

    # On garde que les 2 premières phrases
    sentences = re.split(r'(?<=[.!?]) +', response_text)
    response_text = sentences[0].strip()

    # Mise à jour de l'historique
    history += f"<|user|>\n{user_input}\n<|assistant|>\n{response_text}\n"
    chat_history.append((user_input, response_text))

    return "", chat_history

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Votre message")
    clear = gr.Button("🧹 Réinitialiser la conversation")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ("", []), None, [msg, chatbot])

demo.launch()
