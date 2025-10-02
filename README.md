# 🤖 Chatbot (Zephyr) — Gradio

Un petit chatbot utilisant **Zephyr-7B-Beta** (Hugging Face) avec une interface **Gradio**.  
Il répond en **une seule phrase claire** et propose un bouton **✈️ Envoyer** et un bouton **🧹 Réinitialiser la conversation**.

---

## ✨ Fonctionnalités

- Réponses **concises** en français (1 phrase maximum)
- Historique de conversation (`type="messages"`, format Gradio moderne)
- Bouton **Envoyer** + envoi via **Entrée**
- Nettoyage automatique des réponses (suppression des “Note : …”)
- Compatible **CPU/GPU** grâce à `device_map="auto"` et `torch_dtype="auto"`
- CSS externe (`style.css`) pour personnaliser l’interface

---

## 🧠 Technologie

- **LLM (Large Language Model)** : [Zephyr-7B-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)  
- **Librairie Transformers** : chargement du tokenizer, génération de texte avec `AutoModelForCausalLM`  
- **Gradio** : création de l’interface utilisateur (chat en mode `messages`)  
- **PyTorch** : backend pour l’exécution du modèle (CPU ou GPU)

---

## 🧰 Prérequis

- Python **3.10+**
- Internet au premier lancement (téléchargement du modèle Hugging Face)

---

## ⚙️ Installation

Cloner le dépôt et installer les dépendances :

```bash
pip install -U pip
pip install gradio transformers torch
```

---

## ▶️ Lancement

```bash
python chatbot.py
Gradio démarre et affiche une URL locale du type:
http://127.0.0.1:7860
```

---

## 🖱️ Utilisation

1. Tape ta question dans la barre « Écris ton message… »
2. Appuie sur Entrée ou clique sur ✈️ Envoyer
3. Le chatbot répond en une phrase
4. Clique sur 🧹 Réinitialiser la conversation pour repartir de zéro

---

## 🗂️ Structure du projet

.

├── chatbot.py             # Script principal

├── style.css              # CSS pour personnaliser l’UI

├── README.md 

---

## 🐢 Performances & conseils

- Sur CPU, Zephyr-7B peut être lent (jusqu’à 30–60s).
- Pour améliorer :
    - réduire max_new_tokens (20 au lieu de 60)
    - utiliser un modèle plus petit (Zephyr 3B)

---

## 📝 Exemple
 
<img width="1751" height="895" alt="image" src="https://github.com/user-attachments/assets/a22d20ce-cd13-4095-b3f0-db8c2308ed1a" />
