"""Gradio web UI for the RAG + fine-tuned Qwen3-4B app.
Entry point for Hugging Face Spaces.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from rag import answer, load_model, load_retriever, pick_device
print("Initialising RAG pipeline...")
device = pick_device()
print(f"Device: {device}")
retriever = load_retriever()
tokenizer, model = load_model(device)
print("Ready.\n")

def respond(message: str, history: list[list[str]]) -> str:
    if not message.strip():
        return ""
    return answer(message, tokenizer, model, retriever, device)

with gr.Blocks(title="RAG QA — Qwen3-4B") as demo:
    gr.Markdown(
        """
        # 📚 RAG QA — Qwen3-4B + LoRA
        Ask questions about your documents. The model retrieves relevant context
        from the FAISS index and answers using your fine-tuned adapter.
        """
    )

    chatbot = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=480, show_label=False),
        textbox=gr.Textbox(
            placeholder="Ask a question about your documents...",
            container=False,
            scale=7,
        ),
        examples=[
            "What is this document about?",
            "Summarise the key points.",
            "What does the document say about [topic]?",
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="↩ Undo",
        clear_btn="🗑 Clear",
    )

if __name__ == "__main__":
    demo.launch()