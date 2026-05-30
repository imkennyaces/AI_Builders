"""Ask questions against ./index/ using your fine-tuned LoRA adapter
on top of the Qwen3-4B base model, all running LOCALLY on Apple Silicon.
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import textwrap

import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

BASE_MODEL = os.environ.get("HF_BASE_MODEL", "Qwen/Qwen3-4B")
ADAPTER_ID = os.environ.get("HF_MODEL_ID", "KenKennyfr/qwen-finetuned")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "index"
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant. You will be given some context from the user's "
    "documents along with their question.\n\n"
    "- If the context contains information relevant to the question, use it to "
    "answer and cite the source filenames in square brackets at the end (e.g. [bible.txt]).\n"
    "- If the context is not relevant, or the question is general, conversational, "
    "or about something else entirely, just answer normally from your own knowledge. "
    "Do NOT say 'the context does not mention this' — instead, answer the question "
    "directly.\n"
    "- Match the user's language. If they write in Thai, respond in Thai."
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def load_model(device: str):
    print(f"Loading base model {BASE_MODEL} on {device}...")
    print("  (first run downloads ~8 GB; subsequent runs use the local cache)")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, token=HF_TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print(f"Applying LoRA adapter {ADAPTER_ID}...")
    model = PeftModel.from_pretrained(base, ADAPTER_ID, token=HF_TOKEN)
    model = model.to(device)
    model.eval()
    print("Model ready.\n")
    return tokenizer, model


def format_context(docs) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        tag = f"{src}#p{page}" if page is not None else src
        blocks.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, context: str):
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@torch.inference_mode()
def generate(tokenizer, model, messages, device, max_new_tokens=512) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer(question, tokenizer, model, retriever, device) -> str:
    docs = retriever.invoke(question)
    context = format_context(docs)
    messages = build_messages(question, context)
    return generate(tokenizer, model, messages, device)


def interactive(tokenizer, model, retriever, device) -> None:
    print("RAG ready. Ctrl-C to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        print("\nmodel >")
        print(textwrap.fill(answer(q, tokenizer, model, retriever, device), width=100))
        print()


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")
    retriever = load_retriever()
    tokenizer, model = load_model(device)
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(answer(question, tokenizer, model, retriever, device))
    else:
        interactive(tokenizer, model, retriever, device)


if __name__ == "__main__":
    main()