"""LLM-as-judge evaluation: RAG+finetuned  vs.  finetuned-only  vs.  base Qwen (plain).

Loads your fine-tuned Qwen3-4B model ONCE, then for every question in
test_questions.json:
  1. Generates an answer WITH retrieval + fine-tune       (RAG)
  2. Generates an answer WITHOUT retrieval, adapter ON    (FT_ONLY)
  3. Generates an answer WITHOUT retrieval, adapter OFF   (BASE / plain Qwen)
  4. Sends each answer + the reference to a judge LLM for scoring

The judge can be:
  --judge gemini   (free, fast, needs GEMINI_API_KEY in .env)
  --judge qwen     (local base Qwen3-4B with the adapter temporarily disabled
                    -- 100% free, no key, no internet, but slower and more biased)

Outputs:
  - eval_results.csv     (per-question rows)
  - eval_summary.txt     (avg scores, win rates, last 5 Q&A)

Usage:
    python evaluate.py                       # qwen judge by default, all questions
    python evaluate.py --limit 10            # quick smoke test
    python evaluate.py --judge gemini        # use Gemini as judge (recommended)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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

GEMINI_API_KEY = "AIzaSyCTcA0_EiEMra2r1vFxUUhwVjjsPO68GqY"
GEMINI_MODEL = "gemini-2.0-flash"

ANTHROPIC_API_KEY="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"

from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "index"
TEST_FILE = str(_SCRIPT_DIR / "test_questions.json")
RESULTS_CSV = str(_SCRIPT_DIR / "eval_results.csv")
SUMMARY_TXT = str(_SCRIPT_DIR / "eval_summary.txt")
TOP_K = 4
MAX_NEW_TOKENS = 384


SYSTEM_PROMPT_RAG = (
    "You are a helpful assistant. Use the provided context to answer the user's "
    "question. If the context is not relevant, answer from your own knowledge."
)
SYSTEM_PROMPT_PLAIN = (
    "You are a helpful assistant. Answer the user's question concisely."
)

JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating answers to a question.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

CANDIDATE ANSWER:
{answer}

Score the candidate from 1 to 5 based on factual correctness and relevance to the question:
  5 = Fully correct and matches the reference's key facts
  4 = Mostly correct, minor omissions or wording differences
  3 = Partially correct, contains some right and some wrong information
  2 = Mostly incorrect, fails to address the question
  1 = Completely wrong, irrelevant, or refuses to answer

Respond with ONLY a single integer between 1 and 5. No explanation, no other text.
"""

class ClaudeJudge:
    name = "claude-haiku-4-5"

    def __init__(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Missing ANTHROPIC_API_KEY. Add it to your .env file.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def score(self, question, reference, answer):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, reference=reference, answer=answer
        )
        for attempt in range(3):
            try:
                message = self.client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=16,
                    messages=[{"role": "user", "content": prompt}]
                )
                return parse_score(message.content[0].text.strip())
            except Exception as e:
                print(f"  judge error (attempt {attempt+1}/3): {e}")
                time.sleep(2 ** attempt)
        return 0

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str):
    print(f"Loading base {BASE_MODEL} on {device}...")
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
    model = model.to(device).eval()
    print("Model ready.\n")
    return tokenizer, model


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": TOP_K})


def format_context(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


@torch.inference_mode()
def generate(tokenizer, model, messages, device, max_new_tokens=MAX_NEW_TOKENS) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_with_rag(question, tokenizer, model, retriever, device):
    """RAG + fine-tuned: retrieval ON, adapter ON."""
    docs = retriever.invoke(question)
    context = format_context(docs)
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return generate(
        tokenizer, model,
        [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": user},
        ],
        device,
    )


def answer_ft_only(question, tokenizer, model, device):
    """Fine-tuned only: retrieval OFF, adapter ON."""
    return generate(
        tokenizer, model,
        [
            {"role": "system", "content": SYSTEM_PROMPT_PLAIN},
            {"role": "user", "content": question},
        ],
        device,
    )


@torch.inference_mode()
def answer_base_qwen(question, tokenizer, model, device) -> str:
    """Plain base Qwen3-4B: retrieval OFF, adapter OFF."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PLAIN},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with model.disable_adapter():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_score(text: str) -> int:
    """Extract first 1-5 digit from text, return 0 if none found."""
    for ch in text:
        if ch in "12345":
            return int(ch)
    return 0


class GeminiJudge:
    name = "gemini-2.0-flash"

    def __init__(self):
        if not GEMINI_API_KEY:
            raise SystemExit(
                "Missing GEMINI_API_KEY. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env"
            )
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def score(self, question, reference, answer):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, reference=reference, answer=answer
        )
        for attempt in range(3):
            try:
                resp = self.model.generate_content(prompt)
                return parse_score((resp.text or "").strip())
            except Exception as e:
                print(f"  judge error (attempt {attempt+1}/3): {e}")
                time.sleep(2 ** attempt)
        return 0


class LocalQwenJudge:
    """Uses the same loaded Qwen3-4B but with the LoRA adapter DISABLED so the
    judge is the un-fine-tuned base model. Keeps everything local and free,
    but is biased toward its own (base) outputs — prefer Gemini if possible.
    """
    name = "qwen3-4b-base (no adapter)"

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def score(self, question, reference, answer):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, reference=reference, answer=answer
        )
        messages = [{"role": "user", "content": prompt}]
        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        with self.model.disable_adapter(), torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_score(text)


def format_last_n_qa(rows: list[dict], n: int = 5) -> str:
    tail = rows[-n:]
    lines = [f"\n{'='*60}", f"LAST {len(tail)} QUESTIONS & ANSWERS", f"{'='*60}"]
    for r in tail:
        lines.append(f"\nQ{r['idx']}: {r['question']}")
        lines.append(f"  Reference  : {r['reference']}")
        lines.append(f"  RAG+FT     (score {r['score_rag']}):    {r['answer_rag']}")
        lines.append(f"  FT only    (score {r['score_ft']}):    {r['answer_ft']}")
        lines.append(f"  Base Qwen  (score {r['score_base']}):    {r['answer_base']}")
        lines.append(f"  Winner: {r['winner']}")
        lines.append("-" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--tests", default=TEST_FILE, help="Path to test questions JSON")
    parser.add_argument(
        "--judge", choices=["gemini", "qwen", "claude"], default="claude",
        help="Which judge to use. 'gemini' = Gemini 2.0 Flash (free API). "
             "'qwen' = local base Qwen3-4B with adapter disabled.",
    )
    args = parser.parse_args()

    with open(args.tests, encoding="utf-8") as f:
        tests = json.load(f)
    if args.limit:
        tests = tests[: args.limit]
    print(f"Evaluating on {len(tests)} questions.\n")

    device = pick_device()
    tokenizer, model = load_model(device)
    retriever = load_retriever()

    if args.judge == "gemini":
        judge = GeminiJudge()
    elif args.judge == "claude":
        judge = ClaudeJudge()
    else:
        judge = LocalQwenJudge(tokenizer, model, device)
        print("NOTE: using local Qwen-base as judge. Scores are useful for "
              "comparing systems against each other, but absolute numbers are "
              "less reliable than a frontier judge.\n")
    print(f"Judge: {judge.name}\n")

    rows = []
    sum_rag = sum_ft = sum_base = 0
    rag_wins = ft_wins = base_wins = ties = 0

    for i, t in enumerate(tests, 1):
        q = t["question"]
        ref = t.get("reference", "")
        print(f"[{i}/{len(tests)}] {q[:80]}")

        # 1) RAG + fine-tuned
        t0 = time.time()
        ans_rag = answer_with_rag(q, tokenizer, model, retriever, device)
        rag_secs = time.time() - t0

        # 2) Fine-tuned only (no retrieval, adapter ON)
        t0 = time.time()
        ans_ft = answer_ft_only(q, tokenizer, model, device)
        ft_secs = time.time() - t0

        # 3) Plain base Qwen (no retrieval, adapter OFF)
        t0 = time.time()
        ans_base = answer_base_qwen(q, tokenizer, model, device)
        base_secs = time.time() - t0

        # Judge all three
        score_rag = judge.score(q, ref, ans_rag)
        score_ft = judge.score(q, ref, ans_ft)
        score_base = judge.score(q, ref, ans_base)

        sum_rag += score_rag
        sum_ft += score_ft
        sum_base += score_base

        # 3-way verdict: highest score wins; ties on highest -> TIE
        scores = {"RAG": score_rag, "FT_ONLY": score_ft, "BASE": score_base}
        top = max(scores.values())
        winners = [name for name, s in scores.items() if s == top]
        if len(winners) == 1:
            verdict = winners[0]
            if verdict == "RAG":
                rag_wins += 1
            elif verdict == "FT_ONLY":
                ft_wins += 1
            else:
                base_wins += 1
        else:
            verdict = "TIE(" + "/".join(winners) + ")"
            ties += 1

        print(
            f"   RAG ({rag_secs:5.1f}s)={score_rag}  "
            f"FT ({ft_secs:5.1f}s)={score_ft}  "
            f"BASE ({base_secs:5.1f}s)={score_base}  -> {verdict}"
        )

        rows.append({
            "idx": i,
            "question": q,
            "reference": ref,
            "answer_rag": ans_rag,
            "answer_ft": ans_ft,
            "answer_base": ans_base,
            "score_rag": score_rag,
            "score_ft": score_ft,
            "score_base": score_base,
            "winner": verdict,
            "secs_rag": round(rag_secs, 2),
            "secs_ft": round(ft_secs, 2),
            "secs_base": round(base_secs, 2),
        })

    if not rows:
        print("No rows to write -- exiting.")
        return

    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    avg_rag = sum_rag / n
    avg_ft = sum_ft / n
    avg_base = sum_base / n
    pct_rag = avg_rag / 5 * 100
    pct_ft = avg_ft / 5 * 100
    pct_base = avg_base / 5 * 100

    summary = (
        f"\n{'='*60}\n"
        f"EVALUATION SUMMARY ({n} questions)\n"
        f"{'='*60}\n"
        f"Judge:    {judge.name}\n"
        f"System A: RAG + fine-tuned   (retrieval ON,  adapter ON)\n"
        f"System B: fine-tuned only    (retrieval OFF, adapter ON)\n"
        f"System C: base Qwen (plain)  (retrieval OFF, adapter OFF)\n\n"
        f"Average score (out of 5):\n"
        f"  RAG     : {avg_rag:.2f}   ({pct_rag:.1f}% correctness)\n"
        f"  FT_ONLY : {avg_ft:.2f}   ({pct_ft:.1f}% correctness)\n"
        f"  BASE    : {avg_base:.2f}   ({pct_base:.1f}% correctness)\n\n"
        f"Deltas:\n"
        f"  RAG  - FT_ONLY : {avg_rag - avg_ft:+.2f}\n"
        f"  RAG  - BASE    : {avg_rag - avg_base:+.2f}\n"
        f"  FT_ONLY - BASE : {avg_ft  - avg_base:+.2f}\n\n"
        f"Head-to-head (3-way, highest score wins):\n"
        f"  RAG wins     : {rag_wins} ({rag_wins/n*100:.1f}%)\n"
        f"  FT_ONLY wins : {ft_wins} ({ft_wins/n*100:.1f}%)\n"
        f"  BASE wins    : {base_wins} ({base_wins/n*100:.1f}%)\n"
        f"  Ties         : {ties} ({ties/n*100:.1f}%)\n\n"
        f"Per-question rows: {RESULTS_CSV}\n"
        f"{'='*60}\n"
    )

    qa_tail = format_last_n_qa(rows, n=5)
    full_output = summary + qa_tail + "\n"

    print(full_output)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(full_output)


if __name__ == "__main__":
    main()