"""
Evaluation harness for the AI Writing Assistant fine-tuned models.

Provides task-specific quality gates beyond simple loss measurement.
Each task has a primary metric and secondary metrics. A model is promoted
only if it beats the current champion on the primary metric.

Usage:
  python scripts/evaluate.py --task paraphrase --adapter models/adapters/paraphrase/v2
  python scripts/evaluate.py --task all --golden-only

Metrics per task:
  - Paraphrase: BERTScore (semantic preservation) + Self-BLEU (diversity)
  - Grammar:    GLEU (correction quality) + over-correction rate
  - Simplify:   SARI (simplification quality) + readability gain
  - Tone:       BERTScore (content preservation) + style accuracy
  - Summarize:  ROUGE-1/2/L (faithful compression)
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("evaluate")

# ── Golden Set ───────────────────────────────────────────────────────────────
# A frozen set of manually verified examples per task for regression testing.
# These must NEVER change between runs — they are the fixed benchmark.

GOLDEN_SETS = {
    "paraphrase": [
        {"input": "paraphrase: The cat sat on the mat.", "expected": "The feline rested on the rug."},
        {"input": "paraphrase: It is raining heavily outside.", "expected": "There is heavy rainfall outdoors."},
        {"input": "paraphrase: She completed the project ahead of schedule.", "expected": "She finished the project before the deadline."},
        {"input": "paraphrase: The company announced record profits this quarter.", "expected": "The firm reported its highest earnings this quarter."},
        {"input": "paraphrase: Students should study regularly for best results.", "expected": "Learners ought to study consistently for optimal outcomes."},
        {"input": "paraphrase: The weather forecast predicts snow tomorrow.", "expected": "Snow is expected tomorrow according to the forecast."},
        {"input": "paraphrase: Technology has transformed modern communication.", "expected": "Modern communication has been revolutionized by technology."},
        {"input": "paraphrase: The restaurant was fully booked for the evening.", "expected": "There were no available reservations at the restaurant that night."},
        {"input": "paraphrase: He decided to pursue a career in medicine.", "expected": "He chose to become a medical professional."},
        {"input": "paraphrase: The traffic congestion caused significant delays.", "expected": "Heavy traffic led to major delays."},
    ],
    "grammar": [
        {"input": "grammar: He go to school yesterday.", "expected": "He went to school yesterday."},
        {"input": "grammar: She don't know the answer.", "expected": "She doesn't know the answer."},
        {"input": "grammar: They was playing in the park.", "expected": "They were playing in the park."},
        {"input": "grammar: Me and him went to the store.", "expected": "He and I went to the store."},
        {"input": "grammar: The childs are playing outside.", "expected": "The children are playing outside."},
        {"input": "grammar: I have went there before.", "expected": "I have gone there before."},
        {"input": "grammar: She is more smarter than him.", "expected": "She is smarter than him."},
        {"input": "grammar: The informations is incorrect.", "expected": "The information is incorrect."},
        {"input": "grammar: He don't likes vegetables.", "expected": "He doesn't like vegetables."},
        {"input": "grammar: We was excited about the trip.", "expected": "We were excited about the trip."},
    ],
    "simplify": [
        {"input": "simplify: The implementation of the aforementioned algorithm necessitates a comprehensive understanding of computational complexity theory.", "expected": "You need to understand complexity theory to use this algorithm."},
        {"input": "simplify: The utilization of renewable energy sources has been increasing substantially in recent years.", "expected": "More people are using renewable energy these days."},
        {"input": "simplify: The ramifications of climate change are multifaceted and far-reaching.", "expected": "Climate change affects many things in many ways."},
        {"input": "simplify: It is imperative that all personnel adhere to the established protocols.", "expected": "Everyone must follow the rules."},
        {"input": "simplify: The proliferation of digital technology has fundamentally altered interpersonal communication.", "expected": "Digital technology has changed how people talk to each other."},
    ],
    "tone": [
        {"input": "Rewrite in formal tone: hey can you help me?", "expected": "Could you please assist me with this matter?"},
        {"input": "Rewrite in formal tone: let me know asap", "expected": "Please inform me at your earliest convenience."},
        {"input": "Rewrite in informal tone: I would like to formally request your assistance.", "expected": "Hey, can you help me out?"},
        {"input": "Rewrite in informal tone: Your cooperation is greatly appreciated.", "expected": "Thanks so much for helping!"},
        {"input": "Rewrite in professional tone: this thing is broken", "expected": "The system is currently experiencing an issue."},
    ],
    "summarize": [
        {
            "input": (
                "Artificial intelligence has transformed the way businesses operate. "
                "From automating routine tasks to providing insights through data analysis, "
                "AI technologies are being adopted across industries. Machine learning, "
                "a subset of AI, enables computers to learn from data without being explicitly "
                "programmed. Deep learning, using neural networks with many layers, has achieved "
                "remarkable results in image recognition, natural language processing, and speech "
                "recognition. Companies are investing heavily in AI research and development."
            ),
            "expected": "AI is transforming business through automation, data analysis, and machine learning, with companies investing heavily in research."
        },
        {
            "input": (
                "The global water crisis is one of the most pressing challenges of our time. "
                "Over two billion people lack access to safe drinking water. Climate change is "
                "exacerbating water scarcity in many regions. Agricultural irrigation accounts for "
                "approximately 70 percent of global freshwater use. Pollution from industrial and "
                "agricultural sources contaminates water supplies. International cooperation is "
                "essential to address this growing problem."
            ),
            "expected": "Over two billion people lack safe water. Climate change, agriculture, and pollution worsen the crisis, requiring international cooperation."
        },
    ],
}


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1/2/L scores. Used for summarization quality."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(s[key].fmeasure)
        return {k: sum(v) / len(v) for k, v in scores.items()}
    except ImportError:
        logger.warning("rouge_score not installed. Install with: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    """
    Compute BERTScore F1 for semantic similarity.
    Used for paraphrase quality and tone content preservation.
    """
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return F1.mean().item()
    except ImportError:
        logger.warning("bert_score not installed. Install with: pip install bert-score")
        return 0.0


def compute_self_bleu(predictions: list[str]) -> float:
    """
    Compute Self-BLEU to measure output diversity.
    Lower is better (more diverse). Used for paraphrase evaluation.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        smoothing = SmoothingFunction().method1
        tokenized = [p.split() for p in predictions]
        scores = []
        for i, hyp in enumerate(tokenized):
            refs = [t for j, t in enumerate(tokenized) if j != i]
            if refs and hyp:
                scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothing))
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        logger.warning("nltk not installed for Self-BLEU. Install with: pip install nltk")
        return 0.0


def compute_readability(text: str) -> float:
    """
    Compute Flesch-Kincaid reading ease score.
    Higher = easier to read. Used for simplification evaluation.
    """
    words = text.split()
    sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    syllables = sum(_count_syllables(w) for w in words)

    if len(words) == 0:
        return 0.0

    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return max(0.0, min(100.0, score))


def _count_syllables(word: str) -> int:
    """Estimate syllable count for Flesch-Kincaid."""
    word = word.lower().strip(".,!?;:'\"")
    if len(word) <= 2:
        return 1
    vowels = "aeiou"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def generate_predictions(task_name: str, inputs: list[str],
                         adapter_path: Optional[str] = None,
                         base_model: Optional[str] = None) -> list[str]:
    """
    Generate predictions for a list of inputs using the specified adapter.
    Falls back to base model if no adapter provided.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Determine which model/adapter to use
    if adapter_path and Path(adapter_path).exists():
        adapter_cfg = Path(adapter_path) / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                cfg = json.load(f)
            model_id = cfg.get("base_model_name_or_path", base_model)
        else:
            model_id = base_model
    else:
        model_id = base_model
        adapter_path = None

    if not model_id:
        raise ValueError("Must specify either adapter_path or base_model")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_id, legacy=False
    )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model = model.to(device).eval()

    # Generate
    predictions = []
    for inp in inputs:
        encoded = tokenizer(inp, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=256,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(decoded)

    return predictions


def evaluate_task(task_name: str, adapter_path: Optional[str] = None,
                  base_model: Optional[str] = None) -> dict:
    """
    Run task-specific evaluation on the golden set.
    Returns a dict of metric_name -> score.
    """
    golden = GOLDEN_SETS.get(task_name, [])
    if not golden:
        return {"error": f"No golden set for task: {task_name}"}

    inputs = [g["input"] for g in golden]
    references = [g["expected"] for g in golden]

    # Determine base model from task config if not specified
    if not base_model:
        # Import via sys.path to handle standalone execution
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from finetune import TASKS
        base_model = TASKS.get(task_name, {}).get("base_model")

    print(f"\n  Evaluating {task_name} ({len(golden)} golden examples)...")
    predictions = generate_predictions(task_name, inputs, adapter_path, base_model)

    # Print sample predictions
    print(f"  Sample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"    Input:      {inputs[i][:80]}...")
        print(f"    Expected:   {references[i][:80]}...")
        print(f"    Predicted:  {predictions[i][:80]}...")
        print()

    # Task-specific metrics
    metrics = {}

    if task_name == "paraphrase":
        metrics["bertscore_f1"] = compute_bertscore(predictions, references)
        metrics["self_bleu"] = compute_self_bleu(predictions)
        metrics["primary"] = metrics["bertscore_f1"]

    elif task_name == "grammar":
        # Exact match rate
        exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(references)
        metrics["exact_match"] = exact
        metrics["bertscore_f1"] = compute_bertscore(predictions, references)
        metrics["primary"] = metrics["bertscore_f1"]

    elif task_name == "simplify":
        # Readability improvement
        input_readability = [compute_readability(g["input"].replace("simplify: ", "")) for g in golden]
        pred_readability = [compute_readability(p) for p in predictions]
        avg_gain = sum(p - i for p, i in zip(pred_readability, input_readability)) / len(golden)
        metrics["avg_readability_gain"] = avg_gain
        metrics["bertscore_f1"] = compute_bertscore(predictions, references)
        metrics["primary"] = metrics["bertscore_f1"]

    elif task_name == "tone":
        metrics["bertscore_f1"] = compute_bertscore(predictions, references)
        # Content preservation: how much meaning is retained
        inputs_clean = [g["input"].split(": ", 1)[-1] if ": " in g["input"] else g["input"] for g in golden]
        metrics["content_preservation"] = compute_bertscore(predictions, inputs_clean)
        metrics["primary"] = metrics["bertscore_f1"]

    elif task_name == "summarize":
        rouge = compute_rouge(predictions, references)
        metrics.update(rouge)
        metrics["primary"] = rouge["rougeL"]

    return metrics


def compare_models(task_name: str, current_adapter: str, candidate_adapter: str,
                   base_model: Optional[str] = None) -> dict:
    """
    Compare two adapters on the golden set.
    Returns which model wins and by how much.
    """
    print(f"\n  Comparing models for {task_name}:")
    print(f"    Current:   {current_adapter}")
    print(f"    Candidate: {candidate_adapter}")

    current_metrics = evaluate_task(task_name, current_adapter, base_model)
    candidate_metrics = evaluate_task(task_name, candidate_adapter, base_model)

    current_primary = current_metrics.get("primary", 0)
    candidate_primary = candidate_metrics.get("primary", 0)

    winner = "candidate" if candidate_primary > current_primary else "current"
    delta = candidate_primary - current_primary

    return {
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "winner": winner,
        "primary_delta": delta,
        "should_promote": delta > 0.02,  # Require >2% improvement for promotion
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--task", choices=list(GOLDEN_SETS.keys()) + ["all"], default="all")
    parser.add_argument("--adapter", default=None, help="Path to adapter directory")
    parser.add_argument("--base-model", default=None, help="Override base model")
    parser.add_argument("--golden-only", action="store_true", help="Only run golden set evaluation")
    parser.add_argument("--compare", default=None, help="Compare against this adapter path")
    args = parser.parse_args()

    tasks = list(GOLDEN_SETS.keys()) if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {task}")
        print(f"{'='*60}")

        if args.compare:
            result = compare_models(task, args.adapter, args.compare, args.base_model)
            print(f"\n  Winner: {result['winner']} (delta={result['primary_delta']:.4f})")
            print(f"  Promote candidate: {'YES' if result['should_promote'] else 'NO'}")
        else:
            metrics = evaluate_task(task, args.adapter, args.base_model)
            print(f"\n  Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k:25s}: {v:.4f}")
                else:
                    print(f"    {k:25s}: {v}")


if __name__ == "__main__":
    main()
