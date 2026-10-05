import os
import json
import argparse
from typing import Dict, Any, Optional
from datasets import load_dataset, get_dataset_split_names
import random


class BaseDatasetAdapter:
    dataset_path: str = ""
    dataset_name: str = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class TruthfulQAAdapter(BaseDatasetAdapter):
    dataset_path = "truthful_qa"
    dataset_name = "generation"

    def extract_structured_data(self, row):
        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": "",
            "question": row['question'],
            "choices": {},
            "ground_truths": row['correct_answers'],
            "incorrect_answers": row['incorrect_answers']
        }


class HaluEvalQAAdapter(BaseDatasetAdapter):
    dataset_path = "pminervini/HaluEval"
    dataset_name = "qa"

    def extract_structured_data(self, row):
        return {
            "task_type": "qa",
            "system_instruction": "Please answer the question based strictly on the provided context.",
            "context": row['knowledge'],
            "question": row['question'],
            "choices": {},
            "ground_truths": [row['right_answer']],
            "incorrect_answers": [row.get('hallucinated_answer', "")]
        }


class TriviaQAAdapter(BaseDatasetAdapter):
    dataset_path = "trivia_qa"
    dataset_name = "rc.nocontext"

    def extract_structured_data(self, row):
        valid_answers = row['answer'].get('aliases', [row['answer']['value']])
        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": "",
            "question": row['question'],
            "choices": {},
            "ground_truths": valid_answers,
            "incorrect_answers": []
        }


class CoQAAdapter(BaseDatasetAdapter):
    dataset_path = "coqa"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": row['story'],
            "question": row['questions'][0],
            "choices": {},
            "ground_truths": [row['answers']['input_text'][0]],
            "incorrect_answers": []
        }


class SQuADv2Adapter(BaseDatasetAdapter):
    dataset_path = "squad_v2"
    dataset_name = None

    def extract_structured_data(self, row):
        answers = row['answers']['text']
        if len(answers) == 0:
            answers = ["I don't know.", "Unanswerable"]

        return {
            "task_type": "qa",
            "system_instruction": "",
            "context": row['context'],
            "question": row['question'],
            "choices": {},
            "ground_truths": answers,
            "incorrect_answers": []
        }


class ARCChallengeAdapter(BaseDatasetAdapter):
    dataset_path = "ai2_arc"
    dataset_name = "ARC-Challenge"

    def extract_structured_data(self, row):
        choices_dict = {
            label: text for label, text in zip(row['choices']['label'], row['choices']['text'])
        }
        return {
            "task_type": "multiple_choice",
            "system_instruction": "",
            "context": "",
            "question": row['question'],
            "choices": choices_dict,
            "ground_truths": [row['answerKey']],
            "incorrect_answers": []
        }


class XSumAdapter(BaseDatasetAdapter):
    dataset_path = "xsum"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "summarization",
            "system_instruction": "",
            "context": row['document'],
            "question": "Please summarize the above document in one sentence.",
            "choices": {},
            "ground_truths": [row['summary']],
            "incorrect_answers": []
        }


class GSM8KAdapter(BaseDatasetAdapter):
    dataset_path = "gsm8k"
    dataset_name = "main"

    def extract_structured_data(self, row):
        return {
            "task_type": "reasoning",
            "system_instruction": "Please think step by step and provide the final answer at the end.",
            "context": "",
            "question": row['question'],
            "choices": {},
            "ground_truths": [row['answer']],
            "incorrect_answers": []
        }

class HumanEvalAdapter(BaseDatasetAdapter):
    dataset_path = "openai_humaneval"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "coding",
            "system_instruction": "You are an expert Python developer. Complete the provided Python function based on the docstring. Only output valid Python code.",
            "context": "",
            "question": row['prompt'],
            "choices": {},
            "ground_truths": [row['canonical_solution']],
            "incorrect_answers": []
        }


class XLamFunctionCallingAdapter(BaseDatasetAdapter):
    dataset_path = "Salesforce/xlam-function-calling-60k"
    dataset_name = None

    def extract_structured_data(self, row):
        return {
            "task_type": "agent_action",
            "system_instruction": "You are a helpful assistant with access to various tools. Based on the User's question, select the appropriate tool from the Context and output the exact tool call in JSON format. If no tool is needed, answer directly.",
            "context": f"Available Tools:\n{row['tools']}",
            "question": row['query'],
            "choices": {},
            "ground_truths": [row['answers']],
            "incorrect_answers": []
        }


class CoQAMultiTurnAdapter(BaseDatasetAdapter):
    dataset_path = "coqa"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        story = row['story']
        questions = row['questions']
        answers = row['answers']['input_text']
        num_turns = len(questions)

        if num_turns <= 1:
            return {
                "task_type": "qa",
                "system_instruction": "Read the story carefully and answer the question.",
                "context": f"Story:\n{story}",
                "question": questions[0],
                "choices": {},
                "ground_truths": [answers[0]],
                "incorrect_answers": []
            }

        target_turn_idx = random.randint(1, num_turns - 1)
        history_blocks = []
        for i in range(target_turn_idx):
            history_blocks.append(f"User: {questions[i]}\nAssistant: {answers[i]}")

        history_str = "\n\n".join(history_blocks)
        composite_context = (
            f"=== Background Story ===\n{story}\n\n"
            f"=== Conversation History ===\n{history_str}"
        )

        return {
            "task_type": "qa",
            "system_instruction": "Based on the Background Story and the Conversation History, answer the user's latest question accurately.",
            "context": composite_context,
            "question": questions[target_turn_idx],
            "choices": {},
            "ground_truths": [answers[target_turn_idx]],
            "incorrect_answers": []
        }


class HaluBenchAdapter(BaseDatasetAdapter):
    dataset_path = "PatronusAI/HaluBench"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ctx = row.get("context") or row.get("Context") or row.get("passage") or ""
        q = row.get("question") or row.get("Question") or ""
        ans = row.get("answer") or row.get("Answer") or row.get("reference_answer") or row.get("ground_truth") or ""

        label_raw = row.get("is_hallucination", row.get("hallucination", row.get("label")))
        is_hallucinated = str(label_raw).lower() in ["true", "1", "yes"]

        ground_truths = []
        incorrect_answers = []

        if is_hallucinated:
            incorrect_answers.append(ans)
        else:
            ground_truths.append(ans)

        return {
            "task_type": "qa",
            "system_instruction": "Please answer the question based strictly on the provided context.",
            "context": ctx,
            "question": q,
            "choices": {},
            "ground_truths": ground_truths,
            "incorrect_answers": incorrect_answers
        }


class HotpotQAAdapter(BaseDatasetAdapter):
    dataset_path = "hotpotqa/hotpot_qa"
    dataset_name = "fullwiki"

    def _flatten_context(self, ctx: Any, max_sent_per_title: int = 8) -> str:
        if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
            blocks = []
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                blocks.append(f"[{title}] " + " ".join(sents[:max_sent_per_title]))
            return "\n".join(blocks)

        if isinstance(ctx, list):
            blocks = []
            for item in ctx:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    title, sents = item, item
                    if isinstance(sents, list):
                        blocks.append(f"[{title}] " + " ".join(sents[:max_sent_per_title]))
            if blocks:
                return "\n".join(blocks)
        return "" if ctx is None else str(ctx)

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        context_text = self._flatten_context(row.get("context"))

        return {
            "task_type": "qa",
            "system_instruction": "Answer the question by synthesizing information from the multiple provided contexts.",
            "context": context_text,
            "question": row.get("question", ""),
            "choices": {},
            "ground_truths": [row.get("answer", "")],
            "incorrect_answers": []
        }


class CommonsenseQAAdapter(BaseDatasetAdapter):
    dataset_path = "tau/commonsense_qa"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        choices_raw = row.get("choices", {})
        labels = choices_raw.get("label", [])
        texts = choices_raw.get("text", [])
        choices_dict = {str(lab): str(txt) for lab, txt in zip(labels, texts)}

        return {
            "task_type": "multiple_choice",
            "system_instruction": "Use your common sense to select the most appropriate option.",
            "context": "",
            "question": row.get("question", ""),
            "choices": choices_dict,
            "ground_truths": [row.get("answerKey", "")],
            "incorrect_answers": []
        }


class MMLUAdapter(BaseDatasetAdapter):
    dataset_path = "cais/mmlu"
    dataset_name = "all"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        choices_list = row.get("choices") or row.get("options") or []
        choices_dict = {}
        for i, opt in enumerate(choices_list):
            label = chr(65 + i) if i < 26 else str(i)
            choices_dict[label] = str(opt)

        ans_raw = row.get("answer")
        ground_truth_label = ""
        if isinstance(ans_raw, int) and ans_raw < len(choices_list):
            ground_truth_label = chr(65 + ans_raw)
        else:
            ground_truth_label = str(ans_raw)

        subject = row.get("subject", "").replace("_", " ").title()
        instruction = f"This is a multiple-choice question about {subject}." if subject else ""

        return {
            "task_type": "multiple_choice",
            "system_instruction": instruction,
            "context": "",
            "question": row.get("question", ""),
            "choices": choices_dict,
            "ground_truths": [ground_truth_label],
            "incorrect_answers": []
        }


class BelebeleAdapter(BaseDatasetAdapter):
    dataset_path = "facebook/belebele"
    dataset_name = "eng_Latn"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        passage = row.get("passage") or row.get("context") or row.get("flores_passage") or ""
        options = []
        for k in ["mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4", "answer0", "answer1", "answer2", "answer3"]:
            if k in row and row[k]:
                options.append(row[k])

        if not options and "choices" in row:
            options = row["choices"]

        choices_dict = {chr(65 + i): str(opt) for i, opt in enumerate(options)}
        ans_raw = row.get("correct_answer_num", "")
        ground_truth_label = ""
        try:
            idx = int(ans_raw) - 1
            if 0 <= idx < len(options):
                ground_truth_label = chr(65 + idx)
        except (ValueError, TypeError):
            ground_truth_label = str(row.get("label") or row.get("answer") or "")

        return {
            "task_type": "multiple_choice",
            "system_instruction": "Read the passage carefully and select the correct answer.",
            "context": passage,
            "question": row.get("question", ""),
            "choices": choices_dict,
            "ground_truths": [ground_truth_label],
            "incorrect_answers": []
        }


class RAGTruthAdapter(BaseDatasetAdapter):
    dataset_path = "wandb/RAGTruth-processed"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ctx = row.get("context") or row.get("retrieved_context") or row.get("retrieved_docs") or ""
        prompt = row.get("prompt") or row.get("query") or row.get("question") or ""
        response = row.get("response") or row.get("generated_response") or row.get("output") or ""
        label = row.get("label") or row.get("is_hallucination")
        is_hallucinated = str(label).lower() in ["true", "1", "yes"]

        ground_truths = []
        incorrect_answers = []
        if is_hallucinated:
            incorrect_answers.append(response)
        else:
            ground_truths.append(response)

        return {
            "task_type": "qa",
            "system_instruction": "Evaluate or synthesize based on the retrieved context.",
            "context": ctx,
            "question": prompt,
            "choices": {},
            "ground_truths": ground_truths,
            "incorrect_answers": incorrect_answers
        }


class TheoremQAAdapter(BaseDatasetAdapter):
    dataset_path = "TIGER-Lab/TheoremQA"
    dataset_name = None

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        q = row.get("question") or row.get("problem") or ""
        theorem = row.get("theorem") or row.get("theorem_name") or row.get("topic") or ""
        ctx = f"Relevant Theorem/Topic: {theorem}" if theorem else ""

        return {
            "task_type": "reasoning",
            "system_instruction": "Apply the relevant mathematical or scientific theorem to solve the problem.",
            "context": ctx,
            "question": q,
            "choices": {},
            "ground_truths": [str(row.get("answer") or row.get("final_answer") or row.get("solution") or "")],
            "incorrect_answers": []
        }


class MATHAdapter(BaseDatasetAdapter):
    dataset_path = "HuggingFaceH4/MATH"
    dataset_name = "default"

    def extract_structured_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        problem = row.get("problem") or row.get("question") or ""
        solution = row.get("solution") or row.get("answer") or ""

        return {
            "task_type": "reasoning",
            "system_instruction": "Solve the mathematical problem step by step. Put your final answer in \\boxed{}.",
            "context": "",
            "question": problem,
            "choices": {},
            "ground_truths": [solution],
            "incorrect_answers": []
        }


class MBPPAdapter(BaseDatasetAdapter):
    dataset_path = "mbpp"
    dataset_name = "full"

    def extract_structured_data(self, row):
        tests_str = "\n".join(row['test_list'])
        return {
            "task_type": "coding",
            "system_instruction": "You are an expert Python programmer. Write a Python function to solve the problem. Your code must pass the provided assertion tests.",
            "context": f"Your code must pass the following tests:\n{tests_str}",
            "question": row['text'],
            "choices": {},
            "ground_truths": [row['code']],
            "incorrect_answers": []
        }


DATASET_REGISTRY = {
    "truthful_qa": TruthfulQAAdapter,
    "halueval_qa": HaluEvalQAAdapter,
    "trivia_qa": TriviaQAAdapter,
    "coqa": CoQAAdapter,
    "squad_v2": SQuADv2Adapter,
    "arc_challenge": ARCChallengeAdapter,
    "xsum": XSumAdapter,
    "gsm8k": GSM8KAdapter,
    "human_eval": HumanEvalAdapter,
    "xlam_agent": XLamFunctionCallingAdapter,
    "mbpp": MBPPAdapter,
    "coqa_multiturn": CoQAMultiTurnAdapter,
    "halubench": HaluBenchAdapter,
    "hotpotqa": HotpotQAAdapter,
    "commonsenseqa": CommonsenseQAAdapter,
    "mmlu": MMLUAdapter,
    "belebele": BelebeleAdapter,
    "ragtruth": RAGTruthAdapter,
    "theoremqa": TheoremQAAdapter,
    "math": MATHAdapter,
}


def process_dataset(
        adapter_name: str,
        output_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None
):
    if adapter_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset adapter not found: {adapter_name}. Available: {list(DATASET_REGISTRY.keys())}")

    adapter = DATASET_REGISTRY[adapter_name]()

    print(f"[*] Probing available splits for {adapter.dataset_path}...")
    try:
        available_splits = get_dataset_split_names(adapter.dataset_path, adapter.dataset_name)

        if split not in available_splits:
            print(f"[!] Warning: Target split '{split}' does not exist. Available splits: {available_splits}")
            fallback_priority = ["train", "validation", "test", "data"]
            new_split = None

            for fb in fallback_priority:
                if fb in available_splits:
                    new_split = fb
                    break

            if not new_split and len(available_splits) > 0:
                new_split = available_splits[0]

            print(f"[*] 🚀 Adaptive mechanism triggered, falling back to split: '{new_split}'")
            split = new_split

    except Exception as e:
        print(f"[!] Split probing failed, forcing load of '{split}'. Error: {e}")

    print(f"[*] Loading {adapter.dataset_path} via HuggingFace ({split} set)...")
    dataset = load_dataset(adapter.dataset_path, adapter.dataset_name, split=split)
    dataset = dataset.shuffle(seed=42)

    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{adapter_name}_{split}.jsonl"
    out_filepath = os.path.join(output_dir, out_filename)

    print(f"[*] Formatting data and exporting to {out_filepath} ...")

    processed_count = 0
    with open(out_filepath, 'w', encoding='utf-8') as f:
        for i, row in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break

            structured_data = adapter.extract_structured_data(row)

            item = {
                "sample_id": f"{adapter_name}_{split}_{i:06d}",
                "structured_data": structured_data,
                "original_doc": row
            }

            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"[+] Successfully processed and saved {processed_count} entries!")

    if max_samples is not None and processed_count < max_samples:
        print(f"[!] Note: Requested {max_samples} samples, but only {processed_count} were available in this split.")
    print("\n")
    return out_filepath


def main():
    DEFAULT_DATASET = "gsm8k"
    DEFAULT_OUTPUT_DIR = "./unified_datasets"
    DEFAULT_SPLIT = "validation"
    DEFAULT_MAX_SAMPLES = 5

    parser = argparse.ArgumentParser(description="Unified data processor based on Universal Schema")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Dataset name in registry (Default: {DEFAULT_DATASET})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for output JSONL (Default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        help=f"Dataset split e.g. train/validation/test (Default: {DEFAULT_SPLIT})")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"Max entries to process, use 0 for all (Default: {DEFAULT_MAX_SAMPLES})")

    args = parser.parse_args()
    max_samples = args.max_samples if args.max_samples > 0 else None

    process_dataset(
        adapter_name=args.dataset,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=max_samples
    )


if __name__ == "__main__":
    main()