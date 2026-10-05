import os
import json
import time
import gc

import h5py

from prepare_datasets import process_dataset
from hidden_state import HiddenStateExtractor
from llm_judge import run_llm_judge

CONFIG = {
    "dataset_name": "mbpp",
    "dataset_split": "train",
    "max_samples": 100,

    "target_model": "mistralai/Ministral-3-14B-Instruct-2512",
    "system_prompt": "You are a helpful, accurate, and honest AI assistant.",
    "num_shots": 2,
    "max_new_tokens": 8192,

    "model_kwargs": {
        "trust_remote_code": True,
        "attn_implementation": "flex_attention"
    },
    "generation_kwargs": {
        "do_sample": False,
    },
    "template_kwargs": {
        "enable_thinking": False
    },

    "layer_config": {"mode": "middle", "count": 5},
    "token_config": {"mode": "backward", "count": 5},
    "queue_size": 20,

    "judge_model": "gpt-4o-mini",
    "judge_concurrency": 50,

    "base_output_dir": "./experiments"
}


def count_jsonl_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_h5_samples(filepath):
    if not os.path.exists(filepath):
        return 0
    try:
        with h5py.File(filepath, 'r') as f:
            return len(f.keys())
    except Exception:
        return 0


def reconcile_metadata_with_h5(h5_path, metadata_jsonl_path, input_jsonl_path):
    if not os.path.exists(h5_path) or not os.path.exists(metadata_jsonl_path):
        return

    try:
        with h5py.File(h5_path, 'r') as f:
            h5_ids = set(f.keys())
    except Exception:
        return

    meta_by_id = {}
    with open(metadata_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                meta_by_id[item["sample_id"]] = item

    missing_ids = h5_ids - set(meta_by_id.keys())

    if not missing_ids and len(meta_by_id) == count_jsonl_lines(metadata_jsonl_path):
        return

    if missing_ids and os.path.exists(input_jsonl_path):
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = item["sample_id"]
                    if sid in missing_ids:
                        item["model_output_text"] = "[RECOVERED_FROM_H5]"
                        item["total_generated_tokens"] = -1
                        item["saved_tokens_count"] = -1
                        item["h5_file_reference"] = h5_path
                        meta_by_id[sid] = item

    with open(metadata_jsonl_path, 'w', encoding='utf-8') as f:
        for sid in sorted(meta_by_id.keys()):
            if sid in h5_ids:
                f.write(json.dumps(meta_by_id[sid], ensure_ascii=False) + '\n')

    repaired = len(missing_ids & set(meta_by_id.keys()))
    if repaired > 0 or len(meta_by_id) != count_jsonl_lines(metadata_jsonl_path):
        print(f"    [Reconcile] Repaired {repaired} missing metadata entries, deduplicated, and cleaned orphan entries.")


def build_paths(cfg):
    exp_dir = os.path.join(
        cfg["base_output_dir"],
        cfg["target_model"],
        f"{cfg['dataset_name']}_{cfg['max_samples']}samples"
    )
    os.makedirs(exp_dir, exist_ok=True)

    h5_path = os.path.join(exp_dir, "02_hidden_states.h5")

    return {
        "step1_unified_jsonl": os.path.join(exp_dir, "01_structured_data.jsonl"),
        "step2_tensor_h5": h5_path,
        "step2_metadata_jsonl": os.path.join(exp_dir, "02_extracted_metadata.jsonl"),
        "step2_failed_jsonl": os.path.splitext(h5_path)[0] + "_failed.jsonl",
        "step3_scored_jsonl": os.path.join(exp_dir, "03_final_scored_metadata.jsonl"),
        "step3_failed_jsonl": os.path.join(exp_dir, "03_judge_failed.jsonl"),
    }


def run_pipeline(custom_config=None):
    cfg = custom_config if custom_config is not None else CONFIG

    print("=" * 50)
    print("Starting end-to-end hallucination detection pipeline")
    print(f"  Dataset: {cfg['dataset_name']} | Expected samples: {cfg['max_samples']}")
    print(f"  Target model: {cfg['target_model']} (Few-shot: {cfg['num_shots']})")
    print(f"  Judge model: {cfg['judge_model']}")
    print("=" * 50 + "\n")

    paths = build_paths(cfg)
    expected_samples = cfg["max_samples"]

    print(">>> [Step 1/3] Extracting structured data...")
    s1_count = count_jsonl_lines(paths["step1_unified_jsonl"])

    if s1_count > 0:
        print(f"    {s1_count} structured data entries already exist, skipping.")
    else:
        if os.path.exists(paths["step1_unified_jsonl"]):
            os.remove(paths["step1_unified_jsonl"])

        out_path = process_dataset(
            adapter_name=cfg["dataset_name"],
            output_dir=os.path.dirname(paths["step1_unified_jsonl"]),
            split=cfg["dataset_split"],
            max_samples=expected_samples
        )
        if os.path.exists(out_path) and out_path != paths["step1_unified_jsonl"]:
            os.rename(out_path, paths["step1_unified_jsonl"])

        s1_count = count_jsonl_lines(paths["step1_unified_jsonl"])
        if s1_count == 0:
            raise RuntimeError(
                f"Step 1 produced 0 entries! Please check if the dataset '{cfg['dataset_name']}' "
                f"(split='{cfg['dataset_split']}') is available."
            )
        print(f"    Generation complete: {s1_count} entries.")

    if s1_count < expected_samples:
        print(f"    [Note] Dataset actually has only {s1_count} entries (expected {expected_samples}), proceeding with actual count.")

    print("\n>>> [Step 2/3] Extracting hidden states (GPU)...")
    s2_h5_count = count_h5_samples(paths["step2_tensor_h5"])
    s2_failed_count = count_jsonl_lines(paths["step2_failed_jsonl"])
    s2_total_handled = s2_h5_count + s2_failed_count

    if s2_total_handled >= s1_count:
        print(f"    Step 2 complete: extracted {s2_h5_count} entries, failed {s2_failed_count} entries. Skipping.")
    else:
        if s2_total_handled > 0:
            print(
                f"    Resuming: processed {s2_total_handled}/{s1_count} "
                f"(extracted {s2_h5_count}, failed {s2_failed_count}), continuing..."
            )

        extractor = HiddenStateExtractor(
            model_name=cfg["target_model"],
            model_kwargs=cfg.get("model_kwargs", {})
        )
        extractor.process_from_file(
            input_jsonl_path=paths["step1_unified_jsonl"],
            output_h5_path=paths["step2_tensor_h5"],
            output_jsonl_path=paths["step2_metadata_jsonl"],
            layer_config=cfg["layer_config"],
            token_config=cfg["token_config"],
            max_new_tokens=cfg["max_new_tokens"],
            max_queue_size=cfg["queue_size"],
            system_prompt=cfg["system_prompt"],
            num_shots=cfg["num_shots"],
            generation_kwargs=cfg.get("generation_kwargs", {}),
            template_kwargs=cfg.get("template_kwargs", {})
        )

        print("[*] Unloading model, releasing VRAM...")
        if hasattr(extractor, 'model'):
            del extractor.model
        if hasattr(extractor, 'tokenizer'):
            del extractor.tokenizer
        del extractor

        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[+] VRAM released.")

    reconcile_metadata_with_h5(
        paths["step2_tensor_h5"],
        paths["step2_metadata_jsonl"],
        paths["step1_unified_jsonl"]
    )

    print("\n>>> [Step 3/3] LLM Judge scoring...")
    s2_meta_count = count_jsonl_lines(paths["step2_metadata_jsonl"])

    if s2_meta_count == 0:
        print("    [Skip] No metadata to score (Step 2 may have completely failed).")
    else:
        s3_scored_count = count_jsonl_lines(paths["step3_scored_jsonl"])
        s3_failed_count = count_jsonl_lines(paths["step3_failed_jsonl"])
        s3_total_handled = s3_scored_count + s3_failed_count

        if s3_total_handled >= s2_meta_count:
            print(
                f"    Step 3 complete: scored {s3_scored_count} entries, "
                f"failed {s3_failed_count} entries. Skipping."
            )
        else:
            if s3_total_handled > 0:
                print(
                    f"    Resuming: processed {s3_total_handled}/{s2_meta_count}, continuing..."
                )

            run_llm_judge(
                input_filepath=paths["step2_metadata_jsonl"],
                output_filepath=paths["step3_scored_jsonl"],
                failed_filepath=paths["step3_failed_jsonl"],
                model_name=cfg["judge_model"],
                concurrency_limit=cfg["judge_concurrency"]
            )

    print("\n" + "=" * 50)
    print("Pipeline execution complete!")
    print(f"  Output directory: {os.path.dirname(paths['step2_tensor_h5'])}")
    print("=" * 50)


if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")