import os
import gc
import time
import copy
import logging
import traceback
from datetime import datetime

import torch

from generate_pipeline import run_pipeline, CONFIG

os.makedirs("./logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./logs/batch_experiment_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


LLAMA_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct"
]

MISTRAL_MODELS = [
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "mistralai/Ministral-3-14B-Instruct-2512-BF16"
]

QWEN_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-35B-A3B"
]

DATASET_CONFIGS = {
    "trivia_qa": 1024,
    "squad_v2": 1024,
    "arc_challenge": 1024,
    "truthful_qa": 1024,
    "coqa": 1024,
    "xsum": 1024,
    "halueval_qa": 1024,
    "gsm8k": 8192,
    "human_eval": 8192,
    "mbpp": 8192,
    "xlam_agent": 8192
}

COMMON_CONFIG = {
    "dataset_split": "train",
    "max_samples": 10000,
    "system_prompt": "You are a helpful, accurate, and honest AI assistant.",
    "num_shots": 4,
    "model_kwargs": {
        "trust_remote_code": True,
        "attn_implementation": "flex_attention"
    },
    "generation_kwargs": {
        "do_sample": False,
    },
    "layer_config": {"mode": "middle", "count": 5},
    "token_config": {"mode": "backward", "count": 5},
    "queue_size": 50,
    "judge_model": "gpt-4o-mini",
    "judge_concurrency": 50,
    "base_output_dir": "./experiments"
}


def build_experiment_list():
    experiments = []

    model_families = [
        (LLAMA_MODELS, {}),
        (QWEN_MODELS, {"enable_thinking": False}),
        (MISTRAL_MODELS, {}),
    ]

    for model_list, template_kwargs in model_families:
        for model_name in model_list:
            for dataset, max_tokens in DATASET_CONFIGS.items():
                exp = copy.deepcopy(COMMON_CONFIG)
                exp.update({
                    "target_model": model_name,
                    "dataset_name": dataset,
                    "max_new_tokens": max_tokens,
                    "template_kwargs": copy.deepcopy(template_kwargs),
                })
                experiments.append(exp)

    return experiments


def main():
    experiment_list = build_experiment_list()

    logging.info(f"Starting batch experiment tasks, {len(experiment_list)} configurations found.")
    logging.info(f"Detailed log file: {log_filename}")

    total_start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, exp_overrides in enumerate(experiment_list):
        exp_id = f"Exp_{idx + 1:03d}/{len(experiment_list)}"
        model_tag = exp_overrides.get("target_model", "?")
        dataset_tag = exp_overrides.get("dataset_name", "?")

        logging.info("-" * 60)
        logging.info(f"[{exp_id}] Processing: {model_tag} on {dataset_tag}")

        current_config = copy.deepcopy(CONFIG)
        current_config = deep_update(current_config, exp_overrides)

        start_time = time.time()

        try:
            run_pipeline(custom_config=current_config)

            elapsed = time.time() - start_time
            logging.info(f"[{exp_id}] Successfully completed in {elapsed:.2f}s")
            success_count += 1

        except Exception as e:
            elapsed = time.time() - start_time
            fail_count += 1
            logging.error(f"[{exp_id}] Failed in {elapsed:.2f}s")
            logging.error(f"  Error summary: {e}")
            logging.error(f"  Full traceback:\n{traceback.format_exc()}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    total_time = time.time() - total_start_time
    logging.info("=" * 60)
    logging.info("Batch experiments finished!")
    logging.info(f"  Total time elapsed: {total_time / 3600:.2f} hours")
    logging.info(f"  Final stats: Success {success_count}, Failed {fail_count}")
    if fail_count > 0:
        logging.warning("  Please check the log file for failure details.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()