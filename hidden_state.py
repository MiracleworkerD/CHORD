import os
import json
import torch
import h5py
import multiprocessing as mp
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import ml_dtypes
import time
import numpy as np
import gc
from prompt_builder import LLMPromptBuilder


def hdf5_writer_process_worker(data_queue, h5_filepath, jsonl_filepath):
    print(f"[Writer Process] Started (PID: {os.getpid()}). Dedicated to I/O.")
    with open(jsonl_filepath, 'a', encoding='utf-8') as f_json, \
            h5py.File(h5_filepath, 'a') as f_h5:

        while True:
            item = data_queue.get()
            if item is None:
                break

            sample_id = item["sample_id"]
            try:
                if sample_id in f_h5:
                    del f_h5[sample_id]

                grp = f_h5.create_group(sample_id)
                grp.attrs["original_prompt"] = item["prompt"]
                grp.attrs["model_output"] = item["meta_item"]["model_output_text"]

                tokens_grp = grp.create_group("generated_tokens")
                for token_data in item["filtered_tokens_data"]:
                    t_grp = tokens_grp.create_group(f"token_{token_data['forward_idx']:03d}")
                    t_grp.attrs["text"] = token_data["token_str"]
                    t_grp.attrs["forward_idx"] = token_data["forward_idx"]
                    t_grp.attrs["backward_idx"] = token_data["backward_idx"]
                    t_grp.attrs["token_id"] = token_data["token_id"]

                    for layer_name, tensor in token_data["states"].items():
                        t_grp.create_dataset(layer_name, data=tensor, compression="gzip")

                f_h5.flush()
                f_json.write(json.dumps(item["meta_item"], ensure_ascii=False) + '\n')
                f_json.flush()
            except Exception as e:
                print(f"[Writer Process] Error saving {sample_id}: {e}")

    print(f"[Writer Process] Finished and safely closed all file handles.")


class HiddenStateExtractor:
    def __init__(
            self,
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            device=None,
            dtype=torch.bfloat16,
            model_kwargs=None
    ):
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})

        if "device_map" in self.model_kwargs:
            self.device_map = self.model_kwargs.pop("device_map")
            print(f"[*] User specified device_map detected: {self.device_map}")
        else:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    self.device_map = "balanced_low_0"
                    print(f"[*] {gpu_count} GPUs detected, enabling automatic balanced slicing (device_map='balanced_low_0')")
                else:
                    self.device_map = device if device else "cuda:0"
                    print(f"[*] Single GPU environment detected, using device: {self.device_map}")
            else:
                self.device_map = "cpu"
                print("[!] No GPU detected, using CPU (performance will be significantly limited)")

        print(f"[*] Loading model {model_name}...")

        tokenizer_kwargs = {}
        if self.model_kwargs.get("trust_remote_code"):
            tokenizer_kwargs["trust_remote_code"] = True
        if "mistral" in self.model_name.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device_map,
                **self.model_kwargs
            )
        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                print(f"[!] Triggering interception mechanism! Starting dynamic architecture sniffing...")
                config = AutoConfig.from_pretrained(model_name, **self.model_kwargs)
                target_arch = config.architectures[0] if config.architectures else None
                if target_arch and hasattr(transformers, target_arch):
                    print(f"[+] Sniffing successful! Found architecture class: {target_arch}")
                    model_class = getattr(transformers, target_arch)
                    self.model = model_class.from_pretrained(
                        model_name, torch_dtype=dtype,
                        device_map=self.device_map, **self.model_kwargs
                    )
                else:
                    raise RuntimeError(f"Unable to load architecture {target_arch} via reflection")
            else:
                raise e

        self.model.eval()

        if hasattr(self.model.config, "num_hidden_layers"):
            self.total_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "num_hidden_layers"):
            self.total_layers = self.model.config.text_config.num_hidden_layers
        else:
            raise AttributeError("Could not find num_hidden_layers in model configuration")

        self._decoder_layers = self._find_decoder_layers()
        if self._decoder_layers is not None:
            print(f"[+] Hook mode ready: located decoder layers ({len(self._decoder_layers)} layers in total)")
        else:
            print(f"[!] Decoder layers not found, falling back to output_hidden_states mode")

        print(f"[+] Model loading complete! Total hidden layers: {self.total_layers}")

    def _find_decoder_layers(self):
        candidate_paths = [
            ('model', 'layers'),
            ('transformer', 'h'),
            ('gpt_neox', 'layers'),
        ]
        for path_parts in candidate_paths:
            obj = self.model
            try:
                for part in path_parts:
                    obj = getattr(obj, part)
                if hasattr(obj, '__getitem__') and hasattr(obj, '__len__'):
                    return obj
            except AttributeError:
                continue
        return None

    def _resolve_target_layers(self, layer_config):
        if not layer_config:
            return list(range(self.total_layers - 3, self.total_layers))
        if isinstance(layer_config, list):
            return [(l + self.total_layers if l < 0 else l) for l in layer_config]
        elif isinstance(layer_config, str) and layer_config == "all":
            return list(range(self.total_layers))
        elif isinstance(layer_config, dict) and layer_config.get("mode") == "middle":
            count = min(layer_config.get("count", 1), self.total_layers)
            start_idx = (self.total_layers // 2) - (count // 2)
            return list(range(start_idx, start_idx + count))
        else:
            raise ValueError(f"Unsupported layer config: {layer_config}")

    def _resolve_target_tokens(self, total_generated, token_config):
        if not token_config:
            token_config = {"mode": "backward", "count": 5}
        mode = token_config.get("mode", "all")
        count = min(token_config.get("count", total_generated), total_generated)
        if mode == "all":
            return list(range(total_generated))
        elif mode == "forward":
            return list(range(count))
        elif mode == "backward":
            start = max(0, total_generated - count)
            return list(range(start, total_generated))
        else:
            raise ValueError(f"Unsupported token config: {token_config}")

    def generate_and_extract(self, prompt, layer_config, token_config, max_new_tokens, generation_kwargs=None):
        target_layers = self._resolve_target_layers(layer_config)
        gen_kwargs = generation_kwargs or {}

        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            gpu_indices = [v for v in self.model.hf_device_map.values() if isinstance(v, int)]
            first_device = f"cuda:{min(gpu_indices)}" if gpu_indices else self.model.device
        else:
            first_device = self.model.device

        inputs = self.tokenizer(prompt, return_tensors="pt").to(first_device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            final_gen_kwargs = {
                **gen_kwargs,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "max_length": None,
                "return_dict_in_generate": True,
                "output_hidden_states": False,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            effective_do_sample = final_gen_kwargs.get(
                "do_sample", self.model.generation_config.do_sample
            )
            if not effective_do_sample:
                final_gen_kwargs["temperature"] = None
                final_gen_kwargs["top_p"] = None
                final_gen_kwargs["top_k"] = None

            gen_outputs = self.model.generate(**final_gen_kwargs)

        full_sequence = gen_outputs.sequences
        new_tokens = full_sequence[0, prompt_len:]
        total_generated = len(new_tokens)
        full_output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        target_indices = self._resolve_target_tokens(total_generated, token_config)
        if not target_indices:
            return [], full_output_text, total_generated

        max_step = max(target_indices)
        fwd_seq = full_sequence[:, :prompt_len + max_step + 1].clone()
        fwd_mask = torch.ones_like(fwd_seq)

        del gen_outputs, full_sequence, inputs, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        target_positions = [prompt_len + step for step in target_indices]

        captured_states = {}
        hooks = []

        try:
            if self._decoder_layers is not None:
                for l in target_layers:
                    def make_hook(layer_idx, positions):
                        def hook_fn(module, input, output):
                            captured_states[layer_idx] = (
                                output[0][0, positions, :].detach().cpu()
                            )
                        return hook_fn

                    h = self._decoder_layers[l].register_forward_hook(
                        make_hook(l, target_positions)
                    )
                    hooks.append(h)

                with torch.no_grad():
                    self.model(fwd_seq, attention_mask=fwd_mask)
            else:
                with torch.no_grad():
                    fwd_out = self.model(
                        fwd_seq, attention_mask=fwd_mask, output_hidden_states=True
                    )
                for l in target_layers:
                    captured_states[l] = (
                        fwd_out.hidden_states[l + 1][0, target_positions, :].detach().cpu()
                    )
                del fwd_out
        finally:
            for h in hooks:
                h.remove()
            del fwd_seq, fwd_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        layer_tensors = [captured_states[l] for l in target_layers]
        mega_tensor_cpu = torch.stack(layer_tensors).float().numpy().astype(ml_dtypes.bfloat16)

        filtered_tokens_data = []
        for i, step in enumerate(target_indices):
            token_id = new_tokens[step].item()
            token_str = self.tokenizer.decode([token_id])
            step_states = {}
            for j, l in enumerate(target_layers):
                step_states[f"layer_{l:02d}"] = mega_tensor_cpu[j, i, :]

            filtered_tokens_data.append({
                "token_id": token_id,
                "token_str": token_str,
                "forward_idx": step,
                "backward_idx": step - total_generated,
                "states": step_states
            })

        return filtered_tokens_data, full_output_text, total_generated

    def process_from_file(
            self,
            input_jsonl_path: str,
            output_h5_path: str,
            output_jsonl_path: str,
            layer_config: dict = None,
            token_config: dict = None,
            max_new_tokens: int = 20,
            max_queue_size: int = 10,
            system_prompt: str = "You are a helpful assistant.",
            num_shots: int = 0,
            generation_kwargs: dict = None,
            template_kwargs: dict = None
    ):
        gen_kwargs = generation_kwargs or {}
        tpl_kwargs = template_kwargs or {}

        if not os.path.exists(input_jsonl_path):
            raise FileNotFoundError(f"Input file not found: {input_jsonl_path}")

        os.makedirs(os.path.dirname(output_h5_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

        failed_log_path = os.path.splitext(output_h5_path)[0] + "_failed.jsonl"

        existing_ids = set()
        if os.path.exists(output_h5_path):
            try:
                with h5py.File(output_h5_path, 'r') as f:
                    existing_ids = set(f.keys())
            except Exception as e:
                print(f"[!] Failed to read existing HDF5 file ({e}), starting from scratch.")

        failed_ids = set()
        if os.path.exists(failed_log_path):
            try:
                with open(failed_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            failed_ids.add(json.loads(line)["sample_id"])
            except Exception:
                pass

        skip_ids = existing_ids | failed_ids
        if skip_ids:
            print(
                f"[*] Resuming pipeline: {len(existing_ids)} completed, "
                f"{len(failed_ids)} historical failures. Both will be skipped."
            )

        with h5py.File(output_h5_path, 'a') as f:
            f.attrs["llm_model"] = self.model_name
            f.attrs["dataset_source"] = os.path.basename(input_jsonl_path)
            f.attrs["layer_config"] = json.dumps(layer_config or {})
            f.attrs["token_config"] = json.dumps(token_config or {})
            f.attrs["model_kwargs"] = json.dumps(self.model_kwargs)
            f.attrs["generation_kwargs"] = json.dumps(gen_kwargs)
            f.attrs["template_kwargs"] = json.dumps(tpl_kwargs)
            f.attrs["prompt_config"] = json.dumps({"system_prompt": system_prompt, "num_shots": num_shots})
            f.attrs["extraction_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        data_queue = mp.Queue(maxsize=max_queue_size)
        writer_process = mp.Process(
            target=hdf5_writer_process_worker,
            args=(data_queue, output_h5_path, output_jsonl_path)
        )
        writer_process.daemon = True
        writer_process.start()

        prompt_builder = LLMPromptBuilder(
            model_name=self.model_name,
            global_system_prompt=system_prompt,
            num_shots=num_shots,
            template_kwargs=tpl_kwargs
        )

        dataset_items = []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset_items.append(json.loads(line))

        processed_count = 0
        skipped_count = 0
        error_count = 0

        try:
            for item in dataset_items:
                sample_id = item["sample_id"]

                if sample_id in skip_ids:
                    skipped_count += 1
                    continue

                try:
                    print(f"[Main Process] Inferencing {sample_id}...")

                    prompt_str = prompt_builder.build_prompt(
                        target_item=item, few_shot_pool=dataset_items
                    )

                    filtered_tokens_data, model_output_text, total_generated = (
                        self.generate_and_extract(
                            prompt_str, layer_config, token_config,
                            max_new_tokens, generation_kwargs=gen_kwargs
                        )
                    )

                    meta_item = item.copy()
                    meta_item.update({
                        "prompt": prompt_str,
                        "model_output_text": model_output_text,
                        "total_generated_tokens": total_generated,
                        "saved_tokens_count": len(filtered_tokens_data),
                        "h5_file_reference": output_h5_path
                    })

                    data_queue.put({
                        "sample_id": sample_id,
                        "prompt": prompt_str,
                        "filtered_tokens_data": filtered_tokens_data,
                        "meta_item": meta_item
                    }, block=True)

                    processed_count += 1

                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    print(f"[!] Processing failed for sample {sample_id}, skipping: {error_msg}")

                    try:
                        with open(failed_log_path, 'a', encoding='utf-8') as f_fail:
                            f_fail.write(json.dumps({
                                "sample_id": sample_id,
                                "error": error_msg,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }, ensure_ascii=False) + '\n')
                    except Exception:
                        pass

                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\n[!] Keyboard interrupt received! Performing safe shutdown...")

        finally:
            print(f"[*] Waiting for I/O process to save and close files...")
            data_queue.put(None)
            writer_process.join(timeout=30)
            if writer_process.is_alive():
                print("[!] I/O process did not exit in time, forcing termination.")
                writer_process.terminate()
            print(
                f"[+] Extraction pipeline exited! "
                f"Success: {processed_count}, Skipped: {skipped_count}, Failed: {error_count}"
            )


if __name__ == "__main__":
    TEST_DIR = "./TEST"
    os.makedirs(TEST_DIR, exist_ok=True)
    test_input_file = os.path.join(TEST_DIR, "test_dataset.jsonl")
    test_output_h5 = os.path.join(TEST_DIR, "output_tensors.h5")
    test_output_jsonl = os.path.join(TEST_DIR, "output_metadata.jsonl")

    mock_dataset = [
        {
            "sample_id": "test_qa_001",
            "structured_data": {
                "task_type": "qa", "system_instruction": "", "context": "",
                "question": "Capital of France?", "choices": {},
                "ground_truths": ["Paris"], "incorrect_answers": []
            },
            "original_doc": {}
        },
    ]

    with open(test_input_file, "w", encoding="utf-8") as f:
        for item in mock_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    extractor = HiddenStateExtractor(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_kwargs={"trust_remote_code": True}
    )
    extractor.process_from_file(
        input_jsonl_path=test_input_file,
        output_h5_path=test_output_h5,
        output_jsonl_path=test_output_jsonl,
        layer_config={"mode": "middle", "count": 2},
        token_config={"mode": "backward", "count": 2},
        max_new_tokens=10,
    )
    print(f"\n[+] Unit test passed!")