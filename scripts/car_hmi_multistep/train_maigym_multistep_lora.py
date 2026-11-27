"""Multi-Turn & Multi-Image BF16 LoRA Fine-Tuning + CPU BF16 Merge for Tongyi-MAI/MAI-Gym-9B (v2).

Upgrades over v1 (`train_maigym_lora.py`):
  - Supports variable 1..N images per sample (e.g., 1 image for single-step grounding / visual
    readout, 2 sliding-window screenshots for multi-step E2E trajectories).
  - Supports multi-turn conversation history with exact assistant-only loss masking strictly on
    the FINAL assistant response (all earlier system/user/assistant turns masked to -100).
  - Evaluates overall val_loss AND per-category val_loss (`e2e`, `ground`, `readout`) so we can
    monitor joint multi-task convergence with zero catastrophic forgetting.
  - Full BF16 LoRA (`--bf16`, default on 48 GB RTX 4090 with FLA Gated DeltaNet kernels) targeting
    ALL 12 Qwen3.5 LLM linear modules + `visual.merger.linear_fc1/2`, then merges on CPU RAM to
    BF16 safetensors ready for zero-overhead vLLM serving.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import pathlib
import random
import time
from typing import Any

import bitsandbytes as bnb
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup


def load_jsonl(path: pathlib.Path, max_samples: int = 0, seed: int = 42) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples > 0 and max_samples < len(rows):
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    return rows


def to_processor_messages(raw_msgs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """将含 `<image>` 占位符的多轮消息转为 HF 多模态消息，并返回图像占位总数。"""
    out: list[dict[str, Any]] = []
    n_img = 0
    for m in raw_msgs:
        role = m["role"]
        c = m["content"]
        if role == "system":
            out.append({"role": "system", "content": c})
        elif role == "user":
            if "<image>" in c:
                parts: list[dict[str, Any]] = []
                chunks = c.split("<image>")
                for idx, chunk in enumerate(chunks):
                    if chunk:
                        parts.append({"type": "text", "text": chunk})
                    if idx < len(chunks) - 1:
                        parts.append({"type": "image"})
                        n_img += 1
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": "user", "content": [{"type": "text", "text": c}]})
        else:
            out.append({"role": role, "content": c})
    return out, n_img


def bucket_name(task_type: str) -> str:
    if task_type.startswith("e2e"):
        return "e2e"
    if task_type.startswith("ground"):
        return "ground"
    if task_type.startswith("readout"):
        return "readout"
    return "other"


class MultiStepCarHmiDataset(Dataset):
    def __init__(self, jsonl_path: pathlib.Path, root_dir: pathlib.Path, max_samples: int = 0) -> None:
        self.root = root_dir
        self.rows = load_jsonl(jsonl_path, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        imgs = [Image.open(self.root / rel).convert("RGB") for rel in row["images"]]
        return {
            "messages": row["messages"],
            "images": imgs,
            "bucket": bucket_name(str(row.get("task_type") or "")),
        }


class MultiTurnCollator:
    def __init__(self, processor: Any, enable_thinking: bool = False) -> None:
        self.processor = processor
        self.enable_thinking = enable_thinking
        self._checked_buckets: set[str] = set()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        assert len(batch) == 1, "Use batch_size=1 with gradient accumulation for exact variable-length multimodal packing"
        item = batch[0]
        msgs, n_placeholders = to_processor_messages(item["messages"])
        imgs: list[Image.Image] = item["images"]
        assert n_placeholders == len(imgs), f"Placeholder count {n_placeholders} != images count {len(imgs)}"
        assert msgs[-1]["role"] == "assistant", "Final message must be assistant"

        prompt_msgs = msgs[:-1]
        assistant_raw = msgs[-1]["content"]

        prompt_text = self.processor.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        completion = assistant_raw + "<|im_end|>\n"
        full_text = prompt_text + completion

        prompt_inputs = self.processor(text=[prompt_text], images=imgs if imgs else None, return_tensors="pt")
        full_inputs = self.processor(text=[full_text], images=imgs if imgs else None, return_tensors="pt")

        prompt_len = prompt_inputs["input_ids"].shape[1]
        bkt = item["bucket"]
        if bkt not in self._checked_buckets:
            assert torch.equal(full_inputs["input_ids"][:, :prompt_len], prompt_inputs["input_ids"]), (
                f"Prompt prefix mismatch on bucket={bkt}!"
            )
            ans_tokens = full_inputs["input_ids"][0, prompt_len:]
            print(
                f"[SanityCheck:{bkt}] imgs={len(imgs)} prompt_len={prompt_len} ans_len={len(ans_tokens)} "
                f"target={self.processor.tokenizer.decode(ans_tokens)!r}",
                flush=True,
            )
            self._checked_buckets.add(bkt)

        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full_inputs["labels"] = labels
        full_inputs["_bucket"] = bkt
        return full_inputs


@torch.no_grad()
def evaluate_loss(model: Any, loader: DataLoader, device: torch.device, max_batches: int = 240) -> tuple[float, dict[str, float]]:
    model.eval()
    all_losses: list[float] = []
    by_bkt: dict[str, list[float]] = collections.defaultdict(list)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        bkt = batch.pop("_bucket", "other")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        lv = float(out.loss.item())
        all_losses.append(lv)
        by_bkt[bkt].append(lv)
    model.train()
    mean_all = sum(all_losses) / max(1, len(all_losses))
    mean_bkt = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_bkt.items())}
    return mean_all, mean_bkt


def merge_to_bf16(base_model_dir: str, adapter_dir: pathlib.Path, merged_dir: pathlib.Path) -> None:
    print(f"[Merge] Loading BF16 base model on CPU from {base_model_dir} ...", flush=True)
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(base_model_dir)
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_dir,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"[Merge] Attaching LoRA adapter from {adapter_dir} ...", flush=True)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Merge] Saving merged BF16 model to {merged_dir} ...", flush=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(str(merged_dir))
    print(f"[Merge] Done in {time.monotonic() - t0:.1f}s -> {merged_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ymtc2t/models/MAI-Gym-9B")
    ap.add_argument("--data-dir", default="/mnt/ymtc2t/datasets/carhmi_maigym_v2")
    ap.add_argument("--out-dir", default="/mnt/ymtc2t/outputs/maigym9b_carhmi_v2")
    ap.add_argument("--merged-dir", default="/mnt/ymtc2t/models/MAI-Gym-9B-carhmi-v2")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=60)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    data_dir = pathlib.Path(a.data_dir)
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best_adapter"

    processor = AutoProcessor.from_pretrained(a.model)
    train_ds = MultiStepCarHmiDataset(data_dir / "train.jsonl", data_dir, max_samples=a.max_train)
    val_ds = MultiStepCarHmiDataset(data_dir / "val.jsonl", data_dir, max_samples=a.max_val)
    collator = MultiTurnCollator(processor, enable_thinking=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=collator)

    if a.qlora:
        print(f"[Load] Loading 4-bit NF4 QLoRA {a.model} ...", flush=True)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        model = AutoModelForImageTextToText.from_pretrained(
            a.model, quantization_config=bnb_cfg, dtype=torch.bfloat16, device_map={"": 0}
        )
    else:
        print(f"[Load] Loading full BF16 {a.model} ...", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map={"": 0}
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    target_regex = (
        r".*(language_model\..*\.(q_proj|k_proj|v_proj|o_proj|"
        r"in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|"
        r"gate_proj|up_proj|down_proj)|visual\.merger\.(linear_fc1|linear_fc2))$"
    )
    lora_cfg = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_regex,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    total_samples = int(len(train_ds) * a.epochs)
    total_steps = max(1, math.ceil(total_samples / a.grad_accum))
    warmup_steps = max(10, int(total_steps * 0.06))
    print(
        f"[Train] train={len(train_ds)} val={len(val_ds)} epochs={a.epochs} "
        f"total_samples={total_samples} grad_accum={a.grad_accum} total_steps={total_steps} lr={a.lr}",
        flush=True,
    )

    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = torch.device("cuda:0")
    init_val, init_bkt = evaluate_loss(model, val_loader, device, max_batches=min(240, len(val_ds)))
    print(
        f"[Eval step=0] val_loss={init_val:.4f} by_task={init_bkt} "
        f"vram_peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB",
        flush=True,
    )
    best_val = init_val

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    seen = 0
    accum_loss = 0.0
    t0 = time.monotonic()
    log_rows: list[dict[str, Any]] = [{"step": 0, "val_loss": round(init_val, 4), "by_task": init_bkt}]

    while seen < total_samples:
        for batch in train_loader:
            if seen >= total_samples:
                break
            batch.pop("_bucket", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / a.grad_accum
            loss.backward()
            accum_loss += float(out.loss.item())
            seen += 1

            if seen % a.grad_accum == 0 or seen == total_samples:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                denom = a.grad_accum if seen % a.grad_accum == 0 else (seen % a.grad_accum)
                avg_loss = accum_loss / denom
                accum_loss = 0.0
                if step % 10 == 0 or step == 1 or step == total_steps:
                    dt = time.monotonic() - t0
                    eta = dt / step * (total_steps - step)
                    vram = torch.cuda.max_memory_allocated() / 1024**3
                    print(
                        f"[step {step:4d}/{total_steps}] loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"vram={vram:.2f}GB elapsed={dt/60:.1f}m eta={eta/60:.1f}m",
                        flush=True,
                    )
                    log_rows.append({"step": step, "train_loss": round(avg_loss, 4), "lr": scheduler.get_last_lr()[0]})

                if step % a.eval_steps == 0 or step == total_steps:
                    val_loss, val_bkt = evaluate_loss(model, val_loader, device, max_batches=min(240, len(val_ds)))
                    print(f"[Eval step={step}] val_loss={val_loss:.4f} by_task={val_bkt} (best={best_val:.4f})", flush=True)
                    log_rows.append({"step": step, "val_loss": round(val_loss, 4), "by_task": val_bkt})
                    if val_loss < best_val:
                        best_val = val_loss
                        model.save_pretrained(str(best_dir))
                        print(f"  -> saved new best adapter to {best_dir}", flush=True)

    final_dir = out_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    if not best_dir.exists():
        model.save_pretrained(str(best_dir))
    (out_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Train] Finished in {(time.monotonic() - t0)/60:.1f}m! best_val={best_val:.4f}", flush=True)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    if a.merged_dir:
        merge_to_bf16(a.model, best_dir, pathlib.Path(a.merged_dir))


if __name__ == "__main__":
    main()
