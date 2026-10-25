from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DATA = Path(os.environ.get("DATA_DIR", "output/fincomp"))
CKPT = Path(os.environ.get("CKPT_DIR", "checkpoints"))
SFT_EPOCHS = int(os.environ.get("SFT_EPOCHS", "2"))
DPO_EPOCHS = int(os.environ.get("DPO_EPOCHS", "1"))
MAX_LEN = int(os.environ.get("MAX_LEN", "3072"))

T4_COMMON = dict(
    backend="unsloth",
    precision="fp16",
    fp16=True,
    bf16=False,
    quantization={"load_in_4bit": True},
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    gradient_accumulation_steps=16,
    optim="adamw_8bit",
    gradient_checkpointing=True,
    use_gradient_checkpointing=True,
    logging_steps=5,
    save_steps=100,
    seed=13,
)


def prep_dpo(tokenizer_name: str) -> Path:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = DATA / "dpo_train_rendered.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for line in (DATA / "dpo_train.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            f.write(json.dumps({"prompt": prompt, "chosen": r["chosen"] + tok.eos_token, "rejected": r["rejected"] + tok.eos_token}) + "\n")
    return out


def sft() -> None:
    from aligntune.core.backend_factory import create_sft_trainer

    out = CKPT / "qwen3b-fincomp-sft"
    trainer = create_sft_trainer(
        model_name=BASE,
        dataset_name=str(DATA / "sft_train.jsonl"),
        output_dir=str(out),
        num_epochs=SFT_EPOCHS,
        batch_size=1,
        learning_rate=2e-4,
        max_seq_length=MAX_LEN,
        warmup_ratio=0.05,
        load_best_model_at_end=False,
        **T4_COMMON,
    )
    trainer.train()
    print(trainer.save_model(str(out)))


def dpo() -> None:
    from aligntune.core.backend_factory import create_rl_trainer

    sft_merged = CKPT / "qwen3b-fincomp-sft-merged"
    out = CKPT / "qwen3b-fincomp-dpo"
    trainer = create_rl_trainer(
        algorithm="dpo",
        model_name=str(sft_merged),
        dataset_name=str(prep_dpo(str(sft_merged))),
        output_dir=str(out),
        beta=0.1,
        max_seq_length=MAX_LEN,
        max_prompt_length=MAX_LEN - 768,
        batch_size=1,
        num_epochs=DPO_EPOCHS,
        learning_rate=2e-5,
        **T4_COMMON,
    )
    trainer.train()
    print(trainer.save_model(str(out)))


def merge(adapter: Path, base: Path | str, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (adapter / "adapter_config.json").exists():
        if adapter.resolve() != out.resolve():
            shutil.copytree(adapter, out, dirs_exist_ok=True)
        print(f"{adapter} has no adapter_config.json - treated as merged -> {out}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"merging on {device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.float16, device_map=device, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    print("merged in memory, saving ...", flush=True)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(tmp))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"merged {adapter} onto {base} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sft", "dpo", "merge"])
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.stage == "sft":
        sft()
    elif a.stage == "dpo":
        dpo()
    else:
        merge(a.adapter, a.base, a.out or Path(str(a.adapter) + "-merged"))
