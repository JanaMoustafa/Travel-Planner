"""
QLoRA fine-tuning for the Solver using Unsloth when available.

Install training extras (see requirements-train.txt), then:
  pip install unsloth  # follow https://github.com/unslothai/unsloth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from travel_planner.config import DEFAULT_OUTPUT_DIR, DEFAULT_SFT_MODEL_NAME, HF_TOKEN


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_chat(tokenizer, instruction: str, output: str) -> str:
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return (
        f"### Instruction\n{instruction}\n\n"
        f"### Response\n{output}"
    )


def _train_with_transformers(args, rows: list[dict]) -> None:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    import torch

    logger_msg = "Falling back to standard Transformers + PEFT training"
    print(logger_msg)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        token=args.token,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(
        model,
        lora_config,
        autocast_adapter_dtype=False,
    )

    texts: list[str] = []
    for r in rows:
        inst = r.get("instruction", "")
        out = r.get("output", "")
        texts.append(_format_chat(tokenizer, inst, out))

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=not use_bf16,
            bf16=use_bf16,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            remove_unused_columns=False,
        ),
    )

    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT with Unsloth (Llama-3 / Mistral).")
    parser.add_argument("--train_file", type=Path, default=Path("artifacts/sft_train.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default=DEFAULT_SFT_MODEL_NAME)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional cap on total optimizer steps")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank for fine-tuning")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)")
    parser.add_argument("--token", default=HF_TOKEN)
    parser.add_argument(
        "--backend",
        choices=["auto", "unsloth", "transformers"],
        default="auto",
        help="Training backend. 'auto' tries Unsloth then falls back to Transformers/PEFT.",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.train_file)
    if not rows:
        raise SystemExit(f"No rows in {args.train_file}")

    if args.backend == "transformers":
        _train_with_transformers(args, rows)
        return

    try:
        from unsloth import FastLanguageModel  # type: ignore
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments

        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=args.token,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        texts: list[str] = []
        for r in rows:
            inst = r.get("instruction", "")
            out = r.get("output", "")
            texts.append(_format_chat(tokenizer, inst, out))

        ds = Dataset.from_dict({"text": texts})

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            args=TrainingArguments(
                output_dir=str(args.output_dir),
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                num_train_epochs=args.epochs,
                max_steps=args.max_steps,
                learning_rate=args.lr,
                logging_steps=5,
                save_strategy="epoch",
                save_total_limit=2,
                fp16=False,
                bf16=use_bf16,
                optim="adamw_8bit",
            ),
        )

        trainer.train()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        print(f"Saved LoRA adapter to {args.output_dir}")
        return
    except Exception as e:
        if args.backend == "unsloth":
            raise
        print(f"Unsloth path failed, using fallback trainer: {e}")

    _train_with_transformers(args, rows)


if __name__ == "__main__":
    main()
