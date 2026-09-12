"""Stage two: QLoRA fine-tuning. Runs on CUDA only - see README for Colab."""

from __future__ import annotations

import os

from .config import CHECKPOINT_DIR, PRICE_PREFIX, Settings, ensure_dirs, hf_login
from .loader import has_cuda, load_tokenizer, memory_footprint, quant_config


def _as_text(batch, eos: str) -> dict[str, list[str]]:
    """Flatten prompt+completion into one field and terminate it properly.

    Without the EOS the model never learns to stop and will happily invent a
    second product after answering.
    """
    return {
        "text": [p + c + eos for p, c in zip(batch["prompt"], batch["completion"])]
    }


def train(settings: Settings, pairs_repo: str | None = None, use_wandb: bool = False) -> str:
    if not has_cuda():
        raise SystemExit(
            "No CUDA device found. QLoRA training needs a GPU - open the project "
            "on Colab (see README) and run this same command there."
        )

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, set_seed
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    ensure_dirs()
    hf_login()
    set_seed(settings.train.seed)

    repo = pairs_repo or settings.pairs_repo
    print(f"[train] dataset: {repo}")
    data = load_dataset(repo)

    tokenizer = load_tokenizer(settings.data.base_model, padding_side="right")
    eos = tokenizer.eos_token

    train_split = data["train"]
    if settings.train.train_cap:
        cap = min(settings.train.train_cap, len(train_split))
        train_split = train_split.select(range(cap))
        print(f"[train] capped to {cap:,} examples")

    train_split = train_split.map(
        _as_text,
        batched=True,
        fn_kwargs={"eos": eos},
        remove_columns=train_split.column_names,
    )

    print(f"[train] loading {settings.data.base_model} in 4-bit")
    model = AutoModelForCausalLM.from_pretrained(
        settings.data.base_model,
        quantization_config=quant_config(),
        device_map="auto",
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"[train] footprint: {memory_footprint(model)}")

    # Loss is computed only on what follows the price prefix. The model is never
    # rewarded for reproducing the product description it was handed.
    collator = DataCollatorForCompletionOnlyLM(PRICE_PREFIX, tokenizer=tokenizer)

    adapter = LoraConfig(
        r=settings.adapter.rank,
        lora_alpha=settings.adapter.alpha,
        lora_dropout=settings.adapter.dropout,
        target_modules=list(settings.adapter.targets),
        bias="none",
        task_type="CAUSAL_LM",
    )

    if use_wandb:
        os.environ["WANDB_PROJECT"] = "pricesense"
        os.environ["WANDB_LOG_MODEL"] = "end"

    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR / settings.run_name),
        run_name=settings.run_name,
        num_train_epochs=settings.train.epochs,
        per_device_train_batch_size=settings.train.batch_size,
        gradient_accumulation_steps=settings.train.grad_accum,
        learning_rate=settings.train.learning_rate,
        lr_scheduler_type=settings.train.scheduler,
        warmup_ratio=settings.train.warmup_ratio,
        optim=settings.train.optimizer,
        max_seq_length=settings.train.max_seq_len,
        dataset_text_field="text",
        weight_decay=0.001,
        max_grad_norm=0.3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        group_by_length=True,
        logging_steps=settings.train.log_every,
        save_steps=settings.train.save_every,
        save_total_limit=3,
        report_to="wandb" if use_wandb else "none",
        seed=settings.train.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_split,
        data_collator=collator,
        peft_config=adapter,
        processing_class=tokenizer,
    )

    trainer.train()

    target = settings.adapter_repo
    trainer.model.push_to_hub(target)
    tokenizer.push_to_hub(target)
    print(f"[train] adapter pushed to {target}")
    return target