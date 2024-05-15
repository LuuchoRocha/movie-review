from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from config import (
    device,
    evaluation_strategy,
    num_train_epochs,
    save_steps,
    save_strategy,
    use_cpu,
    max_steps,
    weight_decay,
    learning_rate,
    batch_size,
    log_level,
    eval_steps,
)
from ingestion import data, data_collator, id2label, label2id, tokenizer
from metrics import compute_metrics

model = DistilBertForSequenceClassification.from_pretrained(
    "./hungry-bert",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
).to(device)


training_args = TrainingArguments(
    output_dir="./hungry-bert",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    max_steps=max_steps,
    eval_steps=eval_steps,
    weight_decay=weight_decay,
    save_strategy=save_strategy,
    evaluation_strategy=evaluation_strategy,
    load_best_model_at_end=evaluation_strategy == save_strategy,
    use_cpu=use_cpu,
    log_level=log_level,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.eval_dataset = data["test"] if evaluation_strategy != "no" else None

train = trainer.train
save = trainer.save_model
