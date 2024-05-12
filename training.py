from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from hardware import device, use_cpu
from ingestion import data, data_collator, id2label, label2id, tokenizer
from metrics import compute_metrics

model = DistilBertForSequenceClassification.from_pretrained(
    "./movie-bert",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
).to(device)


training_args = TrainingArguments(
    output_dir="./movie-bert",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    save_on_each_node=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    use_cpu=use_cpu,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train = trainer.train
save = trainer.save_model
