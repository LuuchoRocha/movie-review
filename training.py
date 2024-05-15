from transformers import (
    AutoModelForSequenceClassification,
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
from ingestion import train_data, eval_data, id2label, label2id, tokenizer
from metrics import compute_metrics

weight_decay
learning_rate

model = AutoModelForSequenceClassification.from_pretrained(
    "./movie-bert",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
).to(device)


training_args = TrainingArguments(
    output_dir="./movie-bert-2",
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
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

train = trainer.train
save = trainer.save_model
