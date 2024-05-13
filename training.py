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
)
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
    # per_device_train_batch_size=24,
    # per_device_eval_batch_size=24,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    weight_decay=0.01,
    save_on_each_node=True,
    save_strategy=save_strategy,
    evaluation_strategy=evaluation_strategy,
    load_best_model_at_end=evaluation_strategy == save_strategy,
    use_cpu=use_cpu,
    push_to_hub=True,
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
