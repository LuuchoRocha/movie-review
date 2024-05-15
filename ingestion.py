from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
dataset = load_from_disk("datasets/imdb")
train_data = dataset["train"].map(
    lambda value: tokenizer(value["text"], padding="max_length", truncation=True), batched=True
)
eval_data = dataset["test"].map(
    lambda value: tokenizer(value["text"], padding="max_length", truncation=True), batched=True
)
id2label = {0: "Negativo", 1: "Positivo"}
label2id = {"Negativo": 0, "Positivo": 1}
