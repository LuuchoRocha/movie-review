import torch
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
tokenizer = DistilBertTokenizerFast.from_pretrained("./movie-bert")
model = DistilBertForSequenceClassification.from_pretrained(
    "./movie-bert",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

def classify(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

