from torch import no_grad
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from ingestion import id2label, label2id

tokenizer = DistilBertTokenizerFast.from_pretrained("./movie-bert")
model = DistilBertForSequenceClassification.from_pretrained(
    "./movie-bert",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)


def classify(text):
    with no_grad():
        logits = model(**tokenizer(text)).logits

    return model.config.id2label[logits.argmax().item()]
