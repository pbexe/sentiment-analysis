from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import csv
import urllib.request

def example_model():
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    return AutoModelForSequenceClassification.from_pretrained(MODEL), tokenizer, labels

