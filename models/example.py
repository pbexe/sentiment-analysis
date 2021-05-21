import csv
import urllib.request
from typing import List, Tuple

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def example_model() -> Tuple[any, any, List[str]]:
    """Implementation of the example model provided along with the training
    data.

    Returns:
        Tuple[RoberetaForSequenceClassification, RobertaTokenizerFast,
        List[str]]: The trained model as well as the objects required to
        transform the input data.
    """
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    return AutoModelForSequenceClassification.from_pretrained(MODEL), tokenizer, labels

