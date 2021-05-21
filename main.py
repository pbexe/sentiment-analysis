import pickle
from typing import List, Tuple

import numpy as np
from rich import print
from rich.panel import Panel
from rich.traceback import install; install()
from os.path import exists

from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

from console import console
from feature_extraction import miles_cw_extractor
from models import example_model, svm_model
from preprocessing import analysis, preprocess

NUMBER_OF_TRAINING_SAMPLES = 20000

def get_data(type_:str = "train") -> Tuple[List[str], List[int]]:
    """Load data from the provided filesystem

    Args:
        type_ (str, optional): Type of data to loads. Defaults to "train".

    Returns:
        Tuple[List[str], List[int]]: Tuple of x and y data.
    """

    x = []
    y = []
    with open(f"data/{type_}_text.txt", "r") as fp:
        x = fp.readlines()
    with open(f"data/{type_}_labels.txt", "r") as fp:
        y = [int(i) for i in fp.readlines()]
    return x, y


def svm_implementation() -> Tuple[SVC, TfidfTransformer, CountVectorizer]:
    """Implementation of an SVM classifier

    Returns:
        Tuple[SVC, TfidfTransformer, CountVectorizer]: The SVM model as well as
        the objects required to transform the data for the model.
    """
    
    console.log("Getting Data")
    train_x, train_y = get_data()
    console.log(len(train_x))
    # console.log(train_x[:5])
    # console.log(train_y[:5])
    train_x = train_x[:NUMBER_OF_TRAINING_SAMPLES]
    train_y = train_y[:NUMBER_OF_TRAINING_SAMPLES]
    console.log("Pre-processing Data")
    train_x_processed, tfid, vectorizer, get_best = miles_cw_extractor(train_x, train_y)
    console.log("Generating Model")
    model = svm_model(train_x_processed, train_y)
    return model, tfid, vectorizer, get_best


if __name__ == "__main__":
    console.rule("Building Models")
    console.log("Building SVM")
    if exists("checkpoint.p"):
        console.log("Checkpoint found in FS")
        svm_model, tfid, vectorizer, get_best= pickle.load(open("checkpoint.p", "rb"))
    else:
        with console.status("Generating new SVM...", spinner="moon"):
            svm_model, tfid, vectorizer, get_best = svm_implementation()
            pickle.dump((svm_model, tfid, vectorizer, get_best), open( "checkpoint.p", "wb" ) )
    console.log("Loading example model")    
    model, tokenizer, labels = example_model()
    console.rule("Performing Analysis")
    analysis()
    console.rule("Making predictions")
    while 1:
        text = input("sentence>>> ")
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        console.log("[bold]Example model:[/]")
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            console.log(f"{i+1}) {l} {np.round(float(s), 4)}")
        console.log("[bold]SVM:[/]")
        console.log(
            ("negative", "neutral", "positive")[
                svm_model.predict(
                    get_best.transform(
                    tfid.transform(
                        vectorizer.transform([text])
                    ).toarray()
                )
                )[0]
            ]
        )
