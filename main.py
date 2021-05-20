from typing import List, Tuple
import pickle
from console import console

import numpy as np
from rich import print
from rich.panel import Panel
from rich.traceback import install
install()
from scipy.special import softmax

from models import example_model
from preprocessing import analysis, preprocess

from feature_extraction import miles_cw_extractor
from models import svm_model


def get_data(type_:str = "train") -> Tuple[List[str], List[int]]:
    x = []
    y = []
    with open(f"data/{type_}_text.txt", "r") as fp:
        x = fp.readlines()
    with open(f"data/{type_}_labels.txt", "r") as fp:
        y = [int(i) for i in fp.readlines()]
    return x, y


def svm_implementation():
    console.log("Getting Data")
    train_x, train_y = get_data()
    console.log(len(train_x))
    print(train_x[:5])
    print(train_y[:5])
    train_x = train_x[:1000]
    train_y = train_y[:1000]
    console.log("Pre-processing Data")
    train_x_processed, tfid, vectorizer = miles_cw_extractor(train_x, train_y)
    console.log("Generating Model")
    model = svm_model(train_x_processed, train_y)
    return model, tfid, vectorizer


if __name__ == "__main__":
    console.rule("Building Model")
    svm_model, tfid, vectorizer = svm_implementation()
    pickle.dump((svm_model, tfid, vectorizer), open( "checkpoint.p", "wb" ) )
    console.rule("Making predictions")
    # console.log(model.predict(tfid.transform(vectorizer.transform(["That was good"])).toarray()))
    
    model, tokenizer, labels = example_model()
    analysis()
    while 1:
        text = input("sentence>>> ")
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")
        # ------------
        # console.log(svm_model.predict(tfid.transform(vectorizer.transform([text])).toarray()))
        console.log(("negative", "neutral", "positive")[svm_model.predict(tfid.transform(vectorizer.transform([text])).toarray())[0]])