import pickle
from os.path import exists
from random import shuffle
from typing import List, Tuple

import numpy as np
from rich import print
from rich.panel import Panel
from rich.progress import track
from rich.traceback import install
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from console import console
from feature_extraction import miles_cw_extractor
<<<<<<< HEAD
from models import example_model, svm_model
from preprocessing import preprocess
from sklearn.metrics import classification_report

# NUMBER_OF_TRAINING_SAMPLES = 45615
NUMBER_OF_TRAINING_SAMPLES = 5615
=======
from models import example_model, svm_implementation, svm_model
from preprocessing import analysis, get_data, preprocess
>>>>>>> bf0746e05b08bad46692c08787ba7c8d406c8470

install()



if __name__ == "__main__":
    console.rule("Building Models")
    console.log("Building SVM")

    # Prepare SVM model
    if exists("checkpoint.p"):
        console.log("Checkpoint found in FS")
        with console.status("Loading checkpoint from file...", spinner="aesthetic"):
            svm_model, tfid, vectorizer, get_best= pickle.load(open("checkpoint.p", "rb"))
    else:
        svm_model, tfid, vectorizer, get_best = svm_implementation()
        with console.status("Saving checkpoint to file...", spinner="aesthetic"):
            pickle.dump((svm_model, tfid, vectorizer, get_best), open("checkpoint.p", "wb" ))

    with console.status("Loading example model...", spinner="aesthetic"): 
        model, tokenizer, labels = example_model()

<<<<<<< HEAD
    v_x, v_y = get_data("val")
=======
    console.rule("Performing Analysis")
    analysis()

    v_x, v_y = get_data("val", False)
>>>>>>> bf0746e05b08bad46692c08787ba7c8d406c8470
    Y_text_predictions = svm_model.predict(get_best.transform(
        tfid.transform(
            vectorizer.transform(v_x)
            ).toarray()
        )
    )
    print(classification_report(v_y, Y_text_predictions))
    
    Y_text_predictions = []
    for tweet in track(v_x):
        encoded_input = tokenizer(tweet, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        prediction = ranking[0]
        Y_text_predictions.append(prediction)
    print(classification_report(v_y, Y_text_predictions))

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
