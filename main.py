from preprocessing import preprocess, analysis
from models import example_model
from scipy.special import softmax
import numpy as np
from typing import List, Tuple
from rich import print
from rich.traceback import install
install()




def get_data(type_:str = "train") -> Tuple[List[str], List[int]]:
    x = []
    y = []
    with open(f"data/{type_}_text.txt", "r") as fp:
        x = fp.readlines()
    with open(f"data/{type_}_labels.txt", "r") as fp:
        y = [int(i) for i in fp.readlines()]
    return x, y


def svm_implementation():
    train_x, train_y = get_data()
    print(train_x[:5])
    print(train_y[:5])
    1/0


if __name__ == "__main__":
    svm_implementation()
    # model, tokenizer, labels = example_model()
    # analysis()
    # while 1:
    #     text = input("sentence>>> ")
    #     text = preprocess(text)
    #     encoded_input = tokenizer(text, return_tensors='pt')
    #     output = model(**encoded_input)
    #     scores = output[0][0].detach().numpy()
    #     scores = softmax(scores)
    #     ranking = np.argsort(scores)
    #     ranking = ranking[::-1]
    #     for i in range(scores.shape[0]):
    #         l = labels[ranking[i]]
    #         s = scores[ranking[i]]
    #         print(f"{i+1}) {l} {np.round(float(s), 4)}")
