from typing import List, Tuple

from console import console
from feature_extraction import miles_cw_extractor
from numpy import ndarray
from preprocessing import get_data
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm_model(train_x:List[ndarray], train_y:List[int]) -> svm.SVC:
    """Generates the SVM model from the given training data.

    Args:
        train_x (List[ndarray]): The x training data.
        train_y (List[int]): The y training data.

    Returns:
        svm.SVC: The trained model.
    """
    console.log("Specifying Hyperparameters")
    # svm_clf=make_pipeline(StandardScaler(), svm.SVC(decision_function_shape='ovo', verbose=True))
    svm_clf=svm.SVC(decision_function_shape='ovo', verbose=True)
    console.log("Fitting Data")
    svm_clf.fit(train_x, train_y)
    console.log("Data Fitted")

    return svm_clf

def svm_implementation() -> Tuple[SVC, TfidfTransformer, CountVectorizer]:
    """Implementation of an SVM classifier

    Returns:
        Tuple[SVC, TfidfTransformer, CountVectorizer]: The SVM model as well as
        the objects required to transform the data for the model.
    """
    
    console.log("Getting Data")
    train_x, train_y = get_data()
    with console.status("Pre-processing Data...", spinner="aesthetic"):
        train_x_processed, tfid, vectorizer, get_best = miles_cw_extractor(train_x, train_y)
    with console.status("Generating Model...", spinner="aesthetic"):
        model = svm_model(train_x_processed, train_y)
    return model, tfid, vectorizer, get_best
