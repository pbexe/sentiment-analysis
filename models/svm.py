from typing import List

from console import console
from numpy import ndarray
from sklearn import preprocessing, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def svm_model(train_x:List[ndarray], train_y:List[int]) -> svm.SVC:
    """Generates the SVM model from the given training data.

    Args:
        train_x (List[ndarray]): The x training data.
        train_y (List[int]): The y training data.

    Returns:
        svm.SVC: The trained model.
    """
    console.log("Specifying Hyperparameters")
    svm_clf=make_pipeline(StandardScaler(), svm.SVC(decision_function_shape='ovo', verbose=True))
    # svm_clf=svm.SVC(decision_function_shape='ovo')
    console.log("Fitting Data")
    svm_clf.fit(train_x, train_y)
    console.log("Data Fitted")

    return svm_clf

