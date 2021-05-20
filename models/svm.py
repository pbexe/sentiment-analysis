from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, svm
from console import console



def svm_model(train_x, train_y):
    # svm_clf=make_pipeline(StandardScaler(), svm.SVC(decision_function_shape='ovo'))
    console.log("Specifying Hyperparameters")
    svm_clf=svm.SVC(decision_function_shape='ovo')
    console.log("Fitting Data")
    svm_clf.fit(train_x, train_y)
    console.log("Data Fitted")

    return svm_clf

