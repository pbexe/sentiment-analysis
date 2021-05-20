from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, svm


def svm_model(train_x, train_y):
    svm_clf=make_pipeline(StandardScaler(), svm.SVC(cache_size=10000, decision_function_shape='ovo'))
    svm_clf.fit(train_x, train_y)
    return svm_clf

