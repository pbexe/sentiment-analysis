"""This should probably be updated but I need some feature extraction to start
on my part of the project. This is pretty much what I used in the last CW.
"""

import nltk; nltk.download("stopwords", quiet=True)
from typing import List, Tuple

from console import console
from nltk.corpus import stopwords
from numpy import ndarray
from rich.progress import track
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2


def miles_cw_extractor(train_x:List[str], train_y:List[int]) -> Tuple[List[ndarray], TfidfTransformer, CountVectorizer]:
    """Extracts features from a list of strings

    Args:
        train_x (List[str]): Strings to extract features from
        train_y (List[int]): Corresponding feature labels

    Returns:
        Tuple[List[ndarray], TfidfTransformer, CountVectorizer]: The extracted
        features as well as the objects required to extract future features.
    """
    console.log("Extracting features")
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    tfid = TfidfTransformer()
    vectorizer.fit(train_x)
    tfid.fit(vectorizer.transform(train_x))

    # Iterate through all of the training data and process it
    tweets = []
    for tweet in train_x:
        tweets.append(
            tfid.transform(vectorizer.transform([tweet])).toarray()[0]
        )
    get_best=SelectKBest(chi2, k=500).fit(tweets, train_y)
    tweets = get_best.transform(tweets)

    return tweets, tfid, vectorizer, get_best
