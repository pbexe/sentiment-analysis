"""This should probably be updated but I need some feature extraction to start
on my part of the project. This is pretty much what I used in the last CW.
"""

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from rich.progress import track
from console import console





def miles_cw_extractor(train_x, train_y):
    """Extracts features from a list of strings

    Args:
        stories (List[str]): Strings to extract features from

    Returns:
        List[List[int]]: List of vectors which can be used in a model
    """
    console.log("Extracting features")
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    tfid = TfidfTransformer()
    vectorizer.fit(train_x)
    tfid.fit(vectorizer.transform(train_x))

    # Iterate through all of the training data and process it
    tweets = []
    for tweet in track(train_x):
        tweets.append(
            tfid.transform(vectorizer.transform([tweet])).toarray()[0]
        )
    return tweets, tfid, vectorizer
