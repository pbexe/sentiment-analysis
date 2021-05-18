import pandas as pd
import matplotlib.pyplot as plt
import string

def analysis():
  a = open('./data/train_text.txt', 'r')
  train_tweets = a.read().split('\n')
  a.close()

  b = open('./data/train_labels.txt', 'r')
  train_labels = b.read().split('\n')
  b.close()

  # Create a new list of the tweets without any punctuation
  no_punc = []
  for tweet in train_tweets:
    no_punc.append(tweet.translate(str.maketrans('', '', string.punctuation)))

  # Define a dataframe of the tweets without punctuation and their labels
  df_np = pd.DataFrame()
  df_np['tweet']  = no_punc
  df_np['label']  = train_labels

  # Append the length of each tweet to our dataframe
  tweet_length = []
  for i in df_np['tweet']:
    tweet_length.append(len(i))
  df_np['tweet_length'] = tweet_length

  print(df_np.head())

  lengths = df_np['tweet_length'].plot(
    kind='hist',
    bins=50,
    title='Tweet Length Distribution')

  plt.show()
  