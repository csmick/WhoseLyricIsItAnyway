#!/usr/bin/python3

# Clean the song data and split it into training and testing files

import nltk
import pandas as pd
import string

def clean(text, lowercase=True, remove_punctuation=True, remove_stopwords=True):
    # Convert to lower case
    if lowercase:
        text = text.lower()
    # Remove punctuation
    if remove_punctuation:
        # Create translator for removing punctuation
        punc_translator = dict.fromkeys(map(ord, string.punctuation))
        # Remove punctuation
        text = text.translate(punc_translator)
    # Remove stopwords
    if remove_stopwords:
        # Download stopwords from nltk corpus
        nltk.data.path.append('./.nltk_data')
        nltk.download('stopwords', download_dir='./.nltk_data', quiet=True)
        stopwords = nltk.corpus.stopwords.words('english')
        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in stopwords)
    return text

def restrict_artists(df, n):
    artists = set(df['artist'].value_counts()[:n].index)
    return df[df['artist'].isin(artists)]

def split_data(df, ratio):
    increment = len(df)//(len(df)*ratio)
    training_data = df[df.index % increment != 0]
    testing_data = df[df.index % increment == 0]
    return (training_data, testing_data)
    
if __name__ == "__main__":
    df = pd.read_csv('songdata.csv', header=0)
    df = df[['artist', 'text']]
    df['artist'] = df['artist'].apply(clean, remove_stopwords=False)
    df['text'] = df['text'].apply(clean)
    df = restrict_artists(df, 50)
    training_data, testing_data = split_data(df, 0.2)
    training_data.to_csv('lyrics.train', sep=',', index=False, header=False)
    testing_data.to_csv('lyrics.test', sep=',', index=False, header=False)
    
