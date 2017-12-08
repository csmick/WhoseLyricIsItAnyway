#!/usr/bin/python3

from keras.layers import Dense, Embedding, Input, RNN, LSTMCell
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
import numpy as np

MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100

if __name__ == "__main__":
    tokenizer = Tokenizer()
    artists = []
    artist_ids = {}
    lyrics = []
    artist_id = 0
    with open('lyrics.train', 'r') as f:
        for line in f:
            [artist, song_lyrics] = line.strip().split(',')
            lyrics.append(song_lyrics)
            if artist not in artist_ids:
                artist_ids[artist] = artist_id
                artist_id += 1
            artists.append(artist_ids[artist])
    split_index = len(lyrics)
    with open('lyrics.test', 'r') as f:
        for line in f:
            [artist, song_lyrics] = line.strip().split(',')
            lyrics.append(song_lyrics)
            if artist not in artist_ids:
                artist_ids[artist] = artist_id
                artist_id += 1
            artists.append(artist_ids[artist])
    tokenizer.fit_on_texts(lyrics)
    lyric_sequences = tokenizer.texts_to_sequences(lyrics)
    lyric_sequences = pad_sequences(lyric_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    artists = to_categorical(np.asarray(artists))
    word_index = tokenizer.word_index

    # Split into training and testing data
    x_train = lyric_sequences[:split_index]
    y_train = artists[:split_index]
    x_test = lyric_sequences[split_index:]
    y_test = artists[split_index:]

    # Store word embeddings
    embeddings_index = {}
    with open('glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Create embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Create embedding layer
    embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

    # Train neural network
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    x = RNN(LSTMCell(128))(embedded_sequences)
    preds = Dense(len(artist_ids), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    # Test model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)
