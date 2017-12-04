#!/usr/bin/py

# Basic language model for song data

import collections
import math

class BaselineModel(object):

    def __init__(self):
        self.artists = set()
        self.word_count_by_artist = collections.defaultdict(collections.Counter)
        self.total_words_by_artist = collections.Counter()
        self.word_counts = collections.Counter()
        self.total_word_count = 0

    def train(self, filename):
        with open(filename) as f:
            for line in f:
                if len(line.strip().split(',')) != 2:
                    print(line.strip().split(','))
                [artist, lyrics] = line.strip().split(',')
                self.artists.add(artist)
                for word in lyrics.split():
                    self.word_counts[word] += 1
                    self.word_count_by_artist[word][artist] += 1
                    self.total_words_by_artist[artist] += 1
                    self.total_word_count += 1

    def prob(self, w, a):
        return math.log((((self.word_count_by_artist[w][a]/self.total_words_by_artist[a])*(self.total_words_by_artist[a]/self.total_word_count))+1)/((self.word_counts[w]/self.total_word_count)+self.total_word_count))

if __name__ == "__main__":
    lm = BaselineModel()
    lm.train('lyrics.train')
    line_count = 0
    correct_predictions = 0
    for line in open('lyrics.test'):
        [actual_artist, lyrics] = line.strip().split(',')
        probabilities = collections.defaultdict(int)
        for word in lyrics.split():
            for artist in lm.artists:
                probabilities[artist] += lm.prob(word, artist)
        [predicted_artist, p] = max(probabilities.items(), key=lambda a: a[1])
        if actual_artist == predicted_artist:
            correct_predictions += 1
        line_count += 1

    print('Accuracy: {:.2f}%'.format(100*correct_predictions/line_count))

