# Code for preprocessing sequences of text, taken from https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import os


class Preprocessing:
    def __init__(self, num_words, seq_len, path_to_data="Amazon_short_text"):
        self.data = path_to_data
        self.num_words = num_words
        self.seq_len = seq_len
        self.vocabulary = None
        self.x_tokenized = None
        self.x_padded = None
        self.x_raw = None

    def load_data(self):
        # Reads the raw csv file and split into
        # sentences (x) and target (y)
        self.x_raw = []
        for i in os.listdir(self.data):
            if i != "clusters.csv":
                df = pd.read_csv(self.data + "/" + i)
                text = " ".join(df["option1"].tolist())
                self.x_raw.append(text)

    def clean_text(self):
        # Removes special symbols and just keep
        # words in lower or upper form

        self.x_raw = [x.lower() for x in self.x_raw]
        self.x_raw = [re.sub(r"[^A-Za-z]+", " ", x) for x in self.x_raw]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x_raw = [word_tokenize(x) for x in self.x_raw]

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        self.vocabulary = dict()
        fdist = nltk.FreqDist()

        for sentence in self.x_raw:
            for word in sentence:
                fdist[word] += 1

        common_words = fdist.most_common(self.num_words)

        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = idx + 1

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation

        self.x_tokenized = list()

        for sentence in self.x_raw:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            self.x_tokenized.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0
        pad_idx = 0
        print(self.seq_len)
        self.x_padded = np.zeros((len(self.x_tokenized), self.seq_len))
        print(self.x_padded.shape)
        t = 0
        for sentence in self.x_tokenized:

            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            self.x_padded[t] = sentence[:10000]
            t = t + 1
            print(t)

        # self.x_padded = np.array(self.x_padded,dtype="object")


def prepare_data(num_words=2000, seq_len=10000, path_to_data="Amazon_short_text"):
    pr = Preprocessing(num_words, seq_len, path_to_data="Amazon_short_text")
    pr.load_data()
    pr.clean_text()
    pr.text_tokenization()
    pr.build_vocabulary()
    pr.word_to_idx()
    pr.padding_sentences()
    return pr.x_padded
