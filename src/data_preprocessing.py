import random
import re
import numpy as np

random.seed(1)


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
"doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
"how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
"I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not",
"it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
"might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
"needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
"sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
"should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",
"that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
"they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
"we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
"what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
"where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
"will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll":
"you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


# data_load() function to load test data from provided url
# ARGUMENTS:
# -- url - valid url link pointing to dataset
# RETURNS:
# -- data_unparsed_input - list of sentences(questions) from dataset
# -- data_labels - corresponding labels for sentences in data_unparsed_input
def data_load(file, lowercase):
    data_unparsed_input = list()
    data_labels = list()
    # fetching text data from url
    data = open(file, 'r', encoding="utf8").readlines()
    max_length = 0
    # creating sentences and labels lists
    for sentence in data:
        if lowercase:
            sentence = sentence.lower()
        label, long_data = sentence.split(' ', 1)
        long_data = long_data.replace(' \'', '\'')
        long_data = replace_contractions(long_data)
        long_data = re.sub('[^a-z0-9\s]+', '', long_data)
        long_data = re.sub('[\n]+', '', long_data)
        long_data = re.sub('[0-9]{5,}', '#####', long_data)
        long_data = re.sub('[0-9]{4}', '####', long_data)
        long_data = re.sub('[0-9]{3}', '###', long_data)
        long_data = re.sub('[0-9]{2}', '##', long_data)
        long_data, length = tokenize(long_data)
        if length > max_length:
            max_length = length
        data_labels.append(label)
        data_unparsed_input.append(long_data)

    return data_unparsed_input, data_labels, max_length


def create_dictionary(sentences):
    word_dict = list()
    word_dict.append('')
    word_dict.append('UNK')
    for sentence in sentences:
        for word in sentence:
            if word not in word_dict:
                word_dict.append(word)

    return word_dict


def retrieve_unique_classes(labels):
    return list(set(labels))


def get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


contractions, contractions_re = get_contractions(contraction_dict)


def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


def tokenize(sentence):
    new_sentence = []
    words = sentence.split(' ')
    for word in words:
        new_sentence.append(word)

    new_sentence = list(dict.fromkeys(new_sentence))
    return new_sentence, len(new_sentence)


def word_to_idx_mapping(vocabulary):
    word2idx = dict()
    for i in range(len(vocabulary)):
        word2idx[vocabulary[i]] = i

    return word2idx


def labels_to_idx_mapping(labels):
    return {labels[i]: i for i in range(len(labels))}


def encode_data(data, word2idx, N):
    encoded_data = []
    for item in data:
        new_item = np.zeros(N, dtype=int)
        partial_encoding = np.array([word2idx.get(word, word2idx["UNK"]) for word in item])
        new_item[:len(partial_encoding)] = partial_encoding
        encoded_data.append(np.array(new_item, dtype=object))

    return encoded_data


def encode_labels(labels, label2idx):
    return [label2idx[label] for label in labels]


def retrieve_embeddings(file, vocabulary, embed_dim):
    embeddings = np.zeros((len(vocabulary), embed_dim), dtype="float32")
    embeddings[0] = np.zeros(embed_dim, dtype='float32')
    embeddings[1] = np.random.uniform(-0.25, 0.25, embed_dim)
    i = 2
    with open(file, 'r', encoding="utf8") as file:
        for line in file:
            word, vector = line.split(' ', 1)
            if word in vocabulary:
                values = vector.split(' ')
                vector = [float(i) for i in values]
                embeddings[i] = np.array(vector)
                i += 1

    return embeddings


