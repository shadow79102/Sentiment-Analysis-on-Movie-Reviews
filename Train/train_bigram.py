# -*- coding: utf-8 -*-

from nltk.corpus import movie_reviews
from random import shuffle
from nltk.corpus import stopwords
import string
from nltk import NaiveBayesClassifier as NBC
from nltk import classify, ngrams
from nltk.tokenize import word_tokenize
#import pickle

stopWords = stopwords.words('english')

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)
 
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

def modify_words(words, stopwords_english):
    words_modified = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_modified.append(word)    
    return words_modified

def bag_of_words(words):    
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)    
    return words_dictionary

linking_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most',
                   'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too',
                   'very', 'just', 'but']

bigram_stopwords = set(stopWords) - set(linking_words)

def bag_of_all_words(words, n=2):
    words_modified = modify_words(words, stopWords)
    bigram_words_modified = modify_words(words, bigram_stopwords)
 
    unigram_features = bag_of_words(words_modified)
    bigram_features = bag_of_ngrams(bigram_words_modified)
 
    all_features = unigram_features.copy()
    all_features.update(bigram_features)
 
    return all_features

# positive reviews feature set
pos_features = []
for words in pos_reviews:
    pos_features.append((bag_of_all_words(words), 'pos'))
 
# negative reviews feature set
neg_features = []
for words in neg_reviews:
    neg_features.append((bag_of_all_words(words), 'neg'))

shuffle(pos_features)
shuffle(neg_features)
 
test_feature_set = pos_features[:200] + neg_features[:200]
train_feature_set = pos_features[200:] + neg_features[200:]

classifier = NBC.train(train_feature_set)
 
accuracy = classify.accuracy(classifier, test_feature_set)
print (accuracy)
#f = open('bigram_classifier.pickle', 'wb')
#pickle.dump(classifier, f)
#f.close()

while(1):
    custom_review = input("Enter a custom movie review (Press ENTER key to exit):\n")
    if(len(custom_review) < 1):
        break
    custom_review_tokens = word_tokenize(custom_review)
    custom_feature_set = bag_of_all_words(custom_review_tokens)
    print (classifier.classify(custom_feature_set))
    prob_result = classifier.prob_classify(custom_feature_set)
    print ("confidence: " + (str)(prob_result.prob(prob_result.max())))