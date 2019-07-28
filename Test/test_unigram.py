# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:49:08 2019

@author: VISHNU
"""

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import pickle
import sys

#give the complete path to the pretrained model
f = open('unigram_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

stopWords = stopwords.words('english')

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)
 
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

def bag_of_words(words):
    words_modified = []
 
    for word in words:
        word = word.lower()
        if word not in stopWords and word not in string.punctuation:
            words_modified.append(word)
    
    words_dictionary = dict([word, True] for word in words_modified)
    
    return words_dictionary

while(1):
    custom_review = input("Enter a custom movie review (Press ENTER key to exit):\n")
    if(len(custom_review) < 1):
        sys.exit()
    custom_review_tokens = word_tokenize(custom_review)
    custom_review_set = bag_of_words(custom_review_tokens)
    print (classifier.classify(custom_review_set))
    prob_result = classifier.prob_classify(custom_review_set)
    print ("confidence: " + (str)(prob_result.prob(prob_result.max())))
