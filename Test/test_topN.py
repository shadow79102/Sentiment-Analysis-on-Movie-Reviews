# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 00:14:20 2019

@author: VISHNU
"""

from nltk.corpus import movie_reviews
from nltk import FreqDist, classify
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import pickle
import sys
from random import shuffle

#give the complete path to the pretrained model
f = open('topN_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

reviews = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        reviews.append((movie_reviews.words(fileid), category))

stopWords = stopwords.words('english')

words = [word.lower() for word in movie_reviews.words()]
    
words_modified = []
for word in words:
    if word not in stopWords and word not in string.punctuation:
        words_modified.append(word)
        
words_frequency = FreqDist(words_modified)
most_common_words = words_frequency.most_common(2000)

word_features = [item[0] for item in most_common_words]

def review_features(review):
    document_words = set(review)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

shuffle(reviews)

features = [(review_features(doc), category) for (doc, category) in reviews]

test_feature_set = features[:400]
print (classify.accuracy(classifier, test_feature_set))

while(1):
    custom_review = input("Enter a custom movie review (Press ENTER key to exit):\n")
    if(len(custom_review) < 1):
        sys.exit()
    custom_review_tokens = word_tokenize(custom_review)
    custom_review_set = review_features(custom_review_tokens)
    print (classifier.classify(custom_review_set))
    prob_result = classifier.prob_classify(custom_review_set)
    print ("confidence: " + (str)(prob_result.prob(prob_result.max())))
