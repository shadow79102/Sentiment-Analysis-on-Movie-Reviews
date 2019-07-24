# -*- coding: utf-8 -*-

#import required packages

from nltk.corpus import movie_reviews
from random import shuffle
from nltk import FreqDist
from nltk.corpus import stopwords
import string
from nltk import NaiveBayesClassifier as NBC
from nltk import classify
from nltk.tokenize import word_tokenize
#import pickle

reviews = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        reviews.append((movie_reviews.words(fileid), category))

stopWords = stopwords.words('english')

#cleaning the reviews for improved accuracy
words = [word.lower() for word in movie_reviews.words()]

words_modified = []
for word in words:
    if word not in stopWords and word not in string.punctuation:
        words_modified.append(word)
        
words_frequency = FreqDist(words_modified)
most_common_words = words_frequency.most_common(2000)

word_features = [item[0] for item in most_common_words]

#function to extract features from reviews
def review_features(review):
    document_words = set(review)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#shuffle the reviews for improved accuracy
shuffle(reviews)

#create test and train feature sets (80% test set and 20% train set)
feature_set = [(review_features(doc), category) for (doc, category) in reviews]

test_feature_set = feature_set[:400]
train_feature_set = feature_set[400:]

classifier = NBC.train(train_feature_set)

accuracy = classify.accuracy(classifier, test_feature_set)
print (accuracy)

#uncomment the below lines to save your own trained model
#f = open('topN_classifier.pickle', 'wb')
#pickle.dump(classifier, f)
#f.close()

#testing on a custom movie review
while(1):
    custom_review = input("Enter a custom movie review (Press ENTER key to exit):\n")
    if(len(custom_review) < 1):
        break
    custom_review_tokens = word_tokenize(custom_review)
    custom_review_set = review_features(custom_review_tokens)
    print (classifier.classify(custom_review_set))
    prob_result = classifier.prob_classify(custom_review_set)
    print ("confidence: " + (str)(prob_result.prob(prob_result.max())))
