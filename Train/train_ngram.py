# -*- coding: utf-8 -*-

from nltk.corpus import movie_reviews
from random import shuffle
from nltk.corpus import stopwords
import string
from nltk import NaiveBayesClassifier as NBC
from nltk import classify
from nltk.tokenize import word_tokenize
import pickle
import time

stopWords = stopwords.words('english')

t1 = time.time()

def unigram_words(words):
    words_modified = []
    for word in words:
        word = word.lower()
        if word not in stopWords and word not in string.punctuation:
            words_modified.append(word)
    return words_modified

def generate_ngrams(words, ngram_words, forward = True):
    words_modified = []
    for word in words:
        word = word.lower()
        if word not in string.punctuation:
            words_modified.append(word)
    new_text = []
    index = 0
    if not forward:
        words_modified = list(reversed(words_modified))
    while index < len(words_modified):
        [new_word, new_index] = concatenate_words(index, words_modified, ngram_words, forward)
        new_text.append(new_word)
        index = new_index+1 if index != new_index else index+1
    if not forward:
        return list(reversed(new_text))
    return new_text
 
def concatenate_words(index, text, ngram_words, forward):
    words = text[index]
    if index == len(text)-1:
        return words, index
    if words.split(' ')[0] in ngram_words:
        [new_word, new_index] = concatenate_words(index+1, text, ngram_words, forward)
        if forward:
            words = words + ' ' + new_word
        else:
            words = new_word + ' ' + words
        index = new_index
    return words, index

linking_words = ['and', 'any', 'anyone', 'anything', 'are', 'be', 'best', 'can', 'cannot', 'cant',
                 "can't", 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'done',
                 "don't", 'either', 'else', 'even', 'every', 'for', 'from', 'have', "haven't",
                 "he's", 'is', "isn't", 'it', 'its', "i've", 'just', 'like', 'lots', 'many', 'maybe',
                 'me', 'might', 'more', 'must', 'my', 'never', 'no', 'none', 'not', 'nothing', 'now',
                 'of', 'on', 'once', 'one', 'only', 'or', 'overly', 'perfectly', 'perhaps',
                 'probably', 'seemed', 'seems', "she's", 'should', 'simply', 'so', 'some',
                 'somehow', 'something', 'soon', 'start', 'takes', 'tell', 'thank', "that's",
                 'the', 'their', 'them', 'then', 'there', "there's", 'they', "they're", 'this',
                 'those', 'to', 'too', 'totally', 'tried', 'truly', 'try', 'turns', 'until',
                 'upon', 'use', 'very', 'wait', 'was', 'well', 'went', 'were', 'whether', 'which',
                 'whole', 'why', 'will', 'wish', "won't", 'would',"wouldn't", 'you', "you'll",
                 'your', "you're", 'yourself']

#def bag_of_all_words(words):
    #all_feature_words = unigram_words(words)
    #all_feature_words.extend(generate_ngrams(words, linking_words))
    #all_feature_words.extend(generate_ngrams(words, linking_words, forward = False))
 
    #all_features = dict([word, True] for word in all_feature_words)
 
    #return all_features

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)
 
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

# positive reviews feature set
pos_features = []
for words in pos_reviews:
    pos_features.append((dict([word, True] for word in unigram_words(words)), 'pos'))
    pos_features.append((dict([word, True] for word in generate_ngrams(words, linking_words)), 'pos'))
    pos_features.append((dict([word, True] for word in generate_ngrams(words, linking_words, forward = False)), 'pos'))
 
# negative reviews feature set
neg_features = []
for words in neg_reviews:
    neg_features.append((dict([word, True] for word in unigram_words(words)), 'neg'))
    neg_features.append((dict([word, True] for word in generate_ngrams(words, linking_words)), 'neg'))
    neg_features.append((dict([word, True] for word in generate_ngrams(words, linking_words, forward = False)), 'neg'))

max_accuracy = 0
avg_accuracy = 0

t2 = time.time()

total_time = 0

for i in range(25):
    t3 = time.time()
    shuffle(pos_features)
    shuffle(neg_features)
 
    test_feature_set = pos_features[:300] + neg_features[:300]
    train_feature_set = pos_features[300:] + neg_features[300:]

    classifier = NBC.train(train_feature_set)
 
    accuracy = classify.accuracy(classifier, test_feature_set)
    print (accuracy)
    avg_accuracy += accuracy
    if(accuracy > max_accuracy):
        max_accuracy = accuracy
        f = open('ngram_classifier.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()
    total_time += (time.time() - t3)
    
print ("avg_accuracy: " + (str)(avg_accuracy / 25.0))
print ("avg_execution_time: " + (str)((t2 - t1) + total_time / 25.0))

while(1):
    custom_review = input("Enter a custom movie review (Press ENTER key to exit):\n")
    if(len(custom_review) < 1):
        break
    custom_review_tokens = word_tokenize(custom_review)
    custom_feature_set = dict([word, True] for word in unigram_words(custom_review_tokens))
    custom_feature_set.update(dict([word, True] for word in generate_ngrams(custom_review_tokens, linking_words)))
    custom_feature_set.update(dict([word, True] for word in generate_ngrams(custom_review_tokens, linking_words, forward = False)))
    print (classifier.classify(custom_feature_set))
    prob_result = classifier.prob_classify(custom_feature_set)
    print ("confidence: " + (str)(prob_result.prob(prob_result.max())))