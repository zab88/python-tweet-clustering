# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import collections

#set here number of clusters you want to get
CLUSTERS_COUNT = 2

#the length of feature vector. Less representative features will be omitted
FEATURES_NUM = 1000

#for debug or memory saving number of loaded tweets can be limited by that parameter
TOTAL_TWEETS = 50000

FILE_IN = 'data/in/tweets.json'
FILE_OUT = 'data/out/clustered.html'

#creating character N-gram with length 4
def getCharNGramm(word, num=4):
    #if length of word is less then 4, let's return entire word
    if len(word)<=num:
        return [word]
    iterations = len(word)-num + 1
    res = []
    for i in range(iterations):
        ng = word[i:(num+i)]
        res.append( ng )
    #so we extracted all character 4-gram, for example for hash #abcde it will return ['abcd', 'bcde']
    return res

#extracting hashes of tweets and tweets itself
def getHashTweets():
    tweets = []
    tweets_hash = []
    tweets_hash_N = []

    #start reading json tweets
    file_tweets = open(FILE_IN, 'r')
    for line in file_tweets:
        tweet = json.loads(line)
        #if more then 3 hash tags, we consider it as spam
        if len(tweet[u'entities'][u'hashtags']) > 3:
            continue
        #we do not consider tweets with no hash tags at all
        if len(tweet[u'entities'][u'hashtags']) > 0:
            hashes = []
            hashesN = []
            for el in tweet[u'entities'][u'hashtags']:
                hashes.append(el[u'text'])
                #making "N-gramm hashes" from full hash tag
                hashesTmp = getCharNGramm(el[u'text'])
                for h in hashesTmp:
                    hashesN.append( h )
            tweets_hash.append(hashes)
            tweets_hash_N.append(hashesN)
            tweets.append(tweet)
        if len( tweets ) > TOTAL_TWEETS:
            break
    file_tweets.close()

    return tweets, tweets_hash, tweets_hash_N

#creating dictionary of all features
def get_dict(vectors):
    words_all = []
    for hashes in vectors:
        for hash in hashes:
            words_all.append(hash)

    x=collections.Counter(words_all)
    words = [elt for elt, count in x.most_common(FEATURES_NUM)]

    return words

#making feature vector for every tweet
def get_vector(hashes, dict):
    res_vector = []
    for k, d in enumerate(dict):
        res_vector.append(0)
        for hash in hashes:
            if hash == d:
                res_vector[k] += 1

    return res_vector

#1: loading tweets
tweets, tweets_hash, tweets_hash_N = getHashTweets()
print('total tweets for analyse = ' + str(len(tweets)))

#2: extracting dictionary of features
dict = get_dict(tweets_hash_N)
print('dictionary length ' + str( len(dict) ))

#3: creating feature vector for every tweet
tweets_vectors = []
for hashes in tweets_hash_N:
    tweets_vectors.append( get_vector(hashes, dict) )

print('vectors created, clustering is starting now!')

#4: clustering with K-means
tweets_vectors = np.float32(tweets_vectors)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center = cv2.kmeans(tweets_vectors, CLUSTERS_COUNT, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
res_bin = label.flatten()

#5: creating out file, html formatted
result = [None]*CLUSTERS_COUNT
for k, el in enumerate( res_bin ):
    if result[el] is None:
        result[el] = []
    result[el].append( tweets[k][u'text'].encode('utf-8') )

with open(FILE_OUT, 'w') as outfile:
    for cluster in result:
        outfile.write("<h3>New Cluster ("+str(len(cluster))+")</h3>" + "\n")
        for tt in cluster:
            outfile.write( "<p>" + str(tt) + "</p>\n" )

print('All done. Result in file '+FILE_OUT)