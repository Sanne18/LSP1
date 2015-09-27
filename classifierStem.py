# -*- coding: utf-8 -*-

import os, operator
import re
from nltk.corpus import stopwords
from langdetect import detect
from math import log
from nltk.stem.porter import *
import time
#from evaluate import read_predictions(file)

trainDir = "./train/"
testDir = "./test/"

class Counter(dict):
	def __missing__(self, key):
		return 0

mData = {}
fData = {}
mTokens = Counter()
fTokens = Counter()
n = 0
p = re.compile("[~.,'\":;!@#$%^&*()_\-+=?/|\u201C\u201D\u2018\u2019]")

#the two stopwords list taken from nltk corpus
#stopwordsEn = stopwords.words('english')
#stopwordsNL = stopwords.words('dutch')
twitterSyntax = ['RT', 'rt', 'usermention', 'userment']

stemmerEn = PorterStemmer()

def openData():
	allData = os.listdir(trainDir)
	allData.sort()
	for file in allData:
		if file.startswith("M"):
			mData[file] = fileToEntry(trainDir + file)
		elif file.startswith("F"):
			fData[file] = fileToEntry(trainDir + file)
		else:
			print("Loading of data failed")

def fileToEntry(fileName):
	tweetFile = open(fileName, errors='replace')
	return tweetFile.read().encode('ascii','ignore').decode('ascii')
	
def lineToTokens(line):
	tokenArray = []
	tokens = line.split()
	for token in tokens:
		token = normalize(token)
		# filter tokens on stopwords and very short words
		if token != '':
			#token not in stopwordsNL and token not in stopwordsEn and token 
			if token not in twitterSyntax:
				if len(token) > 2:
					tokenArray.append(token)
	return tokenArray
	
def normalize(str):
	normWord = str.lower()
	normWord = p.sub("", normWord)
	# try to stem the words as good as possible. However, it is hard for the detector to distinguish only between En and NL words.
	# "ik" for example is Dutch, but outputs Swedish.
	normWord = stemmerEn.stem(normWord)
	if normWord.endswith('j'):
		normWord= normWord[:-1]
	elif normWord.endswith('v'):
		normWord = normWord[:-1] + 'f'
	elif normWord.endswith('z'):
		normWord = normWord[:1] + 's'
	return normWord

def tokensToNgrams(tokens, n):
	ngrams = []
	for i in range(n-1, len(tokens)):
		ngram = ""
		for j in range(0, n):
			if j == 0:
				ngram = tokens[i-j] + ngram
			else:
				ngram = tokens[i-j] + " " + ngram
		ngrams.append(ngram)
	return ngrams

def tally(ngrams):
	c = Counter()
	for ngram in ngrams:
		c[ngram] += 1
	return c

def sortCount(dict):
	dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
	return dict_sorted

# This functions can be used to computer the top-1o n-grams, and the occurences of different n-grams (for the assignment LSP1).
def vocabulary(data, n):
	totalNgrams = []
	for int in range(0,3):
		n += 1
		
		for key in mData.keys():
			tokens = lineToTokens(mData[key])
			ngrams = tokensToNgrams(tokens,n)
			for ngram in ngrams:
				totalNgrams.append(ngram)

		NgramsTally = tally(totalNgrams)

		NgramsTally = sortCount(NgramsTally)
		
		if n == 1:
			print("Top ten unigrams: ")
			for j in range(0,11):
				print(NgramsTally[j])

		print("Unique "+ str(n) +"-grams in the corpus: " + str(len(NgramsTally)))
		for i in range(1,5):
			count = 0
			for ngram in NgramsTally:
				if ngram[1] == i:
					count = count + 1
			print(str(count) + " " + str(n) + "-grams were observed for " + str(i) + " time(s)")
		print("\n")
	

# The function bagOfWords prepares the data by calling the functions lineToTokens and tokensToNgrams on the data (mData and fData)
def bagOfWords(data, n):
	totalNgrams = []
	for key in data.keys():
		tokens = lineToTokens(data[key])
		ngrams = tokensToNgrams(tokens,n)
		for ngram in ngrams:
			totalNgrams.append(ngram)

	NgramsTally = tally(totalNgrams)
	NgramsTally = sortCount(NgramsTally)
	unigramList = []
	for ngram in NgramsTally:
		if ngram[1] > 24:
			unigramList.append(ngram)
	return unigramList


def train(data, n, k):
	n += 1
	totalWordNumber = 0
	for file in data:
		totalWordNumber += len(file)
	dataFrequencies = bagOfWords(data, n)
	# a new array trainedData {} to store the trained data.
	trainedData = {}
	# now we want to calculate the probabilities for each word in the data we are using
	for i in range(0, len(dataFrequencies)):
		prob = probTrain(dataFrequencies[i][1], totalWordNumber, k, len(dataFrequencies))
		# we want to store the probabilities in the trainedData{} dictionairy combined with the right key for later use
		# thus, we store the prob and the frequency in the trainedData{} dict with key word.
		trainedData[dataFrequencies[i][0]] = [prob, dataFrequencies[i][1]]
	# we also need to include zero-count probs
	trainedData[""] = [probTrain(0, totalWordNumber, k, len(dataFrequencies)), 0, [""]]
	trainedData = sortCount(trainedData)
	return trainedData

def filter(set0, set1, treshold):
	set0 = dict(set0)
	set1 = dict(set1)
	copy0 = dict(set0)
	copy1 = dict(set1)
	for key in set0.keys():
		if key in set1.keys():
			chance = copy0[key][0] / copy1[key][0]
			if chance > 1: chance = 1/chance
			if chance > treshold:
				del copy0[key]
				del copy1[key]
	return [copy0, copy1]

# The function probTrain calculates and returns the estimated conditional probabilities for the n-grams for females and males separately.		
def probTrain(ngramFrequency, totalWordNumber, k, V):
	return (float(ngramFrequency + k)) / (float(totalWordNumber) + k * V)

def test(testDir, n, k, filteredSets):
	n += 1
	# we need a new dict to store the test data in
	testDict = {}

	testData = os.listdir(testDir)
	testData.sort()
	# we need to loop through the test data and add them to the testData dictionary
	for file in testData:
		# read test data in the same way as train data, but then do not separate female from male (because, of course, this distinction needs to be classified by the classifier).
		if file.startswith("F") or file.startswith("M"):
			testDict[file] = fileToEntry(testDir + file)
	
	# now we need to calculate the probabilities and classify the tweets!
	# we start with the count of correctly classified tweets set to zero
	correctCount = 0
	
	# then we calculate probability using the trainedM and trainedF dictionaries
	# we loop over every file, because we would like to classify all of them
	for key in testDict.keys():
		tokens = lineToTokens(testDict[key])
		testM = testProb(tokens, filteredSets[0], n)
		testF = testProb(tokens, filteredSets[1], n)
		
		# if the probability for male is higher than female and the class is indeed male then add one to correctly classified docs
		if testM > testF:
			if key.startswith("M"):
				correctCount += 1
			#else:
				#print("Incorrectly classified: " + key)
		# else classified as female, check if this is indeed correct, then add one to counter
		else:
			if key.startswith("F"):
				correctCount += 1
			#else:
				#print("Incorrectly classified: " + key)
	accuracy = float(correctCount)/len(testDict.keys())
	testList = [len(testDict.keys()), correctCount, accuracy]
	return testList
	# print outcomes
	#print("From the " + str(len(testDict.keys())) + " tweets are "+ str(correctCount) + " correctly classified")
	#print("The accuracy for k = " + str(k) + " is: "+ str(float(correctCount)/len(testDict.keys())))
	#print("\n")

	
def testProb(tokens, trainData, n):
	prob = 1
	ngrams = []
	trainData = dict(trainData)

	# we need to calculate the log of the sum of the probabilities for each word.
	for i in range(n-1, len(tokens)):
		ngram = ""
		for j in range(0, n):
			if j == 0:
				ngram = tokens[i-j] + ngram
			else:
				ngram = tokens[i-j] + " " + ngram
		# we want to see if the ngram is in the trained data set with the associated probability
		if ngram in trainData.keys():
			# we need to take the log of the probability stored in trainData to avoid underflow
			prob += log(trainData[ngram][0],2)
		else:
			prob += log(trainData[""][0],2)
	return prob
	
def charWords(train1, train2):
	train1 = dict(train1)
	train2 = dict(train2)
	# we would like to train the rank in a new dict
	wordRank = {}
	
	# we would like to compare, so we look if word is in trainedM and trainedF
	for word in train1.keys():
		if word in train2:
			# words should occur 50 times or more
			if train1[word][1] > 49 and train2[word][1] > 49:
				rank = train1[word][0] / train2[word][0]
				wordRank[word] = rank
	sort_wordRank = sortCount(wordRank)
	
	#print(sort_wordRank)
	#we would like to print the top 10 characteristic words for the train1 
	#for i in range(0,100):
		#print(sort_wordRank[i])


openData()

# ----------------------
#the line below will merge mData and fData for the assignment that requires to look at all the documents.
#mData.update(fData)
# ----------------------

# ----------------------
# For the assignment 6 -- Vocabulary, these function can be used. Then please also uncomment mData.update(fData).
#vocabulary(mData, n)
# ----------------------

# ----------------------
# Assignment 5 - Text classification using a unigram language model
# Setting the k-values and calculating the associated accuracy of the model.

#~ k = 0.01
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ test(testDir, n, k)

#~ k = 1
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ test(testDir, n, k)

#~ k = 2
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ test(testDir, n, k)

k = 10
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
accs = []
max = 0
maxindex = 0
for i in range(21):
	filteredSets = filter(trainedM, trainedF, .8+i/100)
	testList = test(testDir, n, k, filteredSets)
	tweetTotal = testList[0]
	correctCount = testList[1]
	accs.append(testList[2])
	if accs[i] > max:
		max = accs[i]
		maxindex = i
	print("From the " + str(tweetTotal) + " tweets are "+ str(correctCount) + " correctly classified")
	print("With a treshold of " + str(.8+i/100) + " the accuracy is " + str(accs[i]))
print("The highest accuracy is at the threshold of " + str(maxindex) + " with an accuracy of " + str(max))


#~ k = 50
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ test(testDir, n, k)

#~ k = 100
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ test(testDir, n, k)
#~ # ----------------------

# ----------------------
# Assignment 6 - Characteristic words with k = 10
#~ k = 10
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ test(testDir, n, k)

#~ print("Top 10 characteristic words (with k =  " + str(k) + ") for males: ")
#charWords(trainedM, trainedF)
#~ print("Top 10 characteristic words (with k =  " + str(k) + ") for females: ")
#~ charWords(trainedF, trainedM)
# ----------------------