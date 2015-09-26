# -*- coding: utf-8 -*-

import os, operator
import re
from nltk.corpus import stopwords
from math import log

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
stopwordsEn = stopwords.words('english')
stopwordsNL = stopwords.words('dutch')
twitterSyntax = ['RT', 'rt', 'usermention']

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
			if token not in stopwordsNL and token not in stopwordsEn and token not in twitterSyntax:
				if len(token) > 2:
					tokenArray.append(token)
	return tokenArray
	
def normalize(str):
	normWord = str.lower()
	normWord = p.sub("", normWord)
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

def mergeCounters(c1, c2):
    for key in c2.keys():
        c1[key] += c2[key]

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


# The function trainModel calculates and stores the n-grams, probabilities and counts in a new dict trainedData.
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

# The function probTrain calculates and returns the estimated conditional probabilities for the n-grams for females and males separately.		
def probTrain(ngramFrequency, totalWordNumber, k, V):
	return (float(ngramFrequency + k)) / (float(totalWordNumber) + k * V)

def test(testDir, n, k):
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
		testM = testProb(tokens, trainedM, n)
		testF = testProb(tokens, trainedF, n)
		
		# if the probability for male is higher than female and the class is indeed male then add one to correctly classified docs
		if testM > testF:
			if key.startswith("M"):
				correctCount += 1
			else:
				print("Incorrectly classified: " + key)
		# else classified as female, check if this is indeed correct, then add one to counter
		else:
			if key.startswith("F"):
				correctCount += 1
			else:
				print("Incorrectly classified: " + key)
	
	# print outcomes
	print("From the " + str(len(testDict.keys())) + " tweets are "+ str(correctCount) + " correctly classified")
	print("The accuracy for k = " + str(k) + " is: "+ str(float(correctCount)/len(testDict.keys())))
	print("\n")

	
def testProb(tokens, trainData, n):
	prob = 1
	ngrams = []
	#make trainData a dict
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
	
openData()

# ----------------------
#the line below will merge mData and fData for the assignment that requires to look at all the documents.
#mData.update(fData)
# ----------------------


# Create the list of words and frequencies of occurence for males and females.
 #nGramsM = bagOfWords(mData)
#~ nGramsF = bagOfWords(fData)

# ----------------------
# For the assignment Vocabulary, these function can be used.
#vocabulary(mData, n)
# ----------------------

k = 0.01
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 0.1
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 0.5
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 1
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 2
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 10
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 50
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

k = 100
trainedM = train(mData, n, k)
trainedF = train(fData, n, k)
test(testDir, n, k)

#print(trainedM)

