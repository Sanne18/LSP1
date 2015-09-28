# -*- coding: utf-8 -*-

#import everything we need.
import os, operator
import re
from math import log
from nltk.stem.porter import *

# Set the directories for the datasets.
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
# Punctuation is removed from the n-grams.
p = re.compile("[~.,'\":;!@#$%^&*()_\-+=?/|\u201C\u201D\u2018\u2019]")
# This is syntax we want to exclude, because it does not add relevant information for our classifier.
twitterSyntax = ['RT', 'rt', 'usermention', 'userment']

# This stemmer is used for stemming the words.
stemmerEn = PorterStemmer()

# This function opens and reads the data to two dictionaries mData{} and fData{}.
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

# This function is called from openData() and returns the read text.
def fileToEntry(fileName):
	tweetFile = open(fileName, errors='replace')
	return tweetFile.read().encode('ascii','ignore').decode('ascii')

# This function normalizes the words. It checks if the word is not in twitterSyntax and if it is longer than 2 characters long.
# This function returns a list normalized tokens providing relevant information for the classifier.
def lineToTokens(line):
	tokenList = []
	tokens = line.split()
	for token in tokens:
		token = normalize(token)
		# filter tokens on stopwords and very short words
		if token != '':
			if token not in twitterSyntax:
				if len(token) > 2:
					tokenList.append(token)
	return tokenList

# This function actually normalizes the tokens. It converts words to lowercase and deletes punctuation.
# It also manually changes 'v' at the of words to 'f' and 'z' to 's' and deletes a 'j' at the end of a word ("katj" -> "kat")
# However, this is done manually and is far from sophisticated.
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

# This function takes the tokens and creates n-gram of them, based on a given 'n'.
# It returns a list containing the ngrams.
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

# This function counts and returns the frequencies of a n-gram.
def tally(ngrams):
	c = Counter()
	for ngram in ngrams:
		c[ngram] += 1
	return c

# This function sorts the dictionary containing the ngrams based on frequency of occurrence. 
# With highest frequency first, in descending order.
def sortCount(dict):
	dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
	return dict_sorted

# This function can be used to computer the top-1o n-grams, and the occurences of different n-grams.
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
# The ngrams are added to an list, counted, sorted and included for further analysis if they occur 25 times or more.
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

# This function is used to train the data in the train dataset. It basically takes the data, frequencies for the words are calculated by using the bagOfWords() function.
# Probabilities are calculated by probTrain() function. A dictionary containing the sorted trainedData is returned (either female or male data, this is given as an argument).
def train(data, n):
	# we need unigrams, n = 0, so increment n with 1.
	n += 1
	totalWordNumber = 0
	for file in data:
		totalWordNumber += len(file)
	dataFrequencies = bagOfWords(data, n)
		# a new dictionary trainedData {} to store the trained data.
	#~ trainedData = {}
	#~ # now we want to calculate the probabilities for each word in the data we are using
	#~ for i in range(0, len(dataFrequencies)):
		#~ prob = probTrain(dataFrequencies[i][1], totalWordNumber)
		#~ # we want to store the probabilities in the trainedData{} dictionary combined with the right key for later use
		#~ # thus, we store the prob and the frequency in the trainedData{} dict with key word.
		#~ trainedData[dataFrequencies[i][0]] = [prob, dataFrequencies[i][1]]
	#~ trainedData = sortCount(trainedData)
	#~ #print(trainedData)
	return dataFrequencies

#~ # The function probTrain calculates and returns the estimated conditional probabilities for the n-grams for females and males separately.		
#~ def probTrain(ngramFrequency, totalWordNumber):
	#~ return float(ngramFrequency) / float(totalWordNumber)

# This function is added to exclude 'stop words' in the train data from analysis. These are words that do not add relevant information to the classifier.
# The chance for a word for males is divided by the chance for a word by females. For results > 1 --> 1/result
# If this result r is 0.88 > r <= 1 than this word is not distinctive enough for classification and these words are excluded from analysis.
def filter(set0, set1, threshold):
	set0 = dict(set0)
	set1 = dict(set1)
	copy0 = dict(set0)
	copy1 = dict(set1)
	for key in set0.keys():
		if key in set1.keys():
			chance = copy0[key] / copy1[key]
			if chance > 1: chance = 1/chance
			if chance > threshold:
				del copy0[key]
				del copy1[key]
	return [copy0, copy1]

# This function computes the performance of the classifier. Using the test files, and specified n and k, and the filtered data sets (see Filter() function above).
# It uses the testProb() function for computing the test probabilities. 
# It then compares the probabilities (is chance that it is a male higher than female or vice versa?).
# If chanceX is higher than chanceY and the document is indeed of class X, than correctCount is incremented with 1.
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
		testResults = testProb(tokens, n, k, filteredSets)
		#testF = testProb(tokens, n, k, filteredSets)
		
		# if the probability for male is higher than female and the class is indeed male then add one to correctly classified docs
		if testResults[0] < testResults[1]:
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
	# Accuracy is the number of correctly classified documents divided by the total number of documents classified.
	accuracy = float(correctCount)/len(testDict.keys())
	testList = [len(testDict.keys()), correctCount, accuracy]
	return testList

# This function calculates the test probabilities. The chance for a document to be of class X/Y. 
# It returns this probability.
def testProb(tokens, n, k, filteredSets):

	mChance = 0.5
	fChance = 0.5
	maleTrainData = filteredSets[0]
	femaleTrainData = filteredSets[1]
	V = len(filteredSets)
	m_chanceList = []
	f_chanceList = []
	m_logProb = 0
	f_logProb = 0
	
	for ngram in tokens:
		male_chance = 0
		female_chance = 0
		for key in maleTrainData.keys():
			if key == ngram:
				male_chance = (maleTrainData[ngram] + k) / (len(maleTrainData) + k * V)
				m_chanceList.append((ngram, male_chance))
		if male_chance == 0:
			male_chance = (k) / (len(maleTrainData) + k * V)
			m_chanceList.append((ngram, male_chance))
		for key in femaleTrainData.keys():
			if key == ngram:
				female_chance = (femaleTrainData[ngram] + k) / (len(femaleTrainData) + k * V)
				f_chanceList.append((ngram, female_chance))
		if female_chance == 0:
			female_chance = (k) / (len(femaleTrainData) + k * V)
			f_chanceList.append((ngram, female_chance))

	for i in m_chanceList:
		m_logProb += log(m_chanceList[i][1],2)
	for i in f_chanceList:
		f_logProb += log(f_chanceList[i][1],2)
	
	#print( "logprob male: " + str(m_logProb))
	#print( "logprob female: " + str(f_logProb))
	probMale = (m_logProb * mChance) / (m_logProb * mChance + f_logProb  * fChance)
	probFemale = (f_logProb * fChance) / (f_logProb * fChance + m_logProb * mChance)
	
	return [probMale, probFemale]

# This function outputs a sorted list (descending) of characteristic words for either males or females.	
def charWords(train1, train2):
	train1 = dict(train1)
	train2 = dict(train2)
	# we would like to train the rank in a new dict
	wordRank = {}
	
	# we would like to compare, so we look if word is in trainedM and trainedF
	for word in train1.keys():
		if word in train2:
			# words should occur 50 times or more to avoid rare words to appear in the top of the list.
			if train1[word][1] > 49 and train2[word][1] > 49:
				rank = train1[word][0] / train2[word][0]
				wordRank[word] = rank
	sort_wordRank = sortCount(wordRank)
	
	#we would like to print the top 10 characteristic words for the train1 
	for i in range(0,11):
		print(sort_wordRank[i])

openData()

# ----------------------
#the line below will merge mData and fData for the assignment that requires to look at all the documents.
#mData.update(fData)
# ----------------------

# ----------------------
# For the assignment 4 -- Vocabulary, these function can be used. Then please also uncomment mData.update(fData).
# Vocabulary uses the raw data mData, and does not take into account stop words. These were later excluded in assignment 5.
#vocabulary(mData, n)
# ----------------------

# ----------------------
# Assignment 5 - Text classification using a unigram language model
# Setting the k-values and calculating the associated accuracy of the model.
# Training and testing the model and eventually print the result.

#~ k = 1
#~ trainedM = train(mData, n, k)
#~ trainedF = train(fData, n, k)
#~ filteredSets = filter(trainedM, trainedF, .88)
#~ testList = test(testDir, n, k, filteredSets)
#~ tweetTotal = testList[0]
#~ correctCount = testList[1]
#~ acc = testList[2]
#~ print("From the " + str(tweetTotal) + " tweets are "+ str(correctCount) + " correctly classified")
#~ print("The accuracy is " + str(acc))

# K = 10 showed to be the best value for K (resulting in the highest accuracy with n = 1).
k = 10
trainedM = train(mData, n)
trainedF = train(fData, n)
filteredSets = filter(trainedM, trainedF, .88)
testList = test(testDir, n, k, filteredSets)
tweetTotal = testList[0]
correctCount = testList[1]
acc = testList[2]
print("K is  " + str(k))
print(str(correctCount) + " out of "+ str(tweetTotal)  + " were correctly classified")
print("The accuracy is " + str(acc))
#~ # ----------------------

# ----------------------
# Assignment 6 - Characteristic words with k = 10

#~ print("Top 10 characteristic words (with k =  " + str(k) + ") for males: ")
#~ charWords(filteredSets[0], filteredSets[1])
#~ print("Top 10 characteristic words (with k =  " + str(k) + ") for females: ")
#~ charWords(filteredSets[1], filteredSets[0])

# ----------------------