# -*- coding: utf-8 -*-

import os, operator
import re

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

# def dictToLines(dic):
	# lines = []
	# for key in dic:
		# lines.append(dic[key].splitlines())
	# return lines
	
def lineToTokens(line):
	tokenArray = []
	tokens = line.split()
	for token in tokens:
		token = normalize(token)
		if token != '':
			tokenArray.append(token)
	return tokenArray
	
def normalize(str):
	normWord = str.lower().replace("usermention", "").replace("rt", "").replace("RT","")
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

def vocabulary(data, n):
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
		#print(type(NgramsTally))

		print("Unique "+ str(n) +"-grams in the corpus: " + str(len(NgramsTally)))
		for i in range(1,5):
			count = 0
			for ngram in NgramsTally:
				if ngram[1] == i:
					count = count + 1
			print(str(count) + " " + str(n) + "-grams were observed for " + str(i) + " time(s)")
		print("\n")
	
def selectWords(data):
	for key in mData.keys():
			tokens = lineToTokens(mData[key])
			ngrams = tokensToNgrams(tokens,1)
			for ngram in ngrams:
				totalNgrams.append(ngram)

	NgramsTally = tally(totalNgrams)
	NgramsTally = sortCount(NgramsTally)
	unigramList = []
	for ngram in NgramsTally:
		if ngram[1] > 24:
			unigramList.append(ngram)
	return unigramList

openData()

#mData.update(fData)
totalNgrams = []
totalNgramsM = selectWords(mData)
totalNgramsF = selectWords(fData)

#vocabulary(mData, n)

