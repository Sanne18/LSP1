# -*- coding: utf-8 -*-

import os, operator
import re

trainDir = "./train/"
testDir = "./test/"

mData = {}
fData = {}
mTokens = {}
fTokens = {}
p = re.compile("[<>~.,'\":;!@#$%^&*()_\-+=?/|\u201C\u201D\u2018\u2019]")

class Counter(dict):
	def __missing__(self, key):
		return 0

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
	#print(tokens)
	for token in tokens:
		token = normalize(token)
		#print(token)
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

def frequency(ngrams, n):
    for i in range(1,5):
        count = 0
        for ngram in ngrams:
            if ngram[1] == i:
                count += 1
    return count

openData()

maleNgrams = []

for key in mData.keys():
    tokens = lineToTokens(mData[key])
    for i in range(1,4):
        ngrams = tokensToNgrams(tokens,i)

        for ngram in ngrams:
            maleNgrams.append(ngram)
            
        NgramsTally = tally(maleNgrams)
        NgramsTally = sortCount(NgramsTally)
        frequency(NgramsTally, i)
            #~ for j in range(1,5):
                #~ count = 0
                #~ if NgramsTally == j: count += 1
                #print(str(count) + " " + str(i) + "-gram are observed" + str(j) + "times")
print("unique n-grams with n=" + str(i) + ": " + str(len(NgramsTally)))

print(type(NgramsTally))

