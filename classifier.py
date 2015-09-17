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
	
openData()

maleNgrams = []

for key in mData.keys():
    tokens = lineToTokens(mData[key])
    ngrams = tokensToNgrams(tokens,4)
    for ngram in ngrams:
        maleNgrams.append(ngram)

NgramsTally = tally(maleNgrams)

NgramsTally = sortCount(NgramsTally)

print(len(NgramsTally))

#~ def dictionary(lines):
	#~ word = lineToTokens(lines)
	#~ for i in range(1,4):
		#~ ngrams = tokenToNgram(word, i)
		#~ sorted_ngrams = sortCount(tally(ngrams))
		#~ for j in range(0,11):
			#~ print(sorted_ngrams[j])
		
		#~ for j in range(1,5):
			#~ count = 0;
			#~ for word in sorted_ngrams:
				#~ if word[1] == j:
					#~ count += 1
			#~ print(str(count) + " " + str(i) + "-grams occur " + str(j) + " time(s)")
		#~ print( "Unique n-grams with n=" + str(i) + ": " + str(len(sorted_ngrams)))
		#~ print ("")


#files = fData.values()
#print(files)

#print(dest)

#~ dest = dict(list(fData.items()) + list(mData.items()))

#~ for key in dest.items():
	#~ dictionary(dest[key].values())










