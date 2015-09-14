from sys import stdin
import numpy as np
import math

#searches stopwords in corpus, creates lists in which positions of stopwords are (list from before)
def findStopwords(Corpus, stopwords):	
	Liste = stopwords[:]
	for i in range (0, len(Liste)):
		Liste[i] = []
	for i in range(0,len(Corpus)):
		n = 0
		while (Corpus[i] != stopwords[n]) and n<len(stopwords)-1:
			n = n+1
		if (Corpus[i] == stopwords[n]):
			Liste[n].append(i)
	return Liste

#creates incidence matrix of a given graph
def getStopWordGraph(Corpus, stopwords):
	Stopwordliste = findStopwords(Corpus, stopwords)
	n = len(Stopwordliste)
	a = np.zeros(shape=(n,n))
	for i in range(0,n):
		if Stopwordliste[i]!=None:
			for l in range(0, len(Stopwordliste[i])):
				for j in range(i, n):
					for k in range(0, len(Stopwordliste[j])):
						if Stopwordliste[j][k] < Stopwordliste[i][l]:
							a[i][j] = a[i][j]+math.exp(-abs(Stopwordliste[i][l]-Stopwordliste[j][k]))
							a[j][i] = a[i][j]
	return(a)
#calculates Kullback-Leibler Divergence for P and Q
def KullLeibDiv(P, Q):
	KL1 = 0
	KL2 = 0
	for i in range (0, len(P)):
		if (P[i]!=0 and Q[i]!=0):
			KL1 = KL1 + P[i] * math.log(P[i]/Q[i])
			KL2 = KL2 + Q[i] * math.log(Q[i]/P[i])
	KL = (KL1 + KL2)/2
	return(KL)

#normalizes edge weights for a given graph
def normalizeWeights(G):
	for i in range(0,len(G)):
		sum = 0
		for j in range(0,len(G)):
			sum = sum + G[i][j]
		if sum != 0:
			for j in range(0,len(G)):
				G[i][j] = G[i][j]/sum
	return G

def authorKLdiv(G1,G2,GTst):
	eps = np.finfo(float).eps
	#set to 0 if stopword is not in the test text,
	for i in range(0,len(GTst)):
		for j in range(0,len(GTst)):
			if (GTst[i][j] == 0):
				G1[i][j] = 0
				G2[i][j] = 0
	#normalize Edge weights for the three graphs
	G1 = normalizeWeights(G1)
	G2 = normalizeWeights(G2)
	GTst = normalizeWeights(GTst)
	#replace 0 with eps for all edges
	for i in range(0,len(GTst)):
		for j in range(0,len(GTst)):
			if (GTst[i][j] == 0):
				GTst[i][j] = eps
				G1[i][j] = eps
				G2[i][j] = eps
			else:
				if (G1[i][j] == 0):
					G1[i][j] = eps
				if (G2[i][j] == 0):
					G2[i][j] = eps
	KL1 = 0
	KL2 = 0
	for i in range(0,len(GTst)):
		kl1 = KullLeibDiv(G1[i],GTst[i])
		kl2 = KullLeibDiv(G2[i],GTst[i])
		KL1 = KL1 + kl1
		KL2 = KL2 + kl2
	if (KL1<KL2):
		return 1
	else:
		return 2

#puts corpus in the right form, removes punctuation marks etc. from corpus
def prepareCorpus(Corpus):
	Corpus = line.split()
	for i in range(0,len(Corpus)):
		Corpus[i] = Corpus[i].lower()
		Corpus[i] = Corpus[i].replace(".","")
		Corpus[i] = Corpus[i].replace(",","")
		Corpus[i] = Corpus[i].replace(";","")
		Corpus[i] = Corpus[i].replace("!","")
		Corpus[i] = Corpus[i].replace("?","")
		Corpus[i] = Corpus[i].replace(":","")
	return Corpus

C1 = open("Corpus1.txt")
C2 = open("Corpus2.txt")
Tst = open("Test.txt")
stopwords = open("Stopwordliste.txt")
for line in stopwords:
	stopwords = line.split()
for line in C1:
	C1 = prepareCorpus(C1)
for line in C2:
	C2 = prepareCorpus(C2)
for line in Tst:
	Tst = prepareCorpus(Tst)
G1 = getStopWordGraph(C1, stopwords)
G2 = getStopWordGraph(C2, stopwords)
GTst = getStopWordGraph(Tst, stopwords)
result = authorKLdiv(G1,G2,GTst)
print(result)
