from copy import deepcopy
import numpy as np
import math
import logging
import jsonhandler
import argparse

#logging.basicConfig(level=logging.INFO)

# create logger
logger = logging.getLogger("logging_tryout2")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s","%Y-%m-%d %H:%M:%S")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

logger.info("Program started")

#number of stopwords that should be looked on
number = 5

#puts corpus in the right form, removes punctuation marks etc. from corpus
def prepareCorpus(C):
    Corpus = []
    Corpus = C.split()
    for i in range(0,len(Corpus)):
        Corpus[i] = Corpus[i].lower()
        Corpus[i] = Corpus[i].replace("\n","")
        Corpus[i] = Corpus[i].replace(".","")
        Corpus[i] = Corpus[i].replace(",","")
        Corpus[i] = Corpus[i].replace(";","")
        Corpus[i] = Corpus[i].replace("!","")
        Corpus[i] = Corpus[i].replace("?","")
        Corpus[i] = Corpus[i].replace(":","")
        Corpus[i] = Corpus[i].replace("'","")
        Corpus[i] = Corpus[i].replace("`","")
        Corpus[i] = Corpus[i].replace("´","")
    return Corpus

def addStopword(Corpus, index, stopwords, Liste):
    n = 0
    subListe = []
    while (Corpus[index] != stopwords[n] and n <len(stopwords)-1):
        n = n+1
    if (Corpus[index] == stopwords[n]):
        subListe = [n,index]
        Liste.append(subListe)
    return Liste

def createStopwordliste(Corpus, stopwords):
    Liste = []
    i = 0
#subListe enthält Tupel mit Stopwort und Position im Text
    while len(Liste)<number and i<len(Corpus):
        addStopword(Corpus, i, stopwords, Liste)
        i = i+1
    return Liste

def adjustStopwordliste(Corpus, Liste, stopwords):
    i = Liste[number-1][1]+1
    while len(Liste) == number and i<len(Corpus):
        Liste = addStopword(Corpus,i,stopwords, Liste)
        i = i+1
    Liste.pop(0)
    return Liste

#creates incidence matrix of a given graph
def getStopWordGraph(Corpus, stopwords):
    Liste_n = createStopwordliste(Corpus, stopwords)
    n = len(stopwords)
    a = np.zeros(shape=(n,n))
#geht Stopwordliste durch
    for i in range(1,len(Liste_n)):
        for j in range(0, i):
            k = Liste_n[i][0]
            l = Liste_n[j][0]
            a[k][l] = a[k][l]+math.exp(-abs(Liste_n[i][1]-Liste_n[j][1]))
            a[l][k] = a[k][l]
    while (len(Liste_n) == number):
        Liste_n = adjustStopwordliste(Corpus, Liste_n, stopwords)
        for i in range(0,len(Liste_n)):
            k = Liste_n[len(Liste_n)-1][0]
            l = Liste_n[i][0]
            a[k][l] = a[k][l]+math.exp(-abs(Liste_n[len(Liste_n)-1][1]-Liste_n[i][1]))
            a[l][k] = a[k][l]
    return(a)

#calculates Kullback-Leibler Divergence for P and Q
def KullLeibDiv(P, Q):
    KL1 = 0
    KL2 = 0
    for i in range (0, len(P)):
        if (P[i] != 0):
            if (Q[i] != 0):
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

def authorKLdiv(G,GTst):
    G2 = deepcopy(G)
    GTst2 = deepcopy(GTst)    
    #set to 0 if stopword is not in the test text,
    for i in range(0,len(GTst2)):
        for j in range(0,len(GTst2)):
            if (GTst2[i][j] == 0):
                G2[i][j] = 0
    #normalize Edge weights for the three graphs
    G2 = normalizeWeights(G2)
    GTst2 = normalizeWeights(GTst2)
    kl = 0
    KL = 0
    for i in range(0,len(GTst2)):
        kl = KullLeibDiv(G2[i],GTst2[i])
        KL = KL + kl
    return (KL)
 
 
def tira(corpusdir, outputdir):
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()    
    
    stopwords = open("Stopwordliste.txt")
    logger.info("Reads in stopwords")
    for line in stopwords:
        stopwords = line.split()
    authors = jsonhandler.candidates
    tests = jsonhandler.unknowns
    raw = {}
    raw_test = {}
    C = {}
    C_test = {}
    Gauthors = {}
    Gtests = {}
    for author in authors:
        logger.info("Reads in text "+ str(author) + "...")
    #    raw[author] = open(author,encoding='iso-8859-1')
        for training in jsonhandler.trainings[author]:
            newtext = jsonhandler.getTrainingText(author, training)
            if author in raw.keys():
                if len(newtext) > len(raw[author]):
                    raw[author] = newtext
            else:
                raw[author] = newtext
        C[author] = prepareCorpus(raw[author])
        logger.info("Calculates Stopword Graph of " + str(author) + "...")
        Gauthors[author] = getStopWordGraph(C[author], stopwords)
    for author in tests:
        logger.info("Reads in test document "+ str(author) + "...")
    #    raw[author] = open(author,encoding='iso-8859-1')
        raw_test[author] = jsonhandler.getUnknownText(author)
        C_test[author] = prepareCorpus(raw_test[author])
        logger.info("Calculates Stopword Graph "+ str(author) + "...")
        Gtests[author] = getStopWordGraph(C_test[author], stopwords)
    results = []
    for testcase in tests:
        print(testcase)
        KL = {}
        for author in authors:
            logger.info("Calculates KL Divergence of " + str(author) + "...")
            KL[author] = authorKLdiv(Gauthors[author], Gtests[testcase])
        print(KL)
        #m = np.argmin(KL)
        m = min(KL, key=KL.get)
        results.append((testcase, m))
    texts = [text for (text,cand) in results]
    cands = [cand for (text,cand) in results]
    
    jsonhandler.storeJson(outputdir, texts, cands)
    

def main():
    parser = argparse.ArgumentParser(description="Tira submission")
    parser.add_argument("-i",action="store")
    parser.add_argument("-o",action="store")
    
    args = vars(parser.parse_args())
    
    corpusdir = args["i"]
    outputdir = args["o"]
    
    tira(corpusdir,outputdir)
    
if __name__ == "__main__":
    main()
