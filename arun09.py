from copy import deepcopy
import numpy as np
import math
import logging
import jsonhandler
import argparse

# create logger
logger = logging.getLogger("logging_tryout2")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

logger.info("Program started")

# number of stopwords that should be looked on
number = 3

# puts corpus in the right form, removes punctuation marks from corpus, puts all letters in lower case
# input Corpus in string from
# output list of words in lower case without puncuation


def prepareCorpus(C):
    Corpus = []
    Corpus = C.split()
    for i in range(0, len(Corpus)):
        Corpus[i] = Corpus[i].lower()
        Corpus[i] = Corpus[i].replace("\n", "")
        Corpus[i] = Corpus[i].replace(".", "")
        Corpus[i] = Corpus[i].replace(",", "")
        Corpus[i] = Corpus[i].replace(";", "")
        Corpus[i] = Corpus[i].replace("!", "")
        Corpus[i] = Corpus[i].replace("?", "")
        Corpus[i] = Corpus[i].replace(":", "")
        Corpus[i] = Corpus[i].replace("'", "")
        Corpus[i] = Corpus[i].replace("`", "")
        Corpus[i] = Corpus[i].replace("´", "")
    return Corpus


def addStopword(Corpus, index, stopwords, Liste):
    n = 0
    subListe = []
    while (Corpus[index] != stopwords[n] and n < len(stopwords) - 1):
        n = n + 1
    if (Corpus[index] == stopwords[n]):
        subListe = [n, index]
        Liste.append(subListe)
    return Liste

# Input: Corpus in form of a list, list of stopwords
# Output: Stop words of the first n stopwords, that are in the corpus,
# less if there are less in there


def createStopwordliste(Corpus, stopwords):
    Liste = []
    i = 0
# subListe consists of tupel: (Stopwort, Position in text)
    while len(Liste) < number and i < len(Corpus):
        addStopword(Corpus, i, stopwords, Liste)
        i = i + 1
    return Liste

# adds a stopword at the end of the list of n stopwords as long as end of corpus is not reached
# removes the first stopword in the list
# Input: Corpus as list, stop word list of length n, list of stopwords
# Output: new list of stopwords of length n or less is end of corpus is reached


def adjustStopwordliste(Corpus, Liste, stopwords):
    i = Liste[number - 1][1] + 1
    while len(Liste) == number and i < len(Corpus):
        Liste = addStopword(Corpus, i, stopwords, Liste)
        i = i + 1
    Liste.pop(0)
    return Liste

# creates incidence matrix of a given graph
# Input: Corpus in list form, list of stopwords
# Output: StopwordGraph as nxn Matrix


def getStopWordGraph(Corpus, stopwords):
    Liste_n = createStopwordliste(Corpus, stopwords)
    n = len(stopwords)
    a = np.zeros(shape=(n, n))
    for i in range(1, len(Liste_n)):
        for j in range(0, i):
            k = Liste_n[i][0]
            l = Liste_n[j][0]
            a[k][l] = a[k][l] + math.exp(-abs(Liste_n[i][1] - Liste_n[j][1]))
            a[l][k] = a[k][l]
    while (len(Liste_n) == number):
        Liste_n = adjustStopwordliste(Corpus, Liste_n, stopwords)
        for i in range(0, len(Liste_n)):
            k = Liste_n[len(Liste_n) - 1][0]
            l = Liste_n[i][0]
            a[k][l] = a[k][l] + math.exp(
                -abs(Liste_n[len(Liste_n) - 1][1] - Liste_n[i][1]))
            a[l][k] = a[k][l]
    return(a)

# calculates Kullback-Leibler Divergence for P and Q
# Input: i-th line of first matrix (array) and i-th line of second matrix (array)
# Output: KL divergence (float)


def KullLeibDiv(P, Q):
    KL1 = 0
    KL2 = 0
    for i in range(0, len(P)):
        if (P[i] != 0):
            if (Q[i] != 0):
                KL1 = KL1 + P[i] * math.log(P[i] / Q[i])
                KL2 = KL2 + Q[i] * math.log(Q[i] / P[i])
    KL = (KL1 + KL2) / 2
    return(KL)

# normalizes edge weights for a given graph
# Input: Graph in form of nxn Matrix
# Output: Normalized graoh in form of nxn Matrix (lines add up to 1)


def normalizeWeights(G):
    for i in range(0, len(G)):
        sum = 0
        for j in range(0, len(G)):
            sum = sum + G[i][j]
        if sum != 0:
            for j in range(0, len(G)):
                G[i][j] = G[i][j] / sum
    return G

# calculates total KL divergence between two graphs in form of nxn matrices
# Input: two graphs
# Output: total KL divergence (float)


def authorKLdiv(G, GTst):
    G2 = deepcopy(G)
    # set to 0 if stopword is not in the test text,
    for i in range(0, len(GTst)):
        for j in range(0, len(GTst)):
            if (GTst[i][j] == 0):
                G2[i][j] = 0
    # normalize Edge weights for the three graphs
    G2 = normalizeWeights(G2)
    kl = 0
    KL = 0
    for i in range(0, len(GTst)):
        kl = KullLeibDiv(G2[i], GTst[i])
        KL = KL + kl
    return (KL)

# reads in function, creates output file


def tira(corpusdir, outputdir):
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()

    stopwords = open("stopwords.txt")
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
        logger.info("Reads in text " + str(author) + "...")
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
        logger.info("Reads in test document " + str(author) + "...")
    #    raw[author] = open(author,encoding='iso-8859-1')
        raw_test[author] = jsonhandler.getUnknownText(author)
        C_test[author] = prepareCorpus(raw_test[author])
        logger.info("Calculates Stopword Graph " + str(author) + "...")
        Gtests[author] = getStopWordGraph(C_test[author], stopwords)
    results = []
    for testcase in tests:
        print(testcase)
        KL = {}
        Gtst = deepcopy(Gtests[testcase])
        Gtst = normalizeWeights(Gtst)
        for author in authors:
            logger.info("Calculates KL Divergence of " + str(author) + "...")
            KL[author] = authorKLdiv(Gauthors[author], Gtst)
        print(KL)
        # m = np.argmin(KL)
        m = min(KL, key=KL.get)
        results.append((testcase, m))
    texts = [text for (text, cand) in results]
    cands = [cand for (text, cand) in results]

    jsonhandler.storeJson(outputdir, texts, cands)

# main function
# run program via commando line: python arun09.py -i PATH_OF_INPUT_FOLDER
# -o PATH_OF_OUTPUT_FOLDER


def main():
    parser = argparse.ArgumentParser(description="Tira submission")
    parser.add_argument("-i", action="store")
    parser.add_argument("-o", action="store")

    args = vars(parser.parse_args())

    corpusdir = args["i"]
    outputdir = args["o"]

    tira(corpusdir, outputdir)

if __name__ == "__main__":
    main()
    print(number)
