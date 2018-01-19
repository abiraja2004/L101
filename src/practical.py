import os
import numpy as np
from gensim.models import Doc2Vec, Word2Vec, KeyedVectors
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

NEG_DIR = os.path.abspath('data/neg')
POS_DIR = os.path.abspath('data/pos')
D2V_MODEL_PATH = os.path.abspath('models/dbow.d2v')
W2V_MODEL_PATH = os.path.abspath('models/GoogleNews-vectors-negative300.bin')

d2v = Doc2Vec.load(D2V_MODEL_PATH)

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def readDirToArray(d):
    return [' '.join(open(os.path.join(d, fn)).readlines()) for fn in os.listdir(d)]

def sentencesToWordsList(sentences):
    return [s.split() for s in sentences]

negSentences = readDirToArray(NEG_DIR)
negWordsList = sentencesToWordsList(negSentences)
posSentences = readDirToArray(POS_DIR)
posWordsList = sentencesToWordsList(posSentences)

def getScores(negVectorsList, posVectorsList):
    dim = len(negVectorsList[0])
    print("dim=", dim)
    data = negVectorsList + posVectorsList
    target = [-1 for _ in negVectorsList] + [1 for _ in posVectorsList]
    acc = cross_val_score(SVC(kernel='rbf'), data, target, cv=10).mean()
    f1 = cross_val_score(SVC(kernel='rbf'), data, target, cv=10, scoring='f1_macro').mean()
    return acc, f1



#vectorizer = CountVectorizer(min_df=0.01)
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyUnigram = X[:len(negSentences)]
#posVectorsList_onlyUnigram = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyUnigram, posVectorsList_onlyUnigram)
#print('unigram rbf', ' acc=', acc, ' f1=', f1)


#vectorizer = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyStopUnigram = X[:len(negSentences)]
#posVectorsList_onlyStopUnigram = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyStopUnigram, posVectorsList_onlyStopUnigram)
#print('stopUnigram', ' acc=', acc, ' f1=', f1)


#vectorizer = CountVectorizer(min_df=1, tokenizer=LemmaTokenizer())
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyStemUnigram = X[:len(negSentences)]
#posVectorsList_onlyStemUnigram = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyStemUnigram, posVectorsList_onlyStemUnigram)
#print('stemUnigram', ' acc=', acc, ' f1=', f1)


#vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=0.01)
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyBigram = X[:len(negSentences)]
#posVectorsList_onlyBigram = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyBigram, posVectorsList_onlyBigram)
#print('only bigram', ' acc=', acc, ' f1=', f1)


#vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=0.01)
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyBigram = X[:len(negSentences)]
#posVectorsList_onlyBigram = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyBigram, posVectorsList_onlyBigram)
#print('uni+bigram poly', ' acc=', acc, ' f1=', f1)


# vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, stop_words=stopwords.words('english'))
# corpus = negSentences + posSentences
# X = vectorizer.fit_transform(corpus).toarray().tolist()
# negVectorsList_onlyStopBigram = X[:len(negSentences)]
# posVectorsList_onlyStopBigram = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyStopBigram, posVectorsList_onlyStopBigram)
# print('only stopBigram', ' acc=', acc, ' f1=', f1)


#vectorizer = TfidfVectorizer(min_df=1)
#vectorizer = TfidfVectorizer(min_df=0.005) # ignore words that occur in less than 10 documents
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyTfidf = X[:len(negSentences)]
#posVectorsList_onlyTfidf = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyTfidf, posVectorsList_onlyTfidf)
#print('unigram tfidf', ' acc=', acc, ' f1=', f1)


#vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=0.005)
#vectorizer = TfidfVectorizer(min_df=0.005) # ignore words that occur in less than 10 documents
#corpus = negSentences + posSentences
#X = vectorizer.fit_transform(corpus).toarray().tolist()
#negVectorsList_onlyTfidf = X[:len(negSentences)]
#posVectorsList_onlyTfidf = X[len(negSentences):]
#acc, f1 = getScores(negVectorsList_onlyTfidf, posVectorsList_onlyTfidf)
#print('uni+bigram tfidf+freq', ' acc=', acc, ' f1=', f1)


w2v = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)
negVectorsList_onlyW2v = [
    np.mean(
        np.array(
            [w2v[w] if w in w2v else np.array([0] * 300) for w in ws]
        ),
        axis=0
    ) for ws in negWordsList
]
posVectorsList_onlyW2v = [
    np.mean(
        np.array(
            [w2v[w] if w in w2v else np.array([0] * 300) for w in ws]
        ),
        axis=0
    ) for ws in posWordsList
]
acc, f1 = getScores(negVectorsList_onlyW2v, posVectorsList_onlyW2v)
print('w2v rbf', ' acc=', acc, ' f1=', f1)


#negVectorsList_onlyD2v = [d2v.infer_vector(ws) for ws in negWordsList]
#posVectorsList_onlyD2v = [d2v.infer_vector(ws) for ws in posWordsList]
#acc, f1 = getScores(negVectorsList_onlyD2v, posVectorsList_onlyD2v)
#print('d2v', ' acc=', acc, ' f1=', f1)
