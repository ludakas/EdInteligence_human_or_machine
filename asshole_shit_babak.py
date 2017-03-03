import numpy as np
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import cess_esp
import pickle

with open ('train_data', 'rb') as fp:
    train_data = pickle.load(fp)

def load_data(file_name):
    """
    Places data in list of labelled sentences format which
    allows gensim to build vocab automatically
    """
    with open(file_name) as f:
        lines = f.readlines()

    labels = list()
    all_dat = list()


    for i, l in enumerate(lines):

        labels.append(int(l[0]))

        l = utils.any2unicode(l)
        all_dat.append(l.split("\t")[-1])




    return all_dat, np.asarray(labels)



def form_vectors(sent):
    count_vect = CountVectorizer()

    count_vect = count_vect.fit(train_data)
    freq_term_matrix = count_vect.transform(train_data)
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    sent_freq_term = count_vect.transform([sent])
    sent_tfidf_matrix = tfidf.transform(sent_freq_term)
    #print freq_term_matrix

    return sent_tfidf_matrix


def apply_pca():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1)
    # pca.fit(X)
    print (pca.fit_transform(X))
    print X
    print pca.components_
    return 1




if __name__ == '__main__':
    data = load_data("data_sub.txt")[0]
    print data[0]

    for i in data:
        #print i
        x = form_vectors(i)
        print x
