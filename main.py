from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
import gensim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pkl
from random import randint
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
sns.set()

class File2DocSpace(Doc2Vec):

    def __init__(self, file_name, epochs=1, size=80,
                 window=4,
                 min_count=2, workers=1,
                 alpha=0.5, min_alpha=0.025, batch_words=30,
                 train= True):

        self.file_name = file_name
        self.D, self.y = self.load_data()
        self.epochs = epochs
        self.dim = size
        # for scatter plot matrix functionality
        self.hues = pd.Series(self.y).map(lambda x: "blue" if x is 0 else "green")
        super(File2DocSpace, self).__init__(self.D, size=size,
                                            window=window,
                                            min_count=min_count, workers=workers,alpha=alpha,
                                            min_alpha=min_alpha, batch_words=batch_words)
        if train:
            self.train2()
            self.X = np.asarray(self.docvecs)
            self.docvecs = None # for RAM efficiency
        else:
            self.X = None

    def load_data(self):
        """
        Places data in list of labelled sentences format which
        allows gensim to build vocab automatically
        """
        with open(self.file_name) as f:
            lines = f.readlines()

        labels = list()
        all_dat = list()
        for i, l in enumerate(lines):

            labels.append(int(l[0]))

            l = gensim.utils.any2unicode(l)
            all_dat.append(LabeledSentence(l.split("\t")[-1], [i]))

        return all_dat, np.asarray(labels)

    def visualize_vec_as_mat(self):
        """
         Draws a set of samples from pargraph vector space
         and visualizes as a matrix

        """
        r1 = randint(self.dim, len(self.D) -1 )
        if self.X is not None:
            matrix = self.X[r1-self.dim:r1, :]
        else:
            matrix = np.asarray(self.docvecs)[r1-self.dim:r1, :]
        plt.matshow(matrix)


    def visualize_scatter_plot_mat(self, dim=6):
        dictio = {}
        r1 = randint(dim, (self.dim) )
        for i in range(r1-dim, r1):
            if self.X is not None:
                dictio["dimension "+ str(i)] = self.X[:,i]
            else:
                dictio["dimension "+ str(i)] = np.asarray(self.docvecs)[:,i]
        dictio["hue"] = self.hues
        df = pd.DataFrame(dictio)
        sns.pairplot(df, hue="hue")

    def train2(self):
        """
        Neural network SGD like trainign for paragraph vector estimation
        """
        for epoch in range(self.epochs):
            print "epoch: ", epoch
            self.train(self.D)
            self.alpha -= 0.002  # decrease the learning rate
            self.min_alpha = self.alpha  # fix the learning rate, no decay


with open ('models/word_model_20_20.pkl', 'rb') as fp:
    trained_model_word = pkl.load(fp)

def load_data(file_name="data/training_data.txt"):
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

        l = gensim.utils.any2unicode(l)
        all_dat.append(l.split("\t")[-1].strip())

    return all_dat, np.asarray(labels)



def load_test_data(file_name="data/test_data.txt"):
    """
    Places data in list of labelled sentences format which
    allows gensim to build vocab automatically
    """
    with open(file_name) as f:
        lines = f.readlines()

    all_dat = list()
    for i, l in enumerate(lines):

        l = gensim.utils.any2unicode(l)
        all_dat.append(l)

    return all_dat




data, labels = load_data()
test_data = load_test_data()

model_word = File2DocSpace("data/training_data.txt", epochs=0, workers=4, size=20)
model_word.X = trained_model_word[0]
model_word.y = trained_model_word[1]

whole_data = data + test_data

vect = TfidfVectorizer(min_df=0.001)
mat = vect.fit_transform(whole_data).toarray()
print mat.shape


pca = PCA(n_components=1650)
mat2 = pca.fit_transform(mat)
test_mat = mat2[len(data):len(whole_data),:]
mat2 = mat2[0:len(data),:] # get the training data back
print mat2.shape
print test_mat.shape

# vect = TfidfVectorizer(min_df=0.001)
# mat = vect.fit_transform(data).toarray()
# print mat.shape
#
# pca = PCA(n_components=1650) # keep 95% variance
# mat2 = pca.fit_transform(mat)
# print mat2.shape
# print sum(pca.explained_variance_ratio_)

model = Sequential()
model.add(Dense(1000, input_dim=1650))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(2, activation='sigmoid'))
#model.add(Activation('softmax'))

model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(mat2, model_word.y, test_size=0.2, random_state=42)
print y_test

h = model.fit(X_train, to_categorical(y_train),
          nb_epoch=100, batch_size=500, validation_data=(X_test, to_categorical(y_test)))

print model.predict_classes(test_mat)
final_results = model.predict_classes(test_mat)


# save labels
# with open("final_results.pkl", "wb") as f:
#     pkl.dump(final_results, f)
