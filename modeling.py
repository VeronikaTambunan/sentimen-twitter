# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 10:04:13 2018

@author: KALIT
"""

import pandas as pd
data_train = pd.read_csv('data_gojek_random_label_3060_stemmed.csv')

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_train.text)
print X_train_counts.shape
count_vect.vocabulary_


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape
print(tfidf_transformer.fit_transform(count_vect.fit_transform(data_train.text)).toarray())

y_emotion = data_train['emotion']

#K-FOLD
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc,classification_report

data_X = X_train_tfidf
kf = KFold(n_splits=10)
for train, test in kf.split(data_X):
    #print("%s %s\n " % (train, test))
    X = X_train_tfidf
    y = y_emotion
    #print train
    print test
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    print("Size of training set: {} size of test set: {}\n".format(X_train.shape, X_test.shape))
    
#TRAINING
#==============================================================================
#RBF
from sklearn.svm import SVC
best_scoreRBF = 0
for gamma in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
    for C in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
        # for each combination of parameters, train an SVC
        svm_rbf = SVC(kernel='rbf', gamma=gamma, C=C)
        svm_rbf.fit(X_train, y_train)
        # evaluate the SVC on the test set
        score = svm_rbf.score(X_test, y_test)
        # if we got a better score, store the score and parameters
        print (score)
        print (svm_rbf.score(X_train, y_train))
        predict = svm_rbf.predict(X_test)
        #print (svm_rbf.predict(X_test))
        
        
        mean = score
        params = {'C': C, 'gamma': gamma}
        predicted = predict
        print("%0.3f for %r" % (mean, params))

        if score > best_scoreRBF:
            best_scoreRBF = score
            best_parameters = {'C': C, 'gamma': gamma}
            best_predict = predicted

            print ("Accuracy:", accuracy_score(y_test, predicted))
            print ("Precision:", precision_score(y_test, predicted, average='macro'))
            print ("Recall:", recall_score(y_test, predicted, average='macro'))
            print ("F1:", f1_score(y_test, predicted, average='macro'))
            evaluate = [accuracy_score(y_test, predicted),precision_score(y_test, predicted, average='macro'),recall_score(y_test, predicted, average='macro'),f1_score(y_test, predicted, average='macro')]

        
#print(svm.predict(best_score))
print("Best score: {:.2f}".format(best_scoreRBF))
print("Best parameters: {}".format(best_parameters))
print("Best predict: {}".format(best_predict))

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt
print( confusion_matrix(y_test, best_predict))
print(classification_report(y_test, best_predict))
skplt.metrics.plot_confusion_matrix(y_test,best_predict)
plt.show()

color = ['Green','Blue','Red','Orange']
algo = ['Accuracy','Precision', 'Recall','F1']
fig = plt.figure()
plt.bar(np.arange(4), evaluate, color=color,bottom=0,alpha=0.7, width = 0.5,label=algo)
plt.xticks(np.arange(4), algo)
plt.ylabel('%')
plt.xlabel('Evaluation')
#plt.subplots_adjust(hspace = 0.5)
plt.show()

#==============================================================================
#LINEAR
from sklearn.svm import SVC
best_scoreLNR = 0
for C in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
        # for each combination of parameters, train an SVC
        linearsvm = SVC(kernel='linear', C=C)
        linearsvm.fit(X_train, y_train)
        # evaluate the SVC on the test set
        score = linearsvm.score(X_test, y_test)
        print (score)
        print (linearsvm.score(X_train, y_train))
        # if we got a better score, store the score and parameters
        #print (linearsvm.predict(X_test))
        predict = linearsvm.predict(X_test)
        
        mean = score
        params = {'C': C}
        predicted = predict
        print("%0.3f for %r" % (mean, params))
        
        if score > best_scoreLNR:
            best_scoreLNR = score
            best_parameters = {'C': C}
            best_predict = predicted
            
            print ("Accuracy:", accuracy_score(y_test, predicted))
            print ("Precision:", precision_score(y_test, predicted, average='macro'))
            print ("Recall:", recall_score(y_test, predicted, average='macro'))
            print ("F1:", f1_score(y_test, predicted, average='macro'))
            evaluate = [accuracy_score(y_test, predicted),precision_score(y_test, predicted, average='macro'),recall_score(y_test, predicted, average='macro'),f1_score(y_test, predicted, average='macro')]
            

print("Best score: {:.2f}".format(best_scoreLNR))
print("Best parameters: {}".format(best_parameters))
print("Best predict: {}".format(best_predict))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt
print( confusion_matrix(y_test, best_predict))
print(classification_report(y_test, best_predict))
skplt.metrics.plot_confusion_matrix(y_test,best_predict)
plt.show()

color = ['Green','Blue','Red','Orange']
algo = ['Accuracy','Precision', 'Recall','F1']
fig = plt.figure()
plt.bar(np.arange(4), evaluate, color=color,bottom=0,alpha=0.7, width = 0.5,label=algo)
plt.xticks(np.arange(4), algo)
plt.ylabel('%')
plt.xlabel('Evaluation')
#plt.subplots_adjust(hspace = 0.5)
plt.show()
#==============================================================================

#==============================================================================

#SIGMOID
from sklearn.svm import SVC
best_scoreSIG = 0
for gamma in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
    for C in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
        # for each combination of parameters, train an SVC
        sigmoid = SVC(kernel='sigmoid', C=C, gamma=gamma)
        sigmoid.fit(X_train, y_train)
        # evaluate the SVC on the test set
        score = sigmoid.score(X_test, y_test)
        # if we got a better score, store the score and parameters
        print (score)
        print (sigmoid.score(X_train, y_train))
        
        predict = sigmoid.predict(X_test)
        
        mean = score
        params = {'C': C, 'gamma':gamma}
        predicted = predict
        print("%0.3f for %r" % (mean, params))
        
        if score > best_scoreSIG:
            best_scoreSIG = score
            best_parameters = {'C': C, 'gamma':gamma}
            best_predict = predicted
            
            print ("Accuracy:", accuracy_score(y_test, predicted))
            print ("Precision:", precision_score(y_test, predicted, average='macro'))
            print ("Recall:", recall_score(y_test, predicted, average='macro'))
            print ("F1:", f1_score(y_test, predicted, average='macro'))
            evaluate = [accuracy_score(y_test, predicted),precision_score(y_test, predicted, average='macro'),recall_score(y_test, predicted, average='macro'),f1_score(y_test, predicted, average='macro')]
            
            
#print(svm.predict(best_score))
print("Best score: {:.2f}".format(best_scoreSIG))
print("Best parameters: {}".format(best_parameters))
print("Best predict: {}".format(best_predict))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt

print( confusion_matrix(y_test, best_predict))
print(classification_report(y_test, best_predict))
skplt.metrics.plot_confusion_matrix(y_test,best_predict)
plt.show()

color = ['Green','Blue','Red','Orange']
algo = ['Accuracy','Precision', 'Recall','F1']
fig = plt.figure()
plt.bar(np.arange(4), evaluate, color=color,bottom=0,alpha=0.7, width = 0.5,label=algo)
plt.xticks(np.arange(4), algo)
plt.ylabel('%')
plt.xlabel('Evaluation')
#plt.subplots_adjust(hspace = 0.5)
plt.show()
#==============================================================================

#==============================================================================
#POLYNOMIAL
from sklearn.svm import SVC
best_scorePOL = 0
for gamma in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
    for C in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
        # for each combination of parameters, train an SVC
        polynomial = SVC(kernel='poly', C=C, gamma=gamma)
        polynomial.fit(X_train, y_train)
        # evaluate the SVC on the test set
        score = polynomial.score(X_test, y_test)
        # if we got a better score, store the score and parameters
        print (score)
        print (polynomial.score(X_train, y_train))
        
        predict = polynomial.predict(X_test)
        
        mean = score
        params = {'C': C, 'gamma':gamma}
        predicted = predict
        print("%0.3f for %r" % (mean, params))
        
        if score > best_scorePOL:
            best_scorePOL = score
            best_parameters = {'C': C, 'gamma':gamma}
            best_predict = predicted

            print ("Accuracy:", accuracy_score(y_test, predicted))
            print ("Precision:", precision_score(y_test, predicted, average='macro'))
            print ("Recall:", recall_score(y_test, predicted, average='macro'))
            print ("F1:", f1_score(y_test, predicted, average='macro'))
            evaluate = [accuracy_score(y_test, predicted),precision_score(y_test, predicted, average='macro'),recall_score(y_test, predicted, average='macro'),f1_score(y_test, predicted, average='macro')]
            
#print(svm.predict(best_score))
print("Best score: {:.2f}".format(best_scorePOL))
print("Best parameters: {}".format(best_parameters))
print("Best predict: {}".format(best_predict))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt
print( confusion_matrix(y_test, best_predict))
print(classification_report(y_test, best_predict))
skplt.metrics.plot_confusion_matrix(y_test,best_predict)
plt.show()

color = ['Green','Blue','Red','Orange']
algo = ['Accuracy','Precision', 'Recall','F1']
fig = plt.figure()
plt.bar(np.arange(4), evaluate, color=color,bottom=0,alpha=0.7, width = 0.5,label=algo)
plt.xticks(np.arange(4), algo)
plt.ylabel('%')
plt.xlabel('Evaluation')
#plt.subplots_adjust(hspace = 0.5)
plt.show()
#==============================================================================

#==============================================================================
#MLP
from sklearn.neural_network import MLPClassifier

best_scoreMLP = 0
for alpha in [0.01, 0.03, 0.04, 0.06, 0.09, 0.1, 0.3, 0.6, 1, 3, 6, 9, 10]:
    for hidden_layer_sizes in [(90,90),(40,40),(30,30),(20,20)]:

        mlp = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes)
        mlp.fit(X_train, y_train)
        
        score = mlp.score(X_test, y_test)
        print (score)
        print (mlp.score(X_train, y_train))
        
        predict = mlp.predict(X_test)
        
        mean = score
        params = {'alpha': alpha, 'hidden_layer_sizes': hidden_layer_sizes}
        predicted = predict
        print("%0.3f for %r" % (mean, params))

        if score > best_scoreMLP:
            best_scoreMLP = score
            best_parameters = {'alpha': alpha, 'hidden_layer_sizes': hidden_layer_sizes}
            best_predict = predicted
            
            print ("Accuracy:", accuracy_score(y_test, predicted))
            print ("Precision:", precision_score(y_test, predicted, average='macro'))
            print ("Recall:", recall_score(y_test, predicted, average='macro'))
            print ("F1:", f1_score(y_test, predicted, average='macro'))
            evaluate = [accuracy_score(y_test, predicted),precision_score(y_test, predicted, average='macro'),recall_score(y_test, predicted, average='macro'),f1_score(y_test, predicted, average='macro')]
            
        
#print(svm.predict(best_score))
print("Best score: {:.2f}".format(best_scoreMLP))
print("Best parameters: {}".format(best_parameters))
print("Best predict: {}".format(best_predict))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scikitplot as skplt
print( confusion_matrix(y_test, best_predict))
print(classification_report(y_test, best_predict))
skplt.metrics.plot_confusion_matrix(y_test,best_predict)
plt.show()

color = ['Green','Blue','Red','Orange']
algo = ['Accuracy','Precision', 'Recall','F1']
fig = plt.figure()
plt.bar(np.arange(4), evaluate, color=color,bottom=0,alpha=0.7, width = 0.5,label=algo)
plt.xticks(np.arange(4), algo)
plt.ylabel('%')
plt.xlabel('Evaluation')
#plt.subplots_adjust(hspace = 0.5)
plt.show()
#==============================================================================


#PLOTTING
score = []
score.append(best_scoreRBF)
score.append(best_scoreLNR)
score.append(best_scoreSIG)
score.append(best_scorePOL)
score.append(best_scoreMLP)
score

color = ['Green','Blue','Yellow','Red','Purple']
algo = ['SVM-RBF','SVM-LINEAR','SVM-SIGMOID','SVM-POLY','MLP']
fig = plt.figure()
plt.bar(np.arange(5), score, color=color,bottom=0, width=0.5,alpha=0.7,label=algo)
plt.xticks(np.arange(5), algo)
plt.ylabel('Accuracy %')
plt.xlabel('Algorithm')
#plt.subplots_adjust(hspace = 0.5)
plt.show()