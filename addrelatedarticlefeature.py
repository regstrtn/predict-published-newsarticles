import numpy as np
import os
import pickle
import codecs
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy
import random
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing

from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA



model_path = os.path.dirname(os.path.abspath(__file__))+"/TrainedClassifiers"
base_path = os.path.dirname(os.path.abspath(__file__))+"/GuardianFeaturesExtracted"

'''
names = ["NN", "SVM", "RBFSVM", "DT",
         "RF", "AdaBoost", "NB", "LinearDiscriminantAnalysis",
         "QuadraticDiscriminantAnalysis"]
'''
'''
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1,probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
'''
names = ["SVM", "RBFSVM"]



classifiers = [
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1,probability=True)]
	
	
def train_model(X_train,y_train,d1):

    print("training started")

    for algo, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        with open(model_path+"/"+algo +"count"+ str(d1)+'.pkl', 'wb') as f1:
            pickle.dump(clf, f1)
        print("Trained with "+algo)
	
	

'''
def load_train_dataset(train_path):
	files = os.listdir(train_path)
	
	X_train0 = numpy.loadtxt('GuardianFeaturesExtracted/0.txt')
	X_train1 = numpy.loadtxt('GuardianFeaturesExtracted/1.txt')
	Y_train0 = numpy.zeros((len(X_train0),), dtype=np.int)
	Y_train1 = numpy.ones((len(X_train1),), dtype=np.int)
	X_train = numpy.concatenate((X_train0,X_train1))
	y_train = numpy.concatenate((Y_train0,Y_train1))
	X_train = numpy.array(X_train)
	y_train = numpy.array(y_train)
	print(len(X_train))
	print(len(y_train))
	return X_train,y_train
'''
	

def retwithrelarticle(data_class,data_class_test,data_class_keep,allFeatureVal,count):
		
		ConvertToNumAll = []
		posneg = 0
		words300 = []
		words1500  = []
		for i in data_class_test:
			#print i
			word = i["tagList"].split('&')
			words300.append(word)
		for i in data_class:
			word = i["tagList"].split('&')
			words1500.append(word)

		for i in data_class_keep:
			relarticle300 = 0
			string1 = i["tagList"].split('&')
			for j in words300:				
				listcom = set(string1)&set(j) # we don't need to list3 to actually be a list
				if len(listcom) > 0:
					relarticle300 = relarticle300 + 1
					
			
			relarticle1500 = 0
			for j in words1500:
				listcom = set(string1)&set(j) # we don't need to list3 to actually be a list
				if len(listcom) > 0:
					relarticle1500 = relarticle1500 + 1

			
			if allFeatureVal[count-300+posneg]["url"] == i["url"]:
				#print posneg
				thistoadd = {"url":allFeatureVal[count-300+posneg]["url"], "abstract":allFeatureVal[count-300+posneg]["abstract"], "authorList":allFeatureVal[count-300+posneg]["authorList"], \
				"tagList":allFeatureVal[count-300+posneg]["tagList"], "dateList":allFeatureVal[count-300+posneg]["dateList"], "productionOffice":allFeatureVal[count-300+posneg]["productionOffice"],\
				"topics": allFeatureVal[count-300+posneg]["topics"], "numberOfWords":allFeatureVal[count-300+posneg]["numberOfWords"], "tonetype": allFeatureVal[count-300+posneg]["tonetype"], "timestamp": allFeatureVal[count-300+posneg]["timestamp"], "relarticle300": relarticle300, "relarticle1500":relarticle1500, "output":allFeatureVal[count-300+posneg]["output"]}
			
			
			ConvertToNumAll.append(thistoadd)
			posneg = posneg+1
		
		
		ConvertToNumAll = numpy.array(ConvertToNumAll) 	
		#print len(ConvertToNumAll)
		return ConvertToNumAll

def main():	
	allFeatureVal = numpy.load('FeatureInValueForm/FeatureInValueForm6MonthTrain.npy')
	print(len(allFeatureVal))
	relatedarticletonum = numpy.load('FeatureInTextFormSorted/FeatureInTextFormSorted.npy')	
	print(len(relatedarticletonum))
	data_class = []
	data_class_test = []
	data_class_keep = []
	
	count = 0
	for i in range(len(allFeatureVal)):
		if allFeatureVal[i]["url"] == relatedarticletonum[i]["url"]:
			count = count +1
		else :
			print("I m here prob")

	print count
	count = 0
	ConvertToNumAll = []
	
	for data in relatedarticletonum:
		
		if count > 1799:
		
			data_class_keep.append(data)
			
			if count%1000 == 1:
				print(count)
			count = count+1
			
			if (len(data_class_keep)) == 300:
				
				withrelarticle = retwithrelarticle(data_class,data_class_test,data_class_keep,allFeatureVal,count)
				posneg = 0
				for j in withrelarticle:
					ConvertToNumAll.append(j)
					
				data_class_test = []
				data_class_forthis = []
				
				posneg = 0
				for i in data_class:
					posneg = posneg+1
					if posneg > 300:
						data_class_forthis.append(i)
				
				for i in data_class_keep:
					data_class_forthis.append(i)
					data_class_test.append(i)
				
				data_class_keep = []				
				data_class = data_class_forthis
				print(len(ConvertToNumAll))
				
		else:
			count = count+1
			data_class.append(data)
			if count > 1500:
				data_class_test.append(data)
				
	numpy.save('FeatureInValueForm/FeatureInValueFormwithRelatedArticle.npy',ConvertToNumAll)
	
if __name__ == '__main__':
    main()
	