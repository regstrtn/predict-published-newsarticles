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
names = ["SVM"]



classifiers = [
    SVC(kernel="linear", C=0.025, probability=True)]
	
	
def train_model(X_train,y_train,d1):

    print("training started")

    for algo, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        with open(model_path+"/"+algo +"Newcount"+ str(d1)+'.pkl', 'wb') as f1:
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
	
def Trainmain(positive_class,negative_class,d1):
	X = []
	y = []
	a = []	
	for i in negative_class:
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"])]
		#a = [i["abstract"],i["authorList"],i["tagList"],i["topics"],i["tonetype"]]
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"]), int(i["numberOfWords"])]
		a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"])]
		X.append(a)
		y.append(0)
	for i in positive_class:
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"])]
		#a = [i["abstract"],i["authorList"],i["tagList"],i["topics"],i["tonetype"]]
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"]), int(i["numberOfWords"])]
		a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"])]
		X.append(a)
		y.append(1)

		
		
	X = numpy.array(X)
	y = numpy.array(y)
	#min_max_scaler = preprocessing.MinMaxScaler()
	#X = min_max_scaler.fit_transform(X)
	
	X = preprocessing.scale(X)
	X = preprocessing.normalize(X, norm='l2')

	#model = LogisticRegression()
	#model = model.fit(X, y)

	#with open(model_path+"/GuardianlogisticregressiondateNew"+str(d1)+'.pkl', 'wb') as f1:
	#	pickle.dump(model, f1)
	
	#print("logistic model trained")
	train_model(X,y,d1)	
	#print(model.score(X,y))
	print("Model has been trained")

	
def main():	

	allFeatureVal = numpy.load('FeatureInValueForm/FeatureInValueFormwithPublishDateOrdered.npy')
	
	print(len(allFeatureVal))
	
	positive_class = []
	negative_class = []
	positive_class_test = []
	negative_class_test = []
	count = 0
	'''
	for data in allFeatureVal:
		if int(data["output"]) == 1:
				positive_class.append(data)
		else:
				negative_class.append(data)
				
		if (len(positive_class)+len(negative_class ))%10000 == 0:
			print(len(positive_class))
			print(len(negative_class))
	'''
	
				
	prev = 0	
	for data in allFeatureVal:
	
		a = data["dateList"]
		d1 = datetime.strptime(a, "%Y-%m-%d")
		if d1.year > 2015:
			print ("Date: ",d1)
			#print data["output"]
					
			if d1 != prev:
				positive_class_forthis = positive_class
				negative_class_forthis = negative_class
				if len(positive_class) > 1.5*len(negative_class):
					positive_class_forthis = random.sample(positive_class, int(1.5*len(negative_class)))
				elif len(negative_class) > 1.5*len(positive_class):
					negative_class_forthis = random.sample(negative_class, int(1.5*len(positive_class)))
				
				print(len(positive_class_test))
				print(len(negative_class_test))
				print(len(positive_class_forthis))
				print(len(negative_class_forthis))
				print(len(positive_class))
				print(len(negative_class))
				
				Trainmain(positive_class_forthis,negative_class_forthis,count)
				
				
				for i in positive_class_test:
					positive_class.append(i)
				for i in negative_class_test:
					negative_class.append(i)	
					
				positive_class_test = []
				negative_class_test = []
					
			
			if int(data["output"]) == 1:
			#	print("here 1")
				positive_class_test.append(data)
			else:
			#	print("here 0")
				negative_class_test.append(data)
			
			if count%1000 == 1:
				print(count)
			count = count+1	
				
			

		else:
			if int(data["output"]) == 1:
				positive_class.append(data)
			else:
				negative_class.append(data)
	
		prev = d1


if __name__ == '__main__':
	
		
		
	main()
	