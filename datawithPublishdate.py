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


def main():	
	allFeatureVal = numpy.load('FeatureInValueForm/FeatureInValueFormwithRelatedArticle.npy')
	print(len(allFeatureVal))
	ConvertToNumAll = []
	
	model = None
	
	with open('myUrlDateArticle.pickle', 'rb') as f1:
		model = pickle.load(f1)
		
	
	for data in allFeatureVal:
		PublishStatus = [] 
		value = model.get(data["url"])
		if value != None:
			print(value)
			PublishStatus = value 
		thistoadd = {"url":data["url"], "abstract":data["abstract"], "authorList":data["authorList"], \
				"tagList":data["tagList"], "dateList":data["dateList"], "productionOffice":data["productionOffice"],\
				"topics": data["topics"], "numberOfWords":data["numberOfWords"], "tonetype": data["tonetype"], "timestamp": data["timestamp"], "relarticle300": data["relarticle300"], "relarticle1500":data["relarticle1500"], "output":data["output"], "PublishStatus": PublishStatus}
			
		ConvertToNumAll.append(thistoadd)
	
	numpy.save('FeatureInValueForm/FeatureInValueFormwithPublishDate.npy',ConvertToNumAll)
	
if __name__ == '__main__':
    main()
	