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
	allFeatureVal = numpy.load('FeatureInValueForm/FeatureInValueFormwithPublishDate.npy')
	print(len(allFeatureVal))
	ConvertToNumAllPos = []
	ConvertToNumAllNeg = []
	TextSortedOrderedPos = []
	TextSortedOrderedNeg = []
	
	
	relatedarticletonum = numpy.load('FeatureInTextFormSorted/FeatureInTextFormSorted.npy')
	relatedarticletonum = relatedarticletonum[1800:]   
	#Need to do this because of the related article feature. 
	#The first 1800 articles have no value against the 'related_article' feature
	
	print(len(relatedarticletonum))
	
	count = 0
	for i in range(len(allFeatureVal)):
		if allFeatureVal[i]["url"] == relatedarticletonum[i]["url"]:
			count = count +1
		else:
			print("prob")
	print(count)
	
	
	
	for data,tag in zip(allFeatureVal,relatedarticletonum):
		if int(data["output"]) == 1:
			ConvertToNumAllPos.append(data)
			TextSortedOrderedPos.append(tag)
		else:
			ConvertToNumAllNeg.append(data)
			TextSortedOrderedNeg.append(tag)
			
	print(len(ConvertToNumAllPos))
	print(len(TextSortedOrderedPos))
	print(len(ConvertToNumAllNeg))
	print(len(TextSortedOrderedNeg))
	
	ConvertToNumAll = []
	TextSortedOrdered = []
	prev = 0
	for data,tag in zip(ConvertToNumAllNeg, TextSortedOrderedNeg):
		a = data["dateList"]
		d1 = datetime.strptime(a, "%Y-%m-%d")
		if prev == 0:
			prev = d1
		if d1 != prev :
			#print("here")
			previn = prev
			removeThis = []
			removeFromTagList = []
			for data1,tag1 in zip(ConvertToNumAllPos, TextSortedOrderedPos):
				ain = data1["dateList"]
				d1in = datetime.strptime(ain, "%Y-%m-%d")
				if d1in != previn:
					break
				removeThis.append(data1)
				ConvertToNumAll.append(data1)
				removeFromTagList.append(tag1)
				TextSortedOrdered.append(tag1)
				
			for rem in removeThis:
				ConvertToNumAllPos.remove(rem)
			for rem in removeFromTagList:
				TextSortedOrderedPos.remove(rem)
				
		ConvertToNumAll.append(data)	
		TextSortedOrdered.append(tag)		
		prev = d1
	
	ConvertToNumAll = numpy.array(ConvertToNumAll)
	TextSortedOrdered = numpy.array(TextSortedOrdered)
	
	print(len(ConvertToNumAll))
	print(len(TextSortedOrdered))
	
	count = 0
	for i in range(len(ConvertToNumAll)):
		if ConvertToNumAll[i]["url"] == TextSortedOrdered[i]["url"]:
			count = count +1
		else:
			print("prob")
	print(count)
	
	if count == len(TextSortedOrdered):
		numpy.save('FeatureInValueForm/FeatureInValueFormwithPublishDateOrdered.npy',ConvertToNumAll)
		numpy.save('FeatureInTextFormSorted/FeatureInTextFormSortedOrdered.npy',TextSortedOrdered)
	
if __name__ == '__main__':
    main()
	