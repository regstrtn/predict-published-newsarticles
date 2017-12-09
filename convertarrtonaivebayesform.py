from sklearn import svm, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pickle, numpy, gzip, sqlite3, random
from datetime import datetime
import sys
import os
reload(sys)
sys.setdefaultencoding("utf8")

model_path = os.path.dirname(os.path.abspath(__file__))+"/TrainedNB"


def predictUsingNaiveBayes(X_train, X_test, Y_train):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, Y_train)
	
    predicted = clf.predict(X_test)
    return predicted,clf


def predictRet(positive_class,negative_class,positive_class_test,negative_class_test,date):
	X_train = []
	Y_train = []
	for i in negative_class:
		X_train.append(i)
		Y_train.append(0)
	for i in positive_class:
		X_train.append(i)
		Y_train.append(1)

	X_train = numpy.array(X_train)
	Y_train = numpy.array(Y_train)
	
	X_test = []
	Y_test = []
	
	for i in negative_class_test:
		X_test.append(i)
		Y_test.append(0)
	for i in positive_class_test:
		X_test.append(i)
		Y_test.append(1)

	X_test = numpy.array(X_test)
	Y_test = numpy.array(Y_test)
	
	urlFeaturesDict = []
	#news_desk_feature_scores= predictUsingNaiveBayes([i["news_desk"] for i in X_train], [i["news_desk"] for i in X_test], Y_train)
	for i in range(len(X_test)):
		featureVal = X_test[i]
		urlFeaturesDict.append(featureVal)
		#urlFeaturesDict.append(X_test[i])
	urlFeaturesDict = numpy.array(urlFeaturesDict) 
	#print(len(urlFeaturesDict))
	return urlFeaturesDict



def main():
	allFeatureText = numpy.load('FeatureInTextForm/featureInTextForm.npy')
	
	positive_class = []
	negative_class = []
	positive_class_test = []
	negative_class_test = []
	count = 1
	ConvertToNumAll = []
	for data in allFeatureText:
		a = data["dateList"]
		d1 = datetime.strptime(a, "%Y-%m-%d")
		if d1.year > 2015 or d1.month> 6:                     												#start of training for SVM
			
			if data["output"] == "1":
				positive_class_test.append(data)
			else:
				negative_class_test.append(data)
			
			
			if len(positive_class_test)+ len(negative_class_test) == 300:
				positive_class_forthis = positive_class
				negative_class_forthis = negative_class
				
				naiveTestElements = predictRet(positive_class_forthis,negative_class_forthis,positive_class_test,negative_class_test,1)
				for j in naiveTestElements:
					ConvertToNumAll.append(j)
				for i in positive_class_test:
					positive_class.append(i)
				for i in negative_class_test:
					negative_class.append(i)	
					
				positive_class_test = []
				negative_class_test = []
					
				
			if count%1000 == 1:
				print(count)
			count = count+1
			
		else:
			if data["output"] == "1":
				positive_class.append(data)
			else:
				negative_class.append(data)
		
	
	numpy.save('FeatureInTextFormSorted/FeatureInTextFormSorted.npy',ConvertToNumAll)	
				
	
if __name__ == '__main__':
    main()