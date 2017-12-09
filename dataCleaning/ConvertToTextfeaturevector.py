from sklearn import svm, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pickle, numpy, gzip, sqlite3, random
import unicodedata
import sys
reload(sys)
sys.setdefaultencoding("utf8")

	
def furtherCleaning():
	abstractList = None
	authorList = None
	topicsList = None
	productionOfficeList = None
	#bodyList = None
	toneTypeList = None
	tagList = None
	numberOfWordsList = None
	outputList = None
	urlList = None
	dateList = None
	timestampList = None
	with open('CleanedData/trailtext.txt') as f:
		abstractList = f.read().splitlines()
		abstractList = numpy.array(abstractList)
	with open('CleanedData/author.txt') as f:
		authorList = f.read().splitlines()
		authorList = numpy.array(authorList)
	with open('CleanedData/topics.txt') as f:
		topicsList = f.read().splitlines()
		topicsList = numpy.array(topicsList)
	with open('CleanedData/productionOffice.txt') as f:
		productionOfficeList = f.read().splitlines()
		productionOfficeList = numpy.array(productionOfficeList)
	#with open('CleanedData/body.txt') as f:
	#	bodyList = f.read().splitlines()
	#	bodyList = numpy.array(bodyList)
	with open('CleanedData/toneType.txt') as f:
		toneTypeList = f.read().splitlines()
		toneTypeList = numpy.array(toneTypeList)
	with open('CleanedData/tagList.txt') as f:
		tagList = f.read().splitlines()
		tagList = numpy.array(tagList)
	with open('CleanedData/numberOfWords.txt') as f:
		numberOfWordsList = f.read().splitlines()
		numberOfWordsList = numpy.array(numberOfWordsList)
	with open('CleanedData/output.txt') as f:
		outputList = f.read().splitlines()
	with open('CleanedData/url.txt') as f:
		urlList = f.read().splitlines()
		urlList = numpy.array(urlList)
	with open('CleanedData/date.txt') as f:
		dateList = f.read().splitlines()
		dateList = numpy.array(dateList)
	with open('CleanedData/timestamp.txt') as f:
		timestampList = f.read().splitlines()
		timestampList = numpy.array(timestampList)
	
	i = 0
	j = 1
	featureInTextForm = []
	
	for out in outputList:
		featureOne = {"url":urlList[i], "abstract":abstractList[i], "authorList":authorList[i], \
                    "tagList":tagList[i], "dateList":dateList[i], "productionOffice":productionOfficeList[i],\
                    "topics":topicsList[i], "numberOfWords":numberOfWordsList[i], "tonetype":toneTypeList[i], "timestamp": timestampList[i] ,"output":out}
		featureInTextForm.append(featureOne)
		i = i+1
	print(len(outputList))
	print(len(urlList))
	print(len(timestampList))
	print(len(featureInTextForm))
	#pickle.dump(featureInTextForm,'featureTextForm/featureInTextForm.txt')
	#numpy.save('featureTextForm/featureInTextForm.npy',featureInTextForm)
	numpy.save('featureInTextForm.npy',featureInTextForm)
	
	
def main():
	abstractList0 = None
	abstractList1 = None
	authorList0 = None
	authorList1 = None
	topicsList0 = None
	topicsList1 = None
	productionOfficeList0 = None
	productionOfficeList1 = None
	toneTypeList0 = None
	toneTypeList1 = None
	tagList0 = None
	tagList1 = None
	numberOfWordsList0 = None
	numberOfWordsList1 = None
	
	
	with open('CleanedData/0trailtext.txt') as f:
		abstractList0 = f.read().splitlines()
		abstractList0 = numpy.array(abstractList0)
	with open('CleanedData/1trailtext.txt') as f:
		abstractList1 = f.read().splitlines()
		abstractList1 = numpy.array(abstractList1)
	with open('CleanedData/0author.txt') as f:
		authorList0 = f.read().splitlines()
		authorList0 = numpy.array(authorList0)
	with open('CleanedData/1author.txt') as f:
		authorList1 = f.read().splitlines()
		authorList1 = numpy.array(authorList1)
	with open('CleanedData/0topics.txt') as f:
		topicsList0 = f.read().splitlines()
		topicsList0 = numpy.array(topicsList0)
	with open('CleanedData/1topics.txt') as f:
		topicsList1 = f.read().splitlines()
		topicsList1 = numpy.array(topicsList1)
	with open('CleanedData/0productionOffice.txt') as f:
		productionOfficeList0 = f.read().splitlines()
		productionOfficeList0 = numpy.array(productionOfficeList0)
	with open('CleanedData/1productionOffice.txt') as f:
		productionOfficeList1 = f.read().splitlines()
		productionOfficeList1 = numpy.array(productionOfficeList1)
	#with open('CleanedData/body.txt') as f:
	#	bodyList = f.read().splitlines()
	#	bodyList = numpy.array(bodyList)
	with open('CleanedData/0toneType.txt') as f:
		toneTypeList0 = f.read().splitlines()
		toneTypeList0 = numpy.array(toneTypeList0)
	with open('CleanedData/1toneType.txt') as f:
		toneTypeList1 = f.read().splitlines()
		toneTypeList1 = numpy.array(toneTypeList1)
	with open('CleanedData/0tagList.txt') as f:
		tagList0 = f.read().splitlines()
		tagList0 = numpy.array(tagList0)
	with open('CleanedData/1tagList.txt') as f:
		tagList1 = f.read().splitlines()
		tagList1 = numpy.array(tagList1)
	with open('CleanedData/0numberOfWords.txt') as f:
		numberOfWordsList0 = f.read().splitlines()
		numberOfWordsList0 = numpy.array(numberOfWordsList0)
	with open('CleanedData/1numberOfWords.txt') as f:
		numberOfWordsList1 = f.read().splitlines()
		numberOfWordsList1 = numpy.array(numberOfWordsList1)
	
	print(tagList1[0].replace ("&", " "))
	
	X_naivetrain = []
	Y_naivetrain = []
	X_trainabstract = []
	Y_trainabstract = []
	X_testabstract = []
	Y_testabstract = []
	num = int(len(abstractList0)/5000)
	print num
	num2 = int(len(abstractList1)/5000)
	print num2
	j = 0
	for i in range(len(abstractList0)):
		if i%num == 0:
			#X_naivetrain.append(abstractList0[i])
			X_naivetrain.append(float(numberOfWordsList0[i]))
			Y_naivetrain.append(0)
		else:
			#X_testabstract.append(abstractList0[i])
			X_testabstract.append(float(numberOfWordsList0[i]))
			Y_testabstract.append(0)
	
	for i in range(len(abstractList1)):
		if i%num2 == 0:
			X_naivetrain.append(float(numberOfWordsList1[i]))
			Y_naivetrain.append(1)
		else:
			X_testabstract.append(float(numberOfWordsList1[i]))
			Y_testabstract.append(1)
	
	X_naivetrain = numpy.array(X_naivetrain)
	Y_naivetrain = numpy.array(Y_naivetrain)
	X_testabstract = numpy.array(X_testabstract)
	Y_testabstract = numpy.array(Y_testabstract)
	#abstract_feature_scores = predictUsingNaiveBayes(X_naivetrain, X_testabstract, Y_naivetrain)
	abstract_feature_scores = X_testabstract
	abstract_feature_scores = numpy.array(abstract_feature_scores)
	#j = 0
	print(len(Y_testabstract))
	print(len(abstract_feature_scores))
	print(abstract_feature_scores[0])
	numpy.savetxt('featureValues/output.txt',Y_testabstract)
	numpy.savetxt('featureValues/wordcount_feature_scores.txt',abstract_feature_scores)
	#for i in range(len(Y_testabstract)):
	#	if abstract_feature_scores[i] == Y_testabstract[i]:
	#		j = j+1
	#print j
	#print(float(j)/len(Y_testabstract))
	#print(abstract_feature_scores)
	print(abstract_feature_scores[0])
	#print(abstract_feature_scores[0][1])
	
if __name__ == "__main__":
	furtherCleaning()
    #main()
	#a = numpy.load('featureTextForm/featureInTextForm.npy')
	#print(len(a))
	#print(a[0]['tagList'])
	#abstract_feature_scores = numpy.loadtxt('featureValues/abstract_feature_scores.txt')
	#print(len(abstract_feature_scores))
	#print(abstract_feature_scores[0][0])
	#print(abstract_feature_scores[0][1])