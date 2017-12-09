import numpy as np
import os
import codecs
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy

from datetime import timedelta, date, datetime
import time


from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from sklearn.decomposition import PCA

model_path = os.path.dirname(os.path.abspath(__file__))+"/TrainedClassifiers"
test_path = os.path.dirname(os.path.abspath(__file__))+"/Test/Guardiantestvectors.txt"

from scipy.optimize import minimize
import scipy.stats as stats


'''

def lik(parameters):
	m = parameters[0]
	sigma = parameters[1]
	k = None
	for i in np.arange(0, len(revmle)):
		y_exp = m*revmle + (1-m)*recmle
		L = (len(revmle)/2 * np.log(2 * np.pi) + len(revmle)/2 * np.log(sigma ** 2) + 1 /(2 * sigma ** 2) * sum((ymle - y_exp) ** 2))
		k = L 
	return k

'''	
	
def regressLL(params):
    # Resave the initial parameter guesses
    b0 = params[0]
    sd = params[1]

    # Calculate the predicted values from the initial parameter guesses
    #yPred = revmle*b0 + recmle*(1-b0)
    yPred = [x*b0 + y*(1-b0) for x,y in zip(revmle, recmle)]

    # Calculate the negative log-likelihood as the negative sum of the log of a normal
    # PDF where the observed values are normally distributed around the mean (yPred)
    # with a standard deviation of sd
    logLik = -np.sum( stats.norm.logpdf(ymle, loc=yPred, scale=sd) )

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)



	
	
	
'''
names = ["NN", "SVM", "RBFSVM", "DT",
         "RF", "AdaBoost", "NB", "LinearDiscriminantAnalysis",
         "QuadraticDiscriminantAnalysis"]
'''

names = ["SVM"]
yPredAll = []
yPredWithoutDiversity = []
yTrueAll = []
MAX_DIVERSITY_OPTIMIZE_ITERATIONS = 50
		 
def evaluate(y_true,y_pred):
	return [accuracy_score(y_true, y_pred),
	f1_score(y_true, y_pred, average=None),
	#f1_score(y_true, y_pred, average='micro'),
	#f1_score(y_true, y_pred, average='macro'),
	#f1_score(y_true, y_pred, average='weighted'),
	#log_loss(y_true,y_pred),
	precision_score(y_true, y_pred, average=None),
	recall_score(y_true, y_pred, average=None)]
	
	#roc_auc_score(y_true, y_pred)]
	
def AnswerWithDiversity(k1,tagList,posClassLen):
	
	print("PosClassLen: {0}".format(posClassLen))
	
	NonMainSetArray = []
	MainSetArray = []
	
	
	kret = np.zeros(len(k1))
	#print(len(tagList))
	#print(len(k1))
	
	k2 = []
	for i in k1:
		k2.append(i[1])
	countSmall = 0
	k2 = np.array(k2)
	k3 = k2
	adder = 10
	print("adder: {0}".format(adder))
	#posClassLen = posClassLen+adder
	posClassLen = 60
	
	ind =  k3.argsort()[-posClassLen:][::-1]
	#print ind
	onlyRec = np.zeros(len(k2))
	for i in ind:
		onlyRec[i] = 1
	for k2iter in range(len(k2)):
		if k2iter not in ind:
			word = tagList[k2iter].split('&')
			thistoadd = {"word":word, "score":k2[k2iter], "index":k2iter}	
			NonMainSetArray.append(thistoadd)
	
	LossWordSet = []
	
	lossCal = 0
	#print NonMainSetArray
	
	for indIter in ind:
		word = tagList[indIter].split('&')
		score = 0
		numberAddedat = []
		for wordIter in word:
			flag = 0
			for LossWordSetIter in LossWordSet:
				if wordIter == LossWordSetIter["word"]:
					flag = 1
					
					LossWordSetIter["number"] = LossWordSetIter["number"]+1
					score = score+(k2[indIter]/LossWordSetIter["number"])
					numberAddedat.append(LossWordSetIter)
					
					break
					 
			if flag == 0:
				thistoadd = {"word":wordIter, "number":1}
				LossWordSet.append(thistoadd)
				numberAddedat.append(thistoadd)
				score = score+k2[indIter]
		

		#print ("LossWordSet")
		#print LossWordSet
		#for i in MainSetArray:
			#print ("MainSetArray")
		#	print i
		#	break
			
		lossCal = lossCal + score
		#print ("numberAddedat")
		#print numberAddedat
		TobeaddedinLoop = numberAddedat
		thistoadd = {"word":TobeaddedinLoop, "score":k2[indIter], "index":indIter}
		MainSetArray.append(thistoadd)
	
	#print lossCal
	#print(len(MainSetArray))
	for kappa in range(MAX_DIVERSITY_OPTIMIZE_ITERATIONS):
		flagMain = 0
		#print kappa
		for i in reversed(MainSetArray):
			
			score = 0
			for wordIter in i["word"]:
				for LossWordSetIter in LossWordSet:
					if wordIter["word"] == LossWordSetIter["word"]:
						score = score+ (i["score"]/LossWordSetIter["number"])
						LossWordSetIter["number"] = LossWordSetIter["number"] - 1
				
			for j in NonMainSetArray:
				numberAddedat = []
				score2 = 0
				for wordIter in j["word"]:
					for LossWordSetIter in LossWordSet:
					
						if wordIter == LossWordSetIter["word"]:
							LossWordSetIter["number"] = LossWordSetIter["number"] + 1
							score2 = score2+ (j["score"]/LossWordSetIter["number"])
							numberAddedat.append(LossWordSetIter)
			
				if score2 - score > 0:
					wordset = []
					for wordIter in i["word"]:
						wordset.append(wordIter["word"])
					thistoadd = {"word":wordset, "score":i["score"], "index":i["index"]}	
					NonMainSetArray.append(thistoadd)
					
					thistoadd = {"word":numberAddedat, "score":j["score"], "index":j["index"]}	
					
					MainSetArray.append(thistoadd)
					NonMainSetArray.remove(j)
					MainSetArray.remove(i)
					flagMain = 1
					break
					
				for wordIter in j["word"]:
					for LossWordSetIter in LossWordSet:
						if wordIter == LossWordSetIter["word"]:
							LossWordSetIter["number"] = LossWordSetIter["number"] - 1
				
				
			if flagMain == 1:
				break
				
			for wordIter in i["word"]:
				for LossWordSetIter in LossWordSet:
					if wordIter["word"] == LossWordSetIter["word"]:
						LossWordSetIter["number"] = LossWordSetIter["number"] + 1
					

			
		if flagMain == 0:
			break
	for i in MainSetArray:
		kret[i["index"]] = 1
	
	
	
	#print MainSetArray
	#print LossWordSet
	
	return kret,onlyRec


	
revmle = []
recmle = []
ymle = []	
	
	
def Testmain(positive_class,negative_class,d1,positive_tagList,negative_tagList,todayDate):
	print(d1)
	X_test = []
	y_test = []
	a = []
	tagList = []
	#print (len(positive_tagList))
	#print (len(negative_tagList))
	#print (len(negative_class))
	timestampList = []
	
	timenow = todayDate
	
	posClassLen = len(positive_class)
	for i,i1 in zip(negative_class,negative_tagList):
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"])]
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"]), int(i["numberOfWords"])]
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["relarticle300"]), int(i["relarticle1500"])]
		a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"])]
		X_test.append(a)
		y_test.append(0)
		tagList.append(i1)
		timestampList.append(float(i["timestamp"]))
		
	for i,i1 in zip(positive_class,positive_tagList):
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"])]
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"]), int(i["numberOfWords"])]
		#a = [int(i["abstract"]),int(i["authorList"]),int(i["relarticle300"]), int(i["relarticle1500"])] 
		a = [int(i["abstract"]),int(i["authorList"]),int(i["tagList"]),int(i["topics"]),int(i["tonetype"]), int(i["relarticle300"]), int(i["relarticle1500"])]
		X_test.append(a)
		y_test.append(1)
		tagList.append(i1)
		timestampList.append(float(i["timestamp"]))
		
	#print X_test[0]
	
	X_test = numpy.array(X_test)
	y_test = numpy.array(y_test)
	#min_max_scaler = preprocessing.MinMaxScaler()
	#X_test = min_max_scaler.fit_transform(X_test)	
	X_test = preprocessing.scale(X_test)
	X_test = preprocessing.normalize(X_test, norm='l2')
	#print X_test[0]
	model = None
	
	timestampList = numpy.array(timestampList)
	
	
	for ia in range(len(timestampList)):
		if timenow - timestampList[ia] != 0:
			timestampList[ia] = 1/(timenow - timestampList[ia])
		else:
			print("timenow-timstamp = 0")
		if timestampList[ia] <= 0:
			print("recency is zero")
	
	tmax = numpy.amax(timestampList)
	for ia in range(len(timestampList)):
		timestampList[ia] = timestampList[ia]/tmax

	for name in names:
		#print("Using model: {0}".format(str(d1)))
		with open(model_path+"/"+name +"Newcount"+str(d1)+'.pkl', 'rb') as f1:
			model = pickle.load(f1)
			
		k1 = model.predict_proba(X_test)
		
		print("k1.shape: {0}".format(k1.shape))
		revmleBefore = []
		for ia in range(len(k1)):
			revmleBefore.append(k1[ia][1])
			
		revmle = numpy.array(revmleBefore)
		recmle = timestampList
		ymle = y_test
		
		#print("Revmle shape: ",revmle.shape)
		#print("Recmle shape: ",recmle.shape)
		#print("Ymle shape: ",ymle.shape)
		
		# Make a list of initial parameter guesses (b0, b1, sd)    
		initParams = [0.7, 1]

		# Run the minimizer
		results = minimize(regressLL, initParams, method='nelder-mead')
		#print(results.x)
		alpha = results['x'][0]
		#print(alpha)
		
		for ia in range(len(k1)):
			k1[ia][1] = alpha*k1[ia][1]+(1-alpha)*timestampList[ia]
		
		
		scores_with_diversity,onlyRec = AnswerWithDiversity(k1,tagList,posClassLen)

		yPredAll.extend(scores_with_diversity)			#k includes scores with diversity 
		yTrueAll.extend(y_test)

		'''
		#Uncomment the whole block to get day wise accuracy and other scores
		eval_with_diversity = evaluate(y_test,scores_with_diversity)	 
		eval_with_onlyRec = evaluate(y_test,onlyRec)
		
		print("WithDiversity")
		print(name+"\t"+str(eval_with_diversity))
		print("WithonlyRec")
		print(name+"\t"+str(eval_with_only_recency))
		'''

		k = model.predict(X_test)
		yPredWithoutDiversity.extend(k)

		'''
		#Uncomment the whole block to get day wise accuracy and other scores
		countSmall = 0
		for ka in k:
			if ka == 1:
				countSmall = countSmall + 1

		scores = evaluate(y_test,k)
		#print countSmall
		print("WithoutDiversity")
		print(name+"\t"+str(scores))
		'''
        
	#numpy.save('urlThis'+str(month)+'.npy',urlThis)
	#print(len(urlThis))
		
	
def main():
	allFeatureVal = numpy.load('FeatureInValueForm/FeatureInValueFormwithPublishDateOrdered.npy')
	print(len(allFeatureVal))
	relatedarticletonum = numpy.load('FeatureInTextFormSorted/FeatureInTextFormSortedOrdered.npy')	
	print(len(relatedarticletonum))
	
	count = 0
	for i in range(len(allFeatureVal)):
		if allFeatureVal[i]["url"] == relatedarticletonum[i]["url"]:
			count = count +1
		#else :
			#print("I m here prob")
	print("count: {0}".format(count))	
	
	
	#for i in range(len(relatedarticletonum))
	
	positive_class_test = []
	negative_class_test = []
	count = 0
	posart = 0
	negart = 0
	positive_tagList = []
	negative_tagList = []
	prev = 0
	for data,tagList in zip(allFeatureVal,relatedarticletonum):
		a = data["dateList"].decode('UTF-8')
		d1 = datetime.strptime(a, "%Y-%m-%d")
		if d1.year > 2015:
			
			if d1 != prev and count!=0 :
				
				todayDate = time.mktime(datetime.strptime(a, "%Y-%m-%d").timetuple())
				
				Testmain(positive_class_test,negative_class_test,count,positive_tagList,negative_tagList,todayDate)
					
				posart = posart + len(positive_class_test)
				negart = negart  + len(negative_class_test)
	
				positive_class_test = []
				negative_class_test = []
				positive_tagList = []
				negative_tagList = []			
			
			if int(data["output"]) == 1:
				positive_class_test.append(data)
				positive_tagList.append(tagList["tagList"].decode('UTF-8'))
			else:
				negative_class_test.append(data)
				negative_tagList.append(tagList["tagList"].decode('UTF-8'))
			
			if count%1000 == 1:
				print("Count: {0}".format(count))
			count = count+1	
			prev = d1
			
		#if count > 50000:
		#	break
            
	posart = posart + len(positive_class_test)
	negart = negart  + len(negative_class_test)
	#Testmain(positive_class_test,negative_class_test,len(allFeatureVal))
	print(posart)
	print(negart)
	print(posart+negart)
	evaluation_scores = evaluate(yTrueAll, yPredAll)
	evaluation_scores_without_diversity = evaluate(yTrueAll, yPredWithoutDiversity)
	print("Final evaluation with diversity: {0}".format(evaluation_scores))
	print("Final evaluation without diversity: {0}".format(evaluation_scores_without_diversity))
	
if __name__ == '__main__':
    main()