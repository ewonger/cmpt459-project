import json
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import csv
import outliers

# improvements : try pruning, try different max_depths, extracting more features

# Selected features
# -----------------
# ---> bathrooms: number of bathrooms , 15 unique values
# ---> bedrooms: number of bathrooms, 9 unique values
# ---> latitude , > 2000 unique
# ---> longitude, > 2000 unique
# ---> price: in USD, > 2000 unique
# ---> extracted features
# ---> extracted description
# ---> extracted grayscale
# ***> interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'


# Remove
# -----------------
# building_id
# created
# description
# display_address
# features: a list of features about this apartment
# listing_id
# manager_id
# photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
# street_address


# def holdOut(data):
# 	trainSet = {}
# 	validSet = {}

# 	for attr in data:
# 		values = data[attr]
# 		numValues = len(values) 
# 		trainSet[attr] = dict(list(values.items())[numValues//2:]);
# 		validSet[attr] = dict(list(values.items())[:numValues//2]);

# 	return trainSet, validSet


def filterFeatures(data, toKeep):
	return {key: data[key] for key in data if key in toKeep}



def getInputAndTargetSamples(data):
	classLabels = data['interest_level'].values();
	numSamples = len(classLabels);

	data.pop('interest_level', None)

	X = getInputSamples(data);


	return X,np.array(list(classLabels))

def getInputSamples(data):

	numSamples = len(list(data.values())[0])

	dataArr = []
	for attr in data:
			dataArr.append(list(data[attr].values()))

	numFeats = len(dataArr);
	X = np.zeros((numSamples, numFeats))

	for i in range(numSamples):
		row = []
		for j in range(numFeats):
			row.append(dataArr[j][i])
		X[i] = row
	return X



def removeMissingValues(data):
	zeroesAllowed = ['bedrooms', 'bathrooms']
	toRemove = set()
	for attr in data:
		for l_id in data[attr]:
			if attr in zeroesAllowed:
				break;
			value = data[attr][l_id]
			if isinstance(value, float):
				if value == 0.0:
					toRemove.add(l_id)

	for attr in data:

		data[attr] = {key: data[attr][key] for key in data[attr] if key not in toRemove}


def crossValidEval(X, y, numSplits, maxDepth = None):
	valLogLosses = []
	valClassificationAccs = []
	trainLogLosses = []
	trainClassificationAccs = []

	kf = KFold(numSplits, False, None);
	for trainIndex, testIndex in kf.split(X):
		XTrain, XValid = X[trainIndex], X[testIndex]
		yTrain, yValid = y[trainIndex], y[testIndex]
		clf = tree.DecisionTreeClassifier(max_depth = maxDepth)
		clf = clf.fit(XTrain,yTrain)


		trainLogLosses.append(log_loss(yTrain, clf.predict_proba(XTrain)))
		trainClassificationAccs.append(clf.score(XTrain, yTrain));

		valLogLosses.append(log_loss(yValid, clf.predict_proba(XValid)))
		valClassificationAccs.append(clf.score(XValid, yValid))

	return np.average(trainLogLosses), np.average(trainClassificationAccs), np.average(valLogLosses), np.average(valClassificationAccs);


def crossValidEvalPruning(X, y, numSplits, maxDepth = None):
	bestAlphas = []
	valLogLosses = []
	valClassificationAccs = []

	kf = KFold(numSplits, False, None);
	for trainIndex, testIndex in kf.split(X):
		XTrain, XValid = X[trainIndex], X[testIndex]
		yTrain, yValid = y[trainIndex], y[testIndex]
		clf = tree.DecisionTreeClassifier(max_depth = maxDepth)
		path = clf.cost_complexity_pruning_path(XTrain, yTrain)
		ccp_alphas = path.ccp_alphas;
		clfs = []
		for ccp_alpha in ccp_alphas:
		    clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha)
		    clf.fit(XTrain, yTrain)
		    clfs.append(clf)

		logLosses = [log_loss(yValid, clf.predict_proba(XValid)) for clf in clfs]
		accs = [clf.score(XValid, yValid) for clf in clfs]
		logIndex = np.argmin(logLosses);
		accIndex = np.argmax(accs);


		bestAlphas.append(ccp_alphas[logIndex])
		valLogLosses.append(logLosses[logIndex])
		valClassificationAccs.append(accs[accIndex])

	print("Val prun log loss: " + str(np.average(valLogLosses)))
	print("Val prun log acc: " + str(np.average(valClassificationAccs)))

	return np.average(bestAlphas);
	


def generatePredictions(XTrain, yTrain, testData, features, maxDepth = None,  ccpAlpha=0.0):
	
	filteredTest = addAndFilterFeatures(testData, True)


	XTest = getInputSamples(filteredTest)
	clf = tree.DecisionTreeClassifier(max_depth = maxDepth, ccp_alpha = ccpAlpha)
	clf = clf.fit(XTrain, yTrain)



	listingIds = np.array(list(testData['listing_id'].values()), dtype=np.int)
	probs = clf.predict_proba(XTest)
	probs[:,[1, 2]] = probs[:,[2, 1]]

	probs = np.array(probs, dtype=np.object)

	values = np.insert(probs, 0, listingIds, axis = 1)


	with open("probs.csv","w+") as my_csv:
	    csvWriter = csv.writer(my_csv,delimiter=',')
	    csvWriter.writerow(['listing_id', 'high', 'medium', 'low'])
	    csvWriter.writerows(values)


def convertExtractedFeature(featureDict):
	newDict = {}
	for key, value in featureDict.items():
		newDict[key] = value.sum()
	return newDict


def addFeatures(data):
	descr = outliers.extractDescriptionData(data)[0];
	data['num_description_words'] = convertExtractedFeature(descr);

	ap_features = outliers.extractFeaturesData(data)[0];
	data['num_features'] = convertExtractedFeature(ap_features);

	photos = data['photos']
	photoDict = {}
	for key, value in photos.items():
		photoDict[key] = len(value)
	data['num_photos'] = photoDict;






def addAndFilterFeatures(data, add = False):
	if add:
		addFeatures(data)
 
	filteredData = filterFeatures(data, features);
	return filteredData


if __name__ == '__main__':
	with open('data/train.json') as json_file:
		data = json.load(json_file)

	with open('data/test.json') as json_file:
		testData = json.load(json_file) 

	features = ['bathrooms', 'bedrooms','longitude', 'latitude', 'price', 'num_description_words', 'num_features', 'num_photos', 'interest_level']
	filteredData = addAndFilterFeatures(data, True);

	# features = ['bathrooms', 'bedrooms','longitude', 'latitude', 'price', 'interest_level']
	# filteredData = addAndFilterFeatures(data, False);

	removeMissingValues(filteredData);

	print(filteredData.keys())

	XTrain,yTrain = getInputAndTargetSamples(filteredData);

	maxDepth = 6;


	# avgLogLossTrain, avgClassAccTrain, avgLogLossVal, avgClassAccVal = crossValidEval(XTrain,yTrain,5, maxDepth)
	# avgLogLossTrain, avgClassAccTrain, avgLogLossVal, avgClassAccVal = crossValidEval(XTrain,yTrain,5)
	# print("Using features: " + " ".join(features))
	# print("Train avg log loss:" + str(avgLogLossTrain))
	# print("Train avg classification accuracy:" + str(avgClassAccTrain))
	# print()
	# print("Val avg log loss:" + str (avgLogLossVal))
	# print("Val avg classification accuracy:" + str(avgClassAccVal))
	# print()



	bestCcp = crossValidEvalPruning(XTrain, yTrain, 5, maxDepth)

	print(bestCcp)

	generatePredictions(XTrain, yTrain, testData, features, maxDepth, bestCcp)


	# for i in range(2, 11) :
	# 	print("max_depth =" + str(i))
	# 	avgLogLossTrain, avgClassAccTrain, avgLogLossVal, avgClassAccVal = crossValidEval(XTrain,yTrain,5, i)
	# 	print("Train avg log loss:" + str(avgLogLossTrain))
	# 	print("Train avg classification accuracy:" + str(avgClassAccTrain))
	# 	print()
	# 	print("Val avg log loss:" + str (avgLogLossVal))
	# 	print("Val avg classification accuracy:" + str(avgClassAccVal))
	# 	print()
