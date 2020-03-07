import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
def interest_levelSwitch(level):
	switcher = {
		'high': 2,
		'medium': 1,
		'low': 0,
	}
	return switcher.get(level, 0)

trainFile = pd.read_json("train.json")
testFile = pd.read_json("test.json")

trainFile['features'] = trainFile['features'].apply(lambda x: len(x))
trainFile['photos'] = trainFile['photos'].apply(lambda x: len(x))
trainFile = trainFile.rename(columns={'photos':'num_photos', 'features':'num_features'})
trainFile = trainFile.drop(columns=['building_id', 'created', 'description', 'manager_id', 'listing_id', 'display_address', 'street_address'])

testFile['features'] = testFile['features'].apply(lambda x: len(x))
testFile['photos'] = testFile['photos'].apply(lambda x: len(x))
testFile = testFile.rename(columns={'photos':'num_photos', 'features':'num_features'})
testFile = testFile.drop(columns=['building_id', 'created', 'description', 'manager_id', 'listing_id', 'display_address', 'street_address'])

# trainFile = trainFile.drop(columns=['num_features','num_photos','building_id', 'created', 'description', 'manager_id', 'listing_id', 'display_address', 'street_address'])
xTrain = trainFile.iloc[:, 0:7].to_numpy()
yTrain = trainFile.iloc[:, 7].to_numpy()

# kf = KFold()
kf = KFold(n_splits=5)

testLogLosses = []
testClassificationAccuracy = []
trainLogLosses = []
trainClassificationAccuracy = []

for train_index, test_index in kf.split(xTrain):
	print("TRAIN:", train_index, "TEST:", test_index)
	x_train, x_test = xTrain[train_index], xTrain[test_index]
	y_train, y_test = yTrain[train_index], yTrain[test_index]

	x_train = StandardScaler().fit_transform(x_train)
	# x_test = StandardScaler().fit_transform(x_test)
	classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)
	classifier.fit(x_train, y_train)
	
	trainLogLosses.append(log_loss(y_train, classifier.predict_proba(x_train)))
	trainClassificationAccuracy.append(classifier.score(x_train, y_train))
	testLogLosses.append(log_loss(y_test, classifier.predict_proba(x_test)))
	testClassificationAccuracy.append(classifier.score(x_test, y_test))

testLogAvg = np.average(testLogLosses)
testClassAvg = np.average(testClassificationAccuracy)
trainLogAvg = np.average(trainLogLosses)
trainClassAvg = np.average(trainClassificationAccuracy)
print(testLogAvg)
print(testClassAvg)
print(trainLogAvg)
print(trainClassAvg)