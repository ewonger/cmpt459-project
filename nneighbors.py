import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

trainFile = pd.read_json("train.json")
testFile = pd.read_json("test.json")

trainFile = trainFile.drop(columns=['building_id', 'created', 'description', 'manager_id', 'listing_id', 'display_address', 'street_address', 'photos', 'features'])

#Improvement
# trainFile['features'] = trainFile['features'].apply(lambda x: len(x))
# trainFile['photos'] = trainFile['photos'].apply(lambda x: len(x))
# trainFile = trainFile.rename(columns={'photos':'num_photos', 'features':'num_features'})
# trainFile = trainFile.drop(columns=['building_id', 'created', 'description', 'manager_id', 'listing_id', 'display_address', 'street_address'])
# testFile['features'] = testFile['features'].apply(lambda x: len(x))
# testFile['photos'] = testFile['photos'].apply(lambda x: len(x))
# testFile = testFile.rename(columns={'photos':'num_photos', 'features':'num_features'})
# testFile = testFile.drop(columns=['building_id', 'created', 'description', 'manager_id', 'listing_id', 'display_address', 'street_address'])

xTrain = trainFile.iloc[:, 0:5].to_numpy()
yTrain = trainFile.iloc[:, 5].to_numpy()
print(trainFile)

#Improvement
# xTrain = trainFile.iloc[:, 0:7].to_numpy()
# yTrain = trainFile.iloc[:, 7].to_numpy()

kf = KFold()

testLogLosses = []
testClassificationAccuracy = []
trainLogLosses = []
trainClassificationAccuracy = []

for train_index, test_index in kf.split(xTrain):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = xTrain[train_index], xTrain[test_index]
    y_train, y_test = yTrain[train_index], yTrain[test_index]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=36)
    # classifier = KNeighborsClassifier(n_neighbors=36)
    classifier.fit(x_train, y_train)

    #    # Calculating error for K values between 1 and 40 and finding best K value
    # error = []
    # for i in range(1, 40):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(x_train, y_train)
    #     pred_i = knn.predict(testFile)
    #     error.append(np.mean(pred_i != y_test))

    # probs = classifier.predict_proba(x_test)
    # probs[:,[1, 2]] = probs[:,[2, 1]]
    # probs = np.array(probs, dtype=np.object)
    # print(probs)
    # print(probs)
    # y_pred = classifier.predict(x_test)
    # print(classification_report(y_test, y_pred))
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