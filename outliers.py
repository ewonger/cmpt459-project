import json
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


# bathrooms: number of bathrooms
# bedrooms: number of bathrooms
# building_id
# created
# description
# display_address
# features: a list of features about this apartment
# latitude
# listing_id
# longitude
# manager_id
# photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
# price: in USD
# street_address
# interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'

def printFeatureTypes():
	for key in data: 
		i = 0
		for s in data[key]:
		
			print(str(key) + "" + str(type(data[key][s])));
			break;


def getMissingValues(data):

	missingValues = dict.fromkeys(data.keys(), 0);
	zeroesAllowed = ['bedrooms', 'bathrooms'] # bedrooms and bathrooms can have 0 as valid values

	for attr in data:
		for l_id in data[attr]:
			if attr in zeroesAllowed:
				break;
			value = data[attr][l_id]
			if isinstance(value, int):
				if value == 0 :
					missingValues[attr] += 1	
			elif isinstance(value, str) :
				if not value or value == '0':
					missingValues[attr] += 1
			elif isinstance(value, float):
				if value == 0.0:
					missingValues[attr] += 1
			elif isinstance(value, list):
				if len(value) == 0:
					missingValues[attr] += 1

	toRemove = [k for k in missingValues if not missingValues[k] > 0]
	for i in toRemove:
		missingValues.pop(i, None)

	return missingValues

def getOutliersSD(data, threshold):
	outliers = dict.fromkeys(data.keys(), 0);

	noOutliersAttr = ['listing_id']
	zeroesAreInvalid = ['latitude', 'longitude'] # zeroes are counted missing values (New York City is 40 lat and 74 long)

	for attr in data:
		for l_id in data[attr]:

			if attr in noOutliersAttr:
				break

			value = data[attr][l_id];
			if isinstance(value, int) or isinstance(value, float):

				d = list(data[attr].values())
				scaled = preprocessing.scale(d)

				for i in range(scaled.size):
					if scaled[i] > threshold or scaled[i] < -threshold:
						invalidOutlier = d[i] == 0 and attr in zeroesAreInvalid
						if not invalidOutlier:
							outliers[attr] += 1
				
			break 

	toRemove = [k for k in outliers if not outliers[k] > 0]
	for i in toRemove:
		outliers.pop(i, None)

	return outliers


def getOutliersIQR(data, percentile = 75):
	outliers = dict.fromkeys(data.keys(), 0);

	noOutliersAttr = ['listing_id']
	zeroesAreInvalid = ['latitude', 'longitude'] # zeroes are counted missing values (New York City is 40 lat and 74 long)

	for attr in data:
		for l_id in data[attr]:

			if attr in noOutliersAttr:
				break

			value = data[attr][l_id];
			if isinstance(value, int) or isinstance(value, float):

				d = list(data[attr].values())
				qlower, qupper = np.percentile(d, 100 - percentile), np.percentile(d, percentile)
				iqr = qupper - qlower
				cutOff = iqr * 1.5
				lower, upper = qlower - cutOff, qupper + cutOff
				outlierValues = [x for x in d if x < lower or x > upper]

				outlierValues = [x for x in outlierValues if not (x == 0 and attr in zeroesAreInvalid)]
				outliers[attr] = len(outlierValues)
				# print(outlierValues)
				
			break 

	toRemove = [k for k in outliers if not outliers[k] > 0]
	for i in toRemove:
		outliers.pop(i, None)

	return outliers

   



def extractDescriptionData(data):
	vectorizer = CountVectorizer()
	values = list(data['description'].values())
	keys = list(data['description'].keys())
	X = vectorizer.fit_transform(values)
	newData = {}
	for i in range(len(keys)):
		newData[keys[i]] = X[i]
	return (newData, vectorizer)


# features extraction:
def extractFeaturesData(data):
	vectorizer = CountVectorizer()
	seperator = " "
	allFeatures = []
	words = list(data['features'].values())
	keys = list(data['description'].keys())


	for v in words:
		featuresJoined = seperator.join(v) 
		allFeatures.append(featuresJoined)


	X = vectorizer.fit_transform(allFeatures)
	newData = {}
	for i in range(len(keys)):
		newData[keys[i]] = X[i]
	return (newData, vectorizer)


def boxPlot(attr, title, ylabel, removeZeros = False):
	d = list(data[attr].values())
	if (removeZeros):
		d = [x for x in d if not x == 0]
	fig = plt.figure();
	plt.boxplot(d)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

if __name__ == '__main__':
	with open('data/train.json') as json_file:
		data = json.load(json_file)

	print(getMissingValues(data))
	print(getOutliersIQR(data))
	# print(extractFeaturesData(data))
	# boxPlot("price", "Prices of apartments", "Price (USD)")
	# boxPlot("bathrooms", "Number of bathrooms of apartments", "# of bathrooms")
	# boxPlot("bedrooms", "Number of bedrooms of apartments", "# of bedrooms")
	# boxPlot("latitude", "Latitude of apartments", "Latitude (degrees)", True)
	# boxPlot("longitude", "Longitude of apartments", "Longitude (degrees)", True)

	d, vec = extractFeaturesData(data)


	# print(list(data['features'].values())[0])




	# d = list(data['description'].values())
	# k = list(data['description'].keys())
	# v = data['description'][k[2]]
	# print(k[2])
	# print(v)
	# print(d[2])
	# print()
	# print()
	# print(x[2])


