import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
df = pd.read_json("train.json")


#Part 1.1
def find_outlier_range( values ):
    first_quartile = np.percentile(values, 25) 
    third_quartile = np.percentile(values, 75)
    
    iqr = third_quartile - first_quartile
    cut_off = 1.5 * iqr
    lower = first_quartile - cut_off
    upper = third_quartile + cut_off

    return lower, upper

def plot_price( dataset ):
    lower, upper = find_outlier_range(dataset)

    filtered_dataset = dataset[~((dataset < lower) | (dataset > upper))]
    n, bins, patches = plt.hist(filtered_dataset, bins='auto')
    plt.title('Histogram of Price with Removed Outliers')
    plt.xlabel('Price (USD)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    plt.close()

def plot_latitude( dataset ):
    lower, upper = find_outlier_range(dataset)

    filtered_dataset = dataset[~((dataset < lower) | (dataset > upper))]
    n, bins, patches = plt.hist(filtered_dataset, bins='auto')
    plt.title('Histogram of Latitude with Removed Outliers')
    plt.xlabel('Latitude')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    plt.close()

def plot_longitude( dataset ):
    lower, upper = find_outlier_range(dataset)

    filtered_dataset = dataset[~((dataset < lower) | (dataset > upper))]
    n, bins, patches = plt.hist(filtered_dataset, bins='auto')
    plt.title('Histogram of Longitude with Removed Outliers')
    plt.xlabel('Longitude')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    plt.close()

price = df['price'].to_numpy()
latitude = df['latitude'].to_numpy()
longitude = df['longitude'].to_numpy()

plot_price(price)
plot_latitude(latitude)
plot_longitude(longitude)



#Part 1.2
def plot_hours( dataset ):
    n, bins, patches = plt.hist(dataset, bins=24)
    plt.close()
    plt.plot(n, ls="-", marker="o")

    plt.title('Hour-wise Listing Trend')
    plt.xticks(np.arange(0, 24, 2))
    plt.xlabel('Hours')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    plt.close()

created = df['created'].map(lambda created: datetime.datetime.strptime(created, '%Y-%m-%d %H:%M:%S').time().strftime('%H')).to_numpy().astype(int)

plot_hours(created)



#Part 1.3
def plot_interest_level( dataset ):
    n, bins, patches = plt.hist(dataset, bins=3, density=True )
    print(n)
    plt.close()
    n[0], n[1] = n[1], n[0]
    objects = ('Low', 'Medium', 'High')
    y_pos = np.arange(3)
    plt.bar(y_pos, n, align='center', alpha=0.5)
    plt.title('Proportion of Target Variable Values')
    plt.xticks(y_pos, objects)
    plt.xlabel('Interest Levels')
    plt.ylabel('Normalized Counts')
    plt.grid(True)
    plt.show()
    plt.close()

interest_level = df['interest_level'].to_numpy()
plot_interest_level(interest_level)