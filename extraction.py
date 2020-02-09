import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv

path = 'two-sigma-connect-rental-listing-inquiries/images_sample/images_sample'
files = [f for f in glob.glob(path + "/681*/681*.jpg")]

def find_outlier_range( values ):
    first_quartile = np.percentile(values, 25)
    third_quartile = np.percentile(values, 75)
    
    iqr = third_quartile - first_quartile
    cut_off = 1.5 * iqr
    lower = first_quartile - cut_off
    upper = third_quartile + cut_off

    return lower, upper

def grayscale( files ):
    cc = len(files)
    frequency = {}
    weight = []
    csv_arr = []
    for f in files:
        print(cc)
        cc -= 1
        freq_temp = {}
        weight_temp = []
        img_gray = Image.open(f,'r').convert('L')
        img_gray.show()
        for arr in np.asarray(img_gray):
            for k in arr:
                if (k in frequency):
                    frequency[k] += 1
                else:
                    frequency[k] = 1   
                if (k in freq_temp):
                    freq_temp[k] += 1
                else:
                    freq_temp[k] = 1  

        for i in range(255):
            if (i not in freq_temp):
                freq_temp[i] = 0
            weight_temp.append(freq_temp[i])
        csv_arr.append(weight_temp)

    for i in range(255):
        if (i not in frequency):
            frequency[i] = 0
        weight.append(frequency[i])

    n, b, patches = plt.hist(np.arange(len(weight)), bins=255, weights=weight)
    plt.title('Luminance Histogram')
    plt.xlabel('Brightness Level')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    plt.close()
    with open("grayscale.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(csv_arr)

grayscale(files)
