import numpy as np
from os.path import abspath


class DataHandler():
    URL = ""

    def __init__(self, URL):
        self.URL = abspath(URL)

    def zscore_data(self, mean, std, data):
        return (data-mean)/std

    def zscores(self, data):
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        std = np.where(std != 0, std, 1)
        return mean, std

    def parse_data_no_header(self):
        return np.genfromtxt(
            self.URL, delimiter=",", dtype='f'
        )

    def parse_data(self):
        return np.loadtxt(
            self.URL, delimiter=",", skiprows=1, usecols=[1, 2, 3]
        )

    def shuffle_data(self, data, seed=0):
        np.random.seed(seed)
        # Shuffle Data
        np.random.shuffle(data)
        return data

    def split_data(self, data):
        # Create Arrays for Training vs Validation
        training_index = round(len(data)*2/3)
        train = data[0:training_index]
        validation_index = training_index+1
        validation = data[validation_index:]
        return train, validation

    def getXYFolded(self, data):
        result = list()
        data = np.array(data)
        for X in data:
            result.append(self.getXY(X))
        return result

    def getXY(self, data, xInd, yInd):
        return data[:, xInd:], data[:, yInd]

    def value_changed(val1, val2):
        if val1 != val2:
            print(val1)