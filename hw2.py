from data_handler import DataHandler
from linear_regression import LinearRegression
from log_regression import LogisticalRegression
from calculations import Calculations

import numpy as np

dh = DataHandler("HW2/spambase.data")
data = dh.parse_data_no_header()
data = dh.shuffle_data(data)
data_train, data_validation = dh.split_data(data)
mean, std = dh.zscores(data_train)
data_train = dh.zscore_data(mean, std, data_train)
print(data_train)

data_validation = dh.zscore_data(mean, std, data_validation)

log = LogisticalRegression(dh, data_train, data_validation)

print(log.regression())
