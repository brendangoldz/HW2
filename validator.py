from scipy import stats

def validate_zscore(train_data):
    # print ("\nZ-score for train_data : \n", stats.zscore(train_data, axis = 1))
    return stats.zscore(train_data, ddof=1)