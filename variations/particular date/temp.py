from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = read_csv('dataset.csv', header=None)
# seasonal difference
X = series.values
print(X)
days_in_year = 365
differenced = difference(X, days_in_year)
print(differenced)
