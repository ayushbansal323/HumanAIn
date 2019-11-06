# split into standard weeks
import numpy as np
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#remove error
def error_remove(dataset):
	days = dataset.groupby(dataset.index.date)
	total=[]
	for name , oneday in days:
		if(len((oneday)%24)==0):
			print(len(oneday))
			print(" ")
		else:
			for i in oneday.values:
				#print(i)
				total.append(i)
				#print(i[3])
	return total

# arima forecast
def arima_forecast(history):
	# convert history into a univariate series
	series = to_series(history)
	# define the model
	model = ARIMA(series, order=(7,0,0))
	# fit the model
	model_fit = model.fit(disp=False)
	# make forecast
	yhat = model_fit.predict(len(series), len(series)+24)
	return yhat

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
	
# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = model_func(history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores
	
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
		
# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-480], data[-480:]
	print(len(train))
	print(len(test))
	# restructure into windows of weekly data
	train = array(split(train, len(train)/24))
	test = array(split(test, len(test)/24))
	return train, test
# load the new file
df = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
#df = pd.read_csv("Power-Networks-LCL.csv")
gk = df.groupby('LCLid')
datas= gk.get_group("MAC000002")


daily_groups = datas.resample('H')
daily_data = daily_groups.sum()

#print(daily_data.values)
# summarize
print(daily_data.shape)
#print((len(daily_data))%24)
#print(daily_data.head())
# save
#daily_data.to_csv('household_power_consumption_days.csv'+name)
dataset= error_remove(daily_data)
train, test = split_dataset(np.array(dataset))
# validate train data
print(train.shape)
#print(train[0, 0, 0], train[-1, -1, 0])
# validate test
#print(test[0])
#print(test[0, 0, 0], test[-1, -1, 0])
series = to_series(train)
#print(train)
# plots
#pyplot.figure()
#lags = 100
# acf
#axis = pyplot.subplot(2, 1, 1)
#plot_acf(series, ax=axis, lags=lags)
# pacf
#axis = pyplot.subplot(2, 1, 2)
#plot_pacf(series, ax=axis, lags=lags)
# show plot
#pyplot.show()


models = dict()
models['arima'] = arima_forecast
days = ['1', '2', '3', '4', '5', '6', '7', '8' , '9' , '10' , '11' , '12' , '13' , '14' , '15' , '16' , '17' , '18' , '19' , '20' , '21' ,'22' ,'23' , '24' ]
for name, func in models.items():
	# evaluate and get scores
	score, scores = evaluate_model(func, train, test)
	# summarize scores
	summarize_scores(name, score, scores)
	# plot scores
	pyplot.plot(days, scores, marker='o', label=name)
	
# show plot
pyplot.legend()
pyplot.show()
	
