import numpy as np
from numpy import ma
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA

import warnings
warnings.filterwarnings("ignore")
#remove error
def error_remove(dataset):
	#group the dataset by date
	days = dataset.groupby(dataset.index.date)
	total=[]
	#check for inconsistance for the day
	for name , oneday in days:
			for i in oneday.values:
				total.append(i)
	return total

# arima forecast
def arima_forecast(dataset):
	# converting dataset into series
	dataset_series = dataset
	model = SARIMA(dataset, order=(3, 0, 0), seasonal_order=(3, 0, 0, 7),
                          simple_differencing=True) #ARIMA:(Autoregressive Integrated Moving Average)
	# fit the model
	model_fit = model.filter(model.start_params)
	# make forecast
	forecast = model_fit.predict(start=len(dataset_series), end=len(dataset_series)+23)
	return forecast
			
# convert list of daily data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [day[:, 0] for day in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# split a dataset into train/test sets
def split_dataset(data):
	# split data into days
	train, test = data[:-24], data[-24:]
	# restructure into windows of daily data
	return train, test
	
	
	
def main():	
	# load the Data
	dataset = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
	#group the data by household id
	dataset_group_id = dataset.groupby('LCLid')

	#define list of list to store top electric consumtion householde KWh
	top_3_24hour_kwh = [[0 for x in range(3)] for y in range(24)]
	#define list of list to store top electric consumtion householde names
	top_3_24hour_names = [["1" for x in range(3)] for y in range(24)] 

	for name_id,dataset_id in dataset_group_id:
		
		#split the dataset
		train, test = split_dataset(dataset_id["KWh"])
	
		#define hours
		hours = ['1', '2', '3', '4', '5', '6', '7', '8' , '9' , '10' , '11' , '12' , '13' , '14' , '15' , '16' , '17' , '18' , '19' , '20' , '21' ,'22' ,'23' , '24' ]


		predict_id = arima_forecast(train)

		#print(predict_id)
		#convert data again from logarithm form
		prediction = np.exp(predict_id)
                
		#find the top three household of every hour
		for i in range(0,len(prediction)):
			hour_temp_score = prediction.iloc[i]
			hour_temp_name = name_id
			for j in range(0,3):
				if hour_temp_score > top_3_24hour_kwh[i][j]:
					top_temp_score = top_3_24hour_kwh[i][j]
					top_temp_name = top_3_24hour_names[i][j]
					top_3_24hour_kwh[i][j] = hour_temp_score
					top_3_24hour_names[i][j] = hour_temp_name
					hour_temp_score = top_temp_score
					hour_temp_name = top_temp_name
		#plot the power consumtion of household
		pyplot.plot(hours, prediction, marker='o', label=name_id)
		
		#for checking the accuracy
		mse = mean_squared_error(np.exp(test),prediction)
		print("RMSE(Root Mean Square Error) :"+str(mse))
	#print the household
	for i in range(len(top_3_24hour_kwh)):
		print("Hour : "+str(i+1)+"\n")
		for j in range(0,3):
			print("Rank : "+str(j)+" HouseHold: "+top_3_24hour_names[i][j]+" Kwh: "+str(top_3_24hour_kwh[i][j])+"\n")
	pyplot.legend()
	pyplot.show()
if __name__ == '__main__':
    main()
