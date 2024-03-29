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
import csv


import warnings
warnings.filterwarnings("ignore")
#remove error
def error_remove(dataset):
	#group the dataset by date
	days = dataset.groupby(dataset.index.date)
	total=[[] for i in range(24)]
	#check for inconsistance for the day
	for name , oneday in days:
		if(len(oneday)%24==0):
			pos=0
			for i in oneday.values:
				total[pos].append(i)
				pos=pos+1
	return total

# arima forecast
def arima_forecast(dataset_series):
	# define the model
	model = ARIMA(dataset_series, order=(1,0,0)) #ARIMA:(Autoregressive Integrated Moving Average)
	# fit the model
	model_fit = model.fit(disp=False)
	# make forecast
	forecast = model_fit.predict(len(dataset_series), len(dataset_series))
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
	train, test = data[1:-6], data[-6:]
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

	mse=[]
	pyplot.figure(1)
	names=[]
	pyplot.title("Each HouseHold Prediction")
	
	for name_id,dataset_id in dataset_group_id:
		names.append(name_id)
		#resample the dataset into hourly
		hourly_datagroup_id = dataset_id.resample('H')
		hourly_dataset_id = hourly_datagroup_id.sum()
		
		#taking logarithm to remove skrewness
		hourly_dataset_id["KWh"]= ma.filled(np.log(ma.masked_equal(hourly_dataset_id["KWh"], 0)), 0)
		
		
		#removing inconsistance of dataset
		filtered_dataset = error_remove(hourly_dataset_id)
		
		#define hours
		hours = ['1', '2', '3', '4', '5', '6', '7', '8' , '9' , '10' , '11' , '12' , '13' , '14' , '15' , '16' , '17' , '18' , '19' , '20' , '21' ,'22' ,'23' , '24' ]
		
		
		prediction=[]
		test=[]

		for i in range(len(filtered_dataset)):
			
			#split the dataset
			train, tests = split_dataset(np.array(filtered_dataset[i]))

			
			predict_id = arima_forecast(train)
			#print(train)
			#break
			#print("predicted: "+str(predict_id[0])+"  actual:"+str(tests[0][0]))
		#convert data again from logarithm form
			prediction.append(np.exp(predict_id[0]))
			test.append(tests[0])

		#print(prediction)
		#find the top three household of every hour
		for i in range(0,len(prediction)):
			hour_temp_score = prediction[i]
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
		mse.append(mean_squared_error(np.exp(test), prediction))
		#print("RMSE(Root Mean Square Error) :"+str(mse))
	#print the household
	lines=[]
	lines.append(['Hour','Top 1','Top 1 Consumtion','Top 2','Top 2 Consumtion','Top 3','Top 3 Consumtion'])
	for i in range(len(top_3_24hour_kwh)):
		line=[]
		line.append(str(i)+":00")
		print("Hour : "+str(i+1)+"\n")
		for j in range(0,3):
			line.append(top_3_24hour_names[i][j])
			line.append(top_3_24hour_kwh[i][j])
			print("Rank : "+str(j)+" HouseHold: "+top_3_24hour_names[i][j]+" Kwh: "+str(top_3_24hour_kwh[i][j])+"\n")
		lines.append(line)
	#print in csv
	with open('output.csv', 'w') as writeFile:
	        writer = csv.writer(writeFile,delimiter=',',dialect='excel',lineterminator='\n')
	        writer.writerows(lines)
	writeFile.close()
	pyplot.legend()
	pyplot.figure(2)
	pyplot.title("Root Mean Square Error")
	pyplot.plot(names, mse, marker='o', label="Root Mean Square Error")
	pyplot.show()
if __name__ == '__main__':
    main()
