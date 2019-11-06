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
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
import csv
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA

import warnings
warnings.filterwarnings("ignore")
#remove error
def error_remove(dataset):
	#group the dataset by date
	days = dataset.groupby(dataset.index.date)
	total=[[] for i in range(24)]
	totalnames=[[] for i in range(24)]
	#check for inconsistance for the day
	for name , oneday in days:
		#print(name)
		if(len(oneday)%24==0):
			pos=0
			for i in oneday.values:
				total[pos].append(i[0])
				totalnames[pos].append(name)
				pos=pos+1
	return total,totalnames

# arima forecast
def arima_forecast(dataset_series,dateset,start_index,end_index):
	# define the model
	#dataset_series.to_csv('train.csv', index=False)
	#print(dataset_series)
	#autocorrelation_plot(dataset_series)
	#print(res)
	res = pd.Series(dataset_series, index=dateset)
	model = SARIMA(res, order=(3, 0, 0), seasonal_order=(3, 0, 0,7),
                          simple_differencing=True) #ARIMA:(Autoregressive Integrated Moving Average)
	# fit the model
	model_fit = model.filter(model.start_params)
	# make forecast
	#print(model_fit.summary())
	forecast= model_fit.predict(end=end_index)
	#forecast = model_fit.predict(len(dataset_series), len(dataset_series))
	return forecast[end_index:end_index]
			
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

	
	start_index = '2014-01-25'
	end_index = '2014-01-26'

	print("Enter date to predict")

	start_index=input()
	end_index=start_index
	pred={}
	for name_id,dataset_id in dataset_group_id:
		names.append(name_id)
		#print(name_id)
		
		#resample the dataset into hourly
		hourly_datagroup_id = dataset_id.resample('H')
		hourly_dataset_id = hourly_datagroup_id.sum()
		
		#taking logarithm to remove skrewness
		hourly_dataset_id["KWh"]= ma.filled(np.log(ma.masked_equal(hourly_dataset_id["KWh"], 0)), 0)
		
		
		#removing inconsistance of dataset
		filtered_dataset,dataset_date = error_remove(hourly_dataset_id)
		
		#define hours
		hours = ['1', '2', '3', '4', '5', '6', '7', '8' , '9' , '10' , '11' , '12' , '13' , '14' , '15' , '16' , '17' , '18' , '19' , '20' , '21' ,'22' ,'23' , '24' ]
		
		
		prediction=[[] for i in range(24)]

		for i in range(len(filtered_dataset)):
			

			
			predict_id = arima_forecast(np.array(filtered_dataset[i]),dataset_date[i],start_index,end_index)
			#print(train)
			#break
			#print("predicted: "+str(predict_id[0])+"  actual:"+str(tests[0][0]))
		#convert data again from logarithm form
			prediction[i].append(np.exp(predict_id))
			#print(np.exp(predict_id))
		pred[name_id]=prediction
		
		

        
	#print in csv
	with open('pred.csv', 'w') as writeFile:
	        header=["time"]
	        for name in pred:
	                header.append(name)
	        writer = csv.writer(writeFile,delimiter=',',dialect='excel',lineterminator='\n')
	        writer.writerow(header)
	        for i in range(24):
	                row=[end_index+" "+str(i)+":00:00"]
	                for name in pred:
	                        row.append(pred[name][i][0][0])
	                        #print(end_index+str(i)+" : "+name+" : "+str(pred[name][i][0][0]))
	                        #writer.writerow([name,end_index+" "+str(i)+":00:00",pred[name][i][0][0]])
	                writer.writerow(row)
	        #for date in pred[list(pred)[0]][0]: 
	        #        print(date)
	        #        print(df)
	        #        print("hola")
	                #for i in range(len(pred[list(pred)[0]])):
	                #       for name in pred:
	                #               print(name+" "+str(date)+" \n"+str(i))
                        
	        #writer.writerows(lines)
	writeFile.close()

if __name__ == '__main__':
    main()
