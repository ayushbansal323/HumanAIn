	
# split into standard weeks
import numpy as np
from numpy import split
from numpy import array
from pandas import read_csv
import pandas as pd
def error_remove(dataset):
	days = dataset.groupby(dataset.index.date)
	total=[]
	for name , oneday in days:
		if(len(oneday)%48==0):
			#print(len(oneday))
			#print(" ")
		#else:
			for i in oneday.values:
				total.append(i[3])
				#print(i[3])
	return total
		
# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	#print(len(data)%24)
	train, test = data[:-480], data[-480:]
	print(len(train))
	print(len(test))
	# restructure into windows of weekly data
	train = array(split(train, len(train)/24))
	test = array(split(test, len(test)/24))
	return train, test
 
# load the new file
df = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
gk = df.groupby('LCLid')
#print(gk.get_group("MAC000002"))
i=0
for name, dataset in gk:
	
	#print(dataset['DateTime'])
	datas= error_remove(dataset)
	
	#print(datas)
	train, test = split_dataset(np.array(datas))
	# validate train data
	#print(train.shape)
	#print(train[0, 0, 0], train[-1, -1, 0])
	# validate test
	#print(test.shape)
	#print(test[0, 0, 0], test[-1, -1, 0])
