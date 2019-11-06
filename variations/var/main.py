import pandas as pd
from sklearn.metrics import mean_squared_error
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from matplotlib import pyplot
import csv
dataset = read_csv('input.csv', header=0, infer_datetime_format=True, parse_dates=['date'], index_col=['date'])
#group the data by household id
df = dataset.groupby('name')

import warnings
warnings.filterwarnings("ignore")
mse=[]
namelist=[]


pyplot.figure(1)
pyplot.title("Each HouseHold Prediction")
#define list of list to store top electric consumtion householde KWh
top_3_24hour_kwh = [[0 for x in range(3)] for y in range(24)]
#define list of list to store top electric consumtion householde names
top_3_24hour_names = [["1" for x in range(3)] for y in range(24)] 
	
for names,data in df:
		
		hours = ['1', '2', '3', '4', '5', '6', '7', '8' , '9' , '10' , '11' , '12' , '13' , '14' , '15' , '16' , '17' , '18' , '19' , '20' , '21' ,'22' ,'23' , '24' ]
		
		data= data.drop(['name'], axis=1)
		data = data.loc[:, (data != data.iloc[1]).any()]
		print(names)
		namelist.append(names)
		train = data[:-1]
		valid = data[-1:]
		model = VAR(endog=train)
		model.select_order(3)
		model_fit = model.fit(maxlags=3, ic='aic')
		prediction = model_fit.forecast(model_fit.y, steps=len(valid))
		mse.append(mean_squared_error(valid, prediction))
		prediction=prediction[0]
		
				#find the top three household of every hour
		for i in range(0,len(prediction)):
			hour_temp_score = prediction[i]
			hour_temp_name = names
			for j in range(0,3):
				if hour_temp_score > top_3_24hour_kwh[i][j]:
					top_temp_score = top_3_24hour_kwh[i][j]
					top_temp_name = top_3_24hour_names[i][j]
					top_3_24hour_kwh[i][j] = hour_temp_score
					top_3_24hour_names[i][j] = hour_temp_name
					hour_temp_score = top_temp_score
					hour_temp_name = top_temp_name
		pyplot.plot(hours, prediction, marker='o', label=names)
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
pyplot.figure(2)
pyplot.legend()
pyplot.title("Mean Square Error")
pyplot.plot(namelist, mse, marker='o', label="Mean Square Error")
pyplot.show()
	#print(names)
	#print(prediction)
