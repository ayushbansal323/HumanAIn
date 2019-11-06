# daily line plots
from numpy import ma
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
# load the new file
dataset = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
#df = pd.read_csv("Power-Networks-LCL.csv")
#gk = df.groupby('LCLid')
#gk.first()
#dataset= gk.get_group("MAC000002")
# histogram plot for each variable
#for idx, row in dataset.iterrows():
#	dataset[idx,"KWh"]=
pyplot.figure()
#dataset["KWh"]=np.log(dataset["KWh"])
dataset["KWh"]=ma.filled(np.log(ma.masked_equal(dataset["KWh"], 1)), 0)
data=[]
p=0
for i in dataset["KWh"]:
        if(i!=0):
                data.append(i)
        else:
                p=p+1
                if((p%3)==0):
                        data.append(i)
pyplot.title("KWh", y=0)
pyplot.hist(data,bins=100)
#pyplot.title(name, y=0)
#for i in range(len(dataset.columns)):
#	pyplot.subplot(len(dataset.columns), 1, i+1)
#	name = dataset.columns[i]
#	dataset[name].hist(bins=100)
#	pyplot.title(name, y=0)
pyplot.show()

##skewed distributions

##multi-step forecasting
