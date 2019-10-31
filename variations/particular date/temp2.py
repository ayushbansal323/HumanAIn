import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt

data_1 = pd.read_csv('AirPassengers.csv')
avg= data_1['#Passengers']
avg=list(avg)
print(avg)
res = pd.Series(avg, index=pd.to_datetime(data_1['Month'],format='%Y-%m'))

ts=np.log(res)
ts_diff = ts - ts.shift()
ts_diff.dropna(inplace=True)
print(ts)
r = ARIMA(ts,(2,1,2))
r = r.fit(disp=-1)

pred = r.predict(start='1961-01',end='1970-01')
dates = pd.date_range('1961-01','1970-01',freq='M')

predictions_ARIMA_diff = pd.Series(pred, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts.ix[0])
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plt.plot(res)
#plt.plot(predictions_ARIMA)

#plt.show()

#print (predictions_ARIMA.head())
#print (ts.head())
