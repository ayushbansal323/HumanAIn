# resample minute data to total for each day
from pandas import read_csv
# load the new file
df = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
#df = pd.read_csv("Power-Networks-LCL.csv")
gk = df.groupby('LCLid')
for name, dataset in gk:
	daily_groups = dataset.resample('H')
	daily_data = daily_groups.sum()
# summarize
#print(daily_data.shape)
#print(daily_data.head())
# save
#daily_data.to_csv('household_power_consumption_days.csv'+name)
	
