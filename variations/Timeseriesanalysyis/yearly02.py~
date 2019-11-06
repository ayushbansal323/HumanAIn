# daily line plots
from pandas import read_csv
from matplotlib import pyplot
# load the new file
df = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
#df = pd.read_csv("Power-Networks-LCL.csv")
gk = df.groupby('Acorn')
#gk.first()
#dataset= gk.get_group("MAC000002")
#print(dataset)
months = [x for x in range(10, 13)]
pyplot.figure()
i=0
for name, dataset in gk:
	# prepare subplot
	#dataset= gk.get_group("MAC000002")
	ax = pyplot.subplot(len(gk), 1, i+1)
	# determine the month to plot
	month = 2012
	# get all observations for the month
	result = dataset[str(month)]
	# plot the active power for the month
	pyplot.plot(result['KWh'])
	# add a title to the subplot
	pyplot.title(name, y=0, loc='left')
	i=i+1
pyplot.show()
