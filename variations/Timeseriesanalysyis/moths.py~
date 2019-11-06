	# monthly line plots
from pandas import read_csv
from matplotlib import pyplot
# load the new file
dataset = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
# plot active power for each year
months = [x for x in range(5, 13)]
pyplot.figure()
for i in range(len(months)):
	# prepare subplot
	ax = pyplot.subplot(len(months), 1, i+1)
	# determine the month to plot
	month = '2012-' + str(months[i])
	# get all observations for the month
	result = dataset[month]
	# plot the active power for the month
	pyplot.plot(result['KWh'])
	# add a title to the subplot
	pyplot.title(month, y=0, loc='left')
pyplot.show()
