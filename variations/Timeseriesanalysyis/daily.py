# daily line plots
from pandas import read_csv
from matplotlib import pyplot
# load the new file
dataset = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
# plot active power for each year
days = [x for x in range(1, 20)]
pyplot.figure()
for i in range(len(days)):
	# prepare subplot
	ax = pyplot.subplot(len(days), 1, i+1)
	# determine the day to plot
	day = '2012-01-' + str(days[i])
	# get all observations for the day
	result = dataset[day]
	# plot the active power for the day
	pyplot.plot(result['KWh'])
	# add a title to the subplot
	pyplot.title(day, y=0, loc='left')
pyplot.show()
