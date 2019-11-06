# line plots
from pandas import read_csv
from matplotlib import pyplot
# load the new file
dataset = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
# line plot for each variable
pyplot.figure()
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
for i in range(len(dataset.columns)):
	pyplot.subplot(len(dataset.columns), 1, i+1)
	name = dataset.columns[i]
	pyplot.plot(dataset[name])
	pyplot.title(name, y=0)
pyplot.show()
