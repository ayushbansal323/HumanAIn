from pandas import read_csv
import csv

import warnings
warnings.filterwarnings("ignore")
def main():	
	# load the Data
	dataset = read_csv('Power-Networks-LCL.csv', header=0, infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
	#group the data by household id
	dataset_group_id = dataset.groupby('LCLid')
	with open('input.csv', 'w') as writeFile:
		writer = csv.writer(writeFile,delimiter=',',dialect='excel',lineterminator='\n')
		headers=["name","date"]
		for i in range(24):
			headers.append((i))
		writer.writerow(headers)
		for name_id,dataset_id in dataset_group_id:
			
			#resample the dataset into hourly
			hourly_datagroup_id = dataset_id.resample('H')
			hourly_dataset_id = hourly_datagroup_id.sum()
			
			days = hourly_dataset_id.groupby(hourly_dataset_id.index.date)
			#removing inconsistance of dataset
			
			for day , oneday in days:
				if(len(oneday)%24==0):
					lines=[]
					lines.append(name_id)
					lines.append(day)
					for i in oneday.values:
						lines.append(i[0])
					writer.writerow(lines)
			
	writeFile.close()
if __name__ == '__main__':
    main()
