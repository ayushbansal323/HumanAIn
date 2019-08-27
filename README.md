# HumanAIn
This repository contains the code of the HumaALn challenge


To run the project type 
# python3 main.py


Information about files:
1. Power-Networks-LCL.csv
2. output.txt contains output of the program
3. household_pridict.png contains the forecast of the electricity 


The following packages are needed to run the program:
1. matplotlib
2. pandas
3. sklearn
4. statsmodels


The implementation Steps are:
1. Input and error removal and log transform:
a. Input is taken from the csv as our data consist of reading of every
half hour and our prediction should be on an hourly basis we
resample our data into hourly format
b. Then there are few data that are missing we either try to fill the
missing data or remove the tuple if it can't be fixed.
c. we should also make log transformation of KWh column to reduce
its skewness.
2. Group data by households:
a. As the dataset consists of multiple households and we have to
predic electric consumption of each household we group the
dataset of each individual household.
3. Train the model and forecast the electric consumption
a. The model we are joining to use is ARIMA(Auto Regressive
Integrated Moving Average) we will git it the historical data to
train the model and then predict the future forecast.
4. Repeat step 3 for all the households.
5. From the predicted data find the top 3 households on hourly basis and
display the output
a. Now as all the model are run and we have predicted the
consumption of every house then we can easily find the top 3
households .
b. Display the output.
