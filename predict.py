import math
import numpy as np
import re
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

YEAR = 365
MONTH = 30

# data from https://www.immihelp.com/visa-bulletin-tracker/
# predict the advance time of next month give historical data

def convert_wait_time_to_days(wait_time):
	'''
		conver year/month/day to days
	'''
	days = 0
	m = re.search("(\d+) Yrs", wait_time)
	if m:
		days += YEAR * int(m.group(1))

	m = re.search("(\d+) Mths", wait_time)
	if m:
		days += MONTH * int(m.group(1)) 

	m = re.search("(\d+) Days", wait_time)
	if m:
		days += int(m.group(1))

	return days

df = pd.read_csv("pd_history.csv")
df['eb2_wait_time'] = df['eb2_wait_time'].apply(lambda x: convert_wait_time_to_days(x))
df['eb3_wait_time'] = df['eb3_wait_time'].apply(lambda x: convert_wait_time_to_days(x))

eb3_current_month_advance_days = []
eb3_wait_time = df['eb3_wait_time'].values

# calculate this month advance days
for i, value in enumerate(eb3_wait_time[:-1]):
	eb3_current_month_advance_days.append(eb3_wait_time[i+1] - value + MONTH)
# the last month does not have advance time
eb3_current_month_advance_days.append(np.nan)
df['eb3_current_month_advance_days'] = eb3_current_month_advance_days

# calculate next month advance days
eb3_next_month_advance_days = eb3_current_month_advance_days[:-1]
# we don't know the first month's next month advance time yet
eb3_next_month_advance_days = [np.nan] + eb3_next_month_advance_days
df['eb3_next_month_advance_days'] = eb3_next_month_advance_days

print df
# add month features
month_names = ['Dec','Nov','Oct','Sep','Aug','Jul','Jun','May','Apr','Mar','Feb','Jan']
for month_name in month_names:
	df[month_name] = df['month'].apply(lambda x: month_name in x)

# drop the first/last month
df = df.dropna(axis=0, how='any')

# normalize the value
for column in ['eb3_current_month_advance_days', 'eb3_wait_time', 'eb2_wait_time']:
	df[column] = (df[column] - df[column].mean()) / (df[column].max() - df[column].min())

# features used
features = ['eb2_wait_time', 'eb3_wait_time', 'eb3_current_month_advance_days'] + month_names

# train/test split
split_months = 6
train, test = df[split_months:], df[:split_months]
train_X, test_X = train.as_matrix(features), test.as_matrix(features)
train_y, test_y = train['eb3_next_month_advance_days'], test['eb3_next_month_advance_days']


def predict(X, y, test_X, test_y):
	reg = linear_model.LinearRegression()
	reg.fit(X,y)
	predictions = reg.predict(test_X)
	print zip(predictions, test_y)
	return predictions

print df.values
predictions = predict(train_X, train_y, test_X, test_y)
print r2_score(test_y, predictions)
