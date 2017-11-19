import numpy as np
import re
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, average_precision_score

YEAR = 365
MONTH = 30

# data from https://www.immihelp.com/visa-bulletin-tracker/
# for those month that has N/A just use previous month's waiting time plus one month
# predict the advance time of next month give historical data


class Model:
	def __init__(self, model):
		self.model = model

	def train(self, X, y):
		self.model.fit(X, y)

	def predict(self, test_X):
		return self.model.predict(test_X)


def convert_wait_time_to_days(wait_time):
	"""
	convert year/month/day to days
	:param wait_time: 
	:return: 
	"""
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


def get_auc(test_y, predictions):
	"""
	Only used to get auc for binary classification problem
	:param test_y: 
	:param predictions: 
	:return: 
	"""
	return average_precision_score(test_y, predictions)

# classification mapping
classification_map = {True: "process", False: "retrogress"}

# define feature
month_names = ['Dec', 'Nov', 'Oct', 'Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr', 'Mar', 'Feb', 'Jan']

# features used
features = ['eb2_wait_time', 'eb3_wait_time', 'eb3_current_month_advance_days']
features += month_names

# read data
df = pd.read_csv("pd_history.csv")
df['eb2_wait_time'] = df['eb2_wait_time'].apply(lambda x: convert_wait_time_to_days(x))
df['eb3_wait_time'] = df['eb3_wait_time'].apply(lambda x: convert_wait_time_to_days(x))

eb3_current_month_advance_days = []
eb3_wait_time = df['eb3_wait_time'].values

# calculate this month advance days
for i, value in enumerate(eb3_wait_time[:-1]):
	# advance time = (this month waiting time) - (last month waiting time) + (one month)
	eb3_current_month_advance_days.append(eb3_wait_time[i+1] - value + MONTH)
# the last month does not have advance time, no previous month to compare
eb3_current_month_advance_days.append(np.nan)
df['eb3_current_month_advance_days'] = eb3_current_month_advance_days

# calculate next month advance days, slide the array to right by one
eb3_next_month_advance_days = eb3_current_month_advance_days[:-1]
# we don't know the first month's next month advance time yet
eb3_next_month_advance_days = [np.nan] + eb3_next_month_advance_days
df['eb3_next_month_advance_days'] = eb3_next_month_advance_days

# add the feature, longest consecutive increase days
longest_days = 0
long_consecutive_advance_days = []
# loop from back
for eb3_next_month_advance_days in df['eb3_current_month_advance_days'].values[::-1]:
	long_consecutive_advance_days = [longest_days] + long_consecutive_advance_days
	if eb3_next_month_advance_days < 0:
		longest_days = 0
	else:
		longest_days += eb3_next_month_advance_days
df['long_consecutive_advance_days'] = long_consecutive_advance_days

# add month features
for month_name in month_names:
	df[month_name] = df['month'].apply(lambda x: month_name in x)

# future month features, the first element is for the future month, get it before drop na
future_month_features = df[:1].as_matrix(features)

# drop the first/last month
df = df.dropna(axis=0, how='any')

# add if next month advance column
df['advance'] = df['eb3_next_month_advance_days'].apply(lambda x: x > 0)

# cut the last 6 month, cause don't have enough data from long_consecutive_advance_days to compare
# df = df[:-6]

# train/test split
test_months = [12, 9, 6, 3, 1]
for test_month in test_months:
	print "test month:", test_month
	train, test = df[test_month:], df[:test_month]
	train_X, test_X = train.as_matrix(features), test.as_matrix(features)
	train_y_days, test_y_days = train['eb3_next_month_advance_days'], test['eb3_next_month_advance_days']
	train_y_advance, test_y_advance = train['advance'], test['advance']

	# predict advance days using linear regression
	predict_days_model = Model(linear_model.LinearRegression())
	predict_days_model.train(train_X, train_y_days)
	predictions = predict_days_model.predict(test_X)
	print "linear regression r2 score:", r2_score(test_y_days, predictions)

	# predict advance/retrogress using logistic regression
	predict_advance_model = Model(linear_model.LogisticRegression())
	predict_advance_model.train(train_X, train_y_advance)
	predictions = predict_advance_model.predict(test_X)
	print "logistic regression auc:", get_auc(test_y_advance, predictions)

	# predict advance/retrogress using random forest
	predict_advance_model = Model(RandomForestClassifier(n_estimators=10))
	predict_advance_model.train(train_X, train_y_advance)

	# get score if testing is greater than 3, else predict last few months' date
	if test_month > 3:
		predictions = predict_days_model.predict(test_X)
		print "linear regression r2 score:", r2_score(test_y_days, predictions)
		predictions = predict_advance_model.predict(test_X)
		print "logistic regression auc:", get_auc(test_y_advance, predictions)
		predictions = predict_advance_model.predict(test_X)
		print "random forest auc:", get_auc(test_y_advance, predictions)
	else:
		for i, x in enumerate(test_X):
			print zip(features, x)
			status = predict_advance_model.predict([x])[0]
			days = str(predict_days_model.predict([x]))
			print "Next month the priority date will {0} by {1} days".format(classification_map[status], days)
			print "Actual {0} by {1} days.".format(classification_map[test_y_advance.values[i]], test_y_days.values[i])

	# the future one month prediction
	print zip(features, future_month_features[0])
	status = predict_advance_model.predict(future_month_features)[0]
	days = str(predict_days_model.predict(future_month_features))
	print "Next month the priority date will {0} by {1} days".format(classification_map[status], days)
