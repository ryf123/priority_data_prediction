import numpy as np
import re
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, average_precision_score

YEAR = 365
MONTH = 30

# data from https://www.immihelp.com/visa-bulletin-tracker/
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

# read data
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

# add month features
month_names = ['Dec','Nov','Oct','Sep','Aug','Jul','Jun','May','Apr','Mar','Feb','Jan']
for month_name in month_names:
	df[month_name] = df['month'].apply(lambda x: month_name in x)

# drop the first/last month
df = df.dropna(axis=0, how='any')

# add if next month advance column
df['advance'] = df['eb3_next_month_advance_days'].apply(lambda x: x > 0)
# features used
features = ['eb2_wait_time', 'eb3_wait_time', 'eb3_current_month_advance_days'] + month_names

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
	predictions = predict_advance_model.predict(test_X)
	print "random forest auc:", get_auc(test_y_advance, predictions)

# single prediction, predict Jan's advance days, given Dec data
single_predict = [1610, 1354, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
print zip(features, single_predict)

status = "process" if predict_advance_model.predict([single_predict]) else "retrogress"
days = str(predict_days_model.predict([single_predict])[0])
print "Next month the priority date will {0} by {1} days".format(status, days)