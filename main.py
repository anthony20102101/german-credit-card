import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:',device)



def categorical_processing(df, relevant_columns):

	for column_name in relevant_columns:
		current_column = df[column_name].values

		converted_column = []

		values_seen = {}

		for value in current_column:

			if not value in values_seen.keys():
				values_seen[value] = len(values_seen.keys())

			value = values_seen[value]
			converted_column.append(value)

		df[column_name] = converted_column

	return df


def min_max_scale(df, relevant_columns):

	for column_name in relevant_columns:

		current_column = df[column_name].values

		converted_column = []

		max_value = max(current_column)
		min_value = min(current_column)

		for value in current_column:

			normalized_value = (value - min_value)/(max_value - min_value)

			converted_column.append(normalized_value)

		df[column_name] = converted_column

	return df

class credicard_dataset(Dataset):

	def __init__(self,X_data, y_target):
		self.X = X_data
		self.y = y_target - 1

	def __getitem__(self, idx):

		x = self.X[idx]
		y = self.y[idx]

		return x,y
	def __len__(self):
		return len(self.y)


class Model(nn.Module):
	#Best accuracy 0.779

	def __init__(self):
		super(Model, self).__init__()

		self.fc1 = nn.Linear(20, 64)
		self.fc2 = nn.Linear(64,64)
		self.fc3 = nn.Linear(64,2)

	def forward(self,x):

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		#x = F.softmax(self.fc3(x))
		x = self.fc3(x)

		return x

class Model2(nn.Module):


	def __init__(self, p=0.5):
		super(Model2, self).__init__()

		self.fc1 = nn.Linear(20, 64)
		self.fc2 = nn.Linear(64,64)
		self.fc3 = nn.Linear(64,2)

		self.drop = nn.Dropout(p)

	def forward(self,x):

		x = self.drop(F.relu(self.fc1(x)))
		x = self.drop(F.relu(self.fc2(x)))
		x = self.fc3(x)

		return x





df = pd.read_csv("german_credit_data_dataset.csv")

catergorical_columns = ['checking_account_status','credit_history', 'purpose',
       'savings', 'present_employment', 'personal', 'other_debtors', 'property',
       'other_installment_plans', 'housing','job', 'telephone', 'foreign_worker']

continous_columns = ["credit_amount", "age"]

df = categorical_processing(df, catergorical_columns)
df = min_max_scale(df, continous_columns)

y = df["customer_type"].values

del df["customer_type"]
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

train_dataset = credicard_dataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

test_dataset = credicard_dataset(X_test, y_test)
test_loader = DataLoader(test_dataset,batch_size = 16, shuffle = True)


clf = RandomForestClassifier(random_state = 10, n_estimators = 10)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))


clf = SVC()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))



md = Model2()
#optimizer = optim.SGD(md.parameters(), lr=0.01, momentum = 0.9)
optimizer = optim.Adam(md.parameters())

criterion = nn.CrossEntropyLoss()

loss_history = []
acc_history =  []

for _x in range(50):

	total_loss = 0.0

	for (X,y) in train_loader:

		optimizer.zero_grad()

		output = md(X.float())
		loss = criterion(output, y)

		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	loss_history.append(total_loss)

	correct = 0
	total = 0

	one = 0
	zero = 0

	md.eval()

	with torch.no_grad():

		for i,(data_curr, label_curr) in enumerate(test_loader,0):


			output = md(data_curr.float())
			_, predicted = torch.max(output,1)
			for x in predicted:

				if x ==1:
					one += 1
				else:
					zero += 1

			total += label_curr.size(0)
			correct += (predicted == label_curr).sum().item()



	acc = correct/total
	acc_history.append(acc)


	#Early stopping
	if len(acc_history) > 35:
		if acc - acc_history[-2] < 0:
			break

	md.train()

print(acc_history)
print(loss_history, end="\n\n\n")

print(max(acc_history))
print(min(acc_history))
print("epochs:", _x)
