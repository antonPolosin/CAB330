# Task.1.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # display plots
import seaborn as sns # generates plots
# Task.2.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

# start of script
def data_prep():
	df = pd.read_csv('organics.csv') #read dataset from organic_datamining.csv
	
	print("################## Initial Data #########################")
	df.info() # displays list of information about the dataset
	
	### Task.1.a. What is the proportion of individuals who purchased organic products?
	# percent of people who has purchased organic products from the supermarket at least once
	# in point decimal proportional ratio
	df.groupby('ORGYN').size()/len(df)
	
	### Task.1.b. Did you have to fix any data quality problems? Detail them.
	### Apply imputation method(s) to the variable(s) that need it. List the variables
	### that needed it. Justify your choice of imputation if needed.
	
	# replace all nulls with U for unknown
	df['GENDER'].fillna('U', inplace=True)
	
	# impute for AGE with rounding, change to int later
	df['AGE'].fillna(round(df['AGE'].mean()), inplace=True)
	# AGE from float to int
	#df['AGE'] = df['AGE'].astype(int)
	
	# impute for NGROUP with mode most occuring group
	df['NGROUP'].fillna(df['NGROUP'].mode()[0], inplace=True)
	
	# delete errornous BILL with values < 1
	mask = df['BILL'] < 1
	df.loc[mask, 'BILL'] = np.nan
	# dropping rows in df based on the errornous values in BILL
	df = df[np.isfinite(df['BILL'])]
	
	# impute REGION with unknown
	df['REGION'].fillna('Unknown', inplace=True)
	
	# AFFL has 1085 missing value(nan) and had outlier values up to 34
	mask = df['AFFL'] > 30
	df.loc[mask, 'AFFL'] = np.nan
	# impute for AFFL with rounding, change to int later
	df['AFFL'].fillna(round(df['AFFL'].mean()), inplace=True)
	# AFFL from float to str because it's categorical
	df['AFFL'] = df['AFFL'].astype(str)
	
	# impute LTIME with mode
	df['LTIME'].fillna(df['LTIME'].mode()[0], inplace=True)
	# LTIME from float to int
	df['LTIME'] = df['LTIME'].astype(int)
	
	
	### Task.1.c. What variables did you include in the analysis and what were their roles and
	### measurement level set? Justify your choice.
	### dropped redundant/unnecessary variables information
	df.drop(['CUSTID', 'DOB', 'EDATE', 'AGEGRP1', 'AGEGRP2', 'TV_REG', 'NEIGHBORHOOD', 'LCDATE', 'ORGANICS'], axis=1, inplace=True)
	
	# Task.1.d. What distribution scheme did you use? What “data partitioning allocation” did
	# you set? Explain your selection. (Hint: Take the lead from Week 2 lecture on
	# data distribution)
	
	# batch testing method for distribution scheme with 70% training 30% test data
	
	df = pd.get_dummies(df)
	print("################## Processed Data #########################")
	df.info()
	print("################## Pre-Process Complete#####################")
	return df
	
	
	#######################Task.2.#################################################################################
	
def decision_tree():
	df = data_prep()
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# model build
	model = DecisionTreeClassifier(max_depth=3)
	model.fit(X_train, y_train)
	
	# print Train and Test data accuracy
	print("Train accuracy:", model.score(X_train, y_train))
	print("Train accuracy:", model.score(X_test, y_test))
	
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))
	
	# grab feature importances from the model and feature name from the original X
	importances = model.feature_importances_
	feature_names = X.columns
	
	# sort them out in descending order
	indices = np.argsort(importances)
	indices = np.flip(indices, axis=0)
	
	# limit to 20 features, you can leave this out to print out everything
	indices = indices[:20]
	
	for i in indices:
		print(feature_names[i], ':', importances[i])
	
	dotfile = StringIO()
	export_graphviz(model, out_file=dotfile, feature_names=X.columns)
	graph = pydot.graph_from_dot_data(dotfile.getvalue())
	graph.write_png("week3_dt_viz.png") # saved in the following file
	
