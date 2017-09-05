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
from sklearn.model_selection import GridSearchCV
# Task.3.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
# Task.4.

from sklearn.neural_network import MLPClassifier

# start of script
def data_prep():
	df = pd.read_csv('organics.csv') #read dataset from organic_datamining.csv
	
	print("################## Initial Data #########################")
	df.info() # displays list of information about the dataset
	
	### Task.1.a. What is the proportion of individuals who purchased organic products?
	# percent of people who has purchased organic products from the supermarket at least once
	# in point decimal proportional ratio
	proportion = df.groupby('ORGYN').size()/len(df)
	print(proportion)
	### Task.1.b. Did you have to fix any data quality problems? Detail them.
	### Apply imputation method(s) to the variable(s) that need it. List the variables
	### that needed it. Justify your choice of imputation if needed.
	
	# replace all nulls with U for unknown
	df['GENDER'].fillna('U', inplace=True)
	
	# impute for AGE with rounding, change to int later
	df['AGE'].fillna(round(df['AGE'].mean()), inplace=True)
	# AGE from float to int
	df['AGE'] = df['AGE'].astype(int)
	
	# impute for NGROUP with mode most occuring group
	df['NGROUP'].fillna(df['NGROUP'].mode()[0], inplace=True)
	
	## delete errornous BILL with values < 1
	#mask = df['BILL'] < 1
	#df.loc[mask, 'BILL'] = np.nan
	## delete outliers BILL with values > 15000
	#mask1 = df['BILL'] > 150000
	#df.loc[mask1, 'BILL'] = np.nan
	## dropping rows in df based on the errornous values in BILL
	#df = df[np.isfinite(df['BILL'])]
	
	# impute REGION with unknown
	df['REGION'].fillna('Unknown', inplace=True)
	
	# AFFL has 1085 missing value(nan) and had outlier values up to 34
	mask = df['AFFL'] > 30
	df.loc[mask, 'AFFL'] = np.nan
	mask1 = df['AFFL'] < 1
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
	df.drop(['CUSTID', 'DOB', 'EDATE', 'AGEGRP1', 'AGEGRP2', 'TV_REG', 'NEIGHBORHOOD', 'LCDATE', 'ORGANICS', 'BILL'], axis=1, inplace=True)
	
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
# default decision tree
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
	print("##################### Decision Tree Model ################################")
	print("Train accuracy:", model.score(X_train, y_train))
	print("Test accuracy:", model.score(X_test, y_test))
	
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
		
	# render decision tree diagram
	dotfile = StringIO()
	export_graphviz(model, out_file=dotfile, feature_names=X.columns)
	graph = pydot.graph_from_dot_data(dotfile.getvalue())
	graph.write_png("week3_dt_viz.png") # saved in the following file

# hyperparameters decision tree
def hp_decision_tree():
	df = data_prep()
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# grid search CV
	params = {'criterion': ['gini'],
          'max_depth': range(2, 5),
          'min_samples_leaf': range(40, 61, 5)}
	
	cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=10)
	cv.fit(X_train, y_train)
	
	print("Train accuracy:", cv.score(X_train, y_train))
	print("Test accuracy:", cv.score(X_test, y_test))
	
	# test the best model
	y_pred = cv.predict(X_test)
	print(classification_report(y_test, y_pred))
	
	# print parameters of the best model
	print(cv.best_params_)
	
	analyse_feature_importance(cv.best_estimator_, X.columns, 20)
	visualize_decision_tree(cv.best_estimator_, X.columns, "dm_best_cv.png")

# default logistic regression
def regression():
	df = data_prep()
	
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# scaling input values because of outlier data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)
	
	# build regression model
	model = LogisticRegression()
	model.fit(X_train, y_train)
	
	# print train and test accuracy
	print("##################### Regression Model ################################")
	print("Train accuracy:", model.score(X_train, y_train))
	print("Test accuracy:", model.score(X_test, y_test))
	
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))
	
	# logistic regression model, all of these weights are stored in .coef_ array of the model
	print(model.coef_)
	
	feature_names = X.columns
	coef = model.coef_[0]
	
	# limit to 20 features, you can leave this out to print out everything
	coef = coef[:20]
	
	for i in range(len(coef)):
		print(feature_names[i], ':', coef[i])

# hyperparameters with GridSearchCV	
def hp_regression():
	df = data_prep()
	
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# scaling input values because of outlier data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)
	
	params = {'C': [pow(10, x) for x in range(-6, 4)]}
	
	cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(), cv=10, n_jobs=-1)
	cv.fit(X_train, y_train)
	
	# test the best model
	print("Train accuracy:", cv.score(X_train, y_train))
	print("Test accuracy:", cv.score(X_test, y_test))
	
	y_pred = cv.predict(X_test)
	print(classification_report(y_test, y_pred))
	
	# print parameters of the best model
	print(cv.best_params_)

######################### Dimensionality reduction #####################################
# Recursive feature elimination	
def rf_regression():
	df = data_prep()
	
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# scaling input values because of outlier data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)
	
	rfe = RFECV(estimator = LogisticRegression(), cv=10)
	rfe.fit(X_train, y_train)
	
	print("Original feature set", X_train.shape)
	print("Number of features after elimination", rfe.n_features_)
	
	X_train_sel = rfe.transform(X_train)
	X_test_sel = rfe.transform(X_test)
	
	# grid search CV
	params = {'C': [pow(10, x) for x in range(-6, 4)]}
	
	cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(), cv=10, n_jobs=-1)
	cv.fit(X_train_sel, y_train)
	
	# test the best model
	print("Train accuracy:", cv.score(X_train_sel, y_train))
	print("Test accuracy:", cv.score(X_test_sel, y_test))
	
	y_pred = cv.predict(X_test_sel)
	print(classification_report(y_test, y_pred))
	
	# print parameters of the best model
	print(cv.best_params_)
	
# Principle Component	
def pc_regression():
	df = data_prep()
	
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# scaling input values because of outlier data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)
	
	pca = PCA()
	pca.fit(X_train)
	
	sum_var = 0
	for idx, val in enumerate(pca.explained_variance_ratio_):
		sum_var += val
		if (sum_var >= 0.95):
			print("N components with > 95% variance =", idx+1)
			break
	
	pca = PCA(n_components=66)
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca = pca.transform(X_test)
	
	# grid search CV
	params = {'C': [pow(10, x) for x in range(-6, 4)]}
	
	cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(), cv=10, n_jobs=-1)
	cv.fit(X_train_pca, y_train)
	
	print("Train accuracy:", cv.score(X_train_pca, y_train))
	print("Test accuracy:", cv.score(X_test_pca, y_test))
	
	# test the best model
	y_pred = cv.predict(X_test_pca)
	print(classification_report(y_test, y_pred))
	
	# print parameters of the best model
	print(cv.best_params_)
	
# feature selection model
def fs_regression():
	df = data_prep()
	
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# scaling input values because of outlier data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)r

	params = {'criterion': ['gini', 'entropy'],
			  'max_depth': range(3, 10),
			  'min_samples_leaf': range(20, 200, 20)}

	cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(), cv=10)
	cv.fit(X_train, y_train)
	
	analyse_feature_importance(cv.best_estimator_, X.columns)
	
	selectmodel = SelectFromModel(cv.best_estimator_, prefit=True)
	X_train_sel_model = selectmodel.transform(X_train)
	X_test_sel_model = selectmodel.transform(X_test)

	print(X_train_sel_model.shape)
	
	params = {'C': [pow(10, x) for x in range(-6, 4)]}

	cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(), cv=10, n_jobs=-1)
	cv.fit(X_train_sel_model, y_train)

	print("Train accuracy:", cv.score(X_train_sel_model, y_train))
	print("Test accuracy:", cv.score(X_test_sel_model, y_test))

	# test the best model
	y_pred = cv.predict(X_test_sel_model)
	print(classification_report(y_test, y_pred))

	# print parameters of the best model
	print(cv.best_params_)
	
########################### Task 4 ##############################################
def neural_network():
	df = data_prep()
	
	# split into y as target variable and X as input variable
	y = df['ORGYN']
	X = df.drop(['ORGYN'], axis=1)
	
	# split data into 70% training data and 30% test data
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.7, random_state=42, stratify=y)
	
	# scaling input values because of outlier data
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)
	
	model = MLPClassifier(max_iter=500)
	model.fit(X_train, y_train)
	
	print("Train accuracy:", model.score(X_train, y_train))
	print("Test accuracy:", model.score(X_test, y_test))
	
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))
	
	print(model)
	
########################### Global Function ###########################################
def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])

def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph.write_png(save_name) # saved in the following file
	
	
