import pandas as pd
import numpy as np
# Task.2.
from sklearn.model_selection import train_test_split

# notes
#anything to do with neighborhood group
#df.groupby(['NGROUP'])['NEIGHBORHOOD'].value_counts()
#df.groupby(['NGROUP'])['NEIGHBORHOOD'].describe()
# end of notes

# start of script
def data_prep()
	df = pd.read_csv('organics.csv') #read dataset from organic_datamining.csv
	
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
	df['AGE'] = df['AGE'].astype(int)
	
	# impute for NGROUP with mode most occuring group
	df['NGROUP'].fillna(df['NGROUP'].mode(), inplace=True)
	
	# impute REGION with unknown
	df['REGION'].fillna('Unknown', inplace=True)
	
	# AFFL has 1085 missing value(nan) and had outlier values up to 34
	mask = df['AFFL'] > 30
	df.loc[mask, 'AFFL'] = np.nan
	# impute for AFFL with rounding, change to int later
	df['AFFL'].fillna(round(df['AFFL'].mean()), inplace=True)
	# AFFL from float to int
	df['AFFL'] = df['AFFL'].astype(int)
	
	# impute LTIME with mode
	df['LTIME'].fillna(['LTIME'].mode(), inplace=True)
	# LTIME from float to int
	df['LTIME'] = df['LTIME'].astype(int)
	
	
	### Task.1.c. What variables did you include in the analysis and what were their roles and
	### measurement level set? Justify your choice.
	### dropped redundant/unnecessary variables information
	df.drop(['CUSTID', 'DOB', 'EDATE', 'AGEGRP1', 'AGEGRP2', 'TV_REG', 'NEIGHBORHOOD', 'LCDATE'], axis=1, inplace=True)
	
	# Task.1.d. What distribution scheme did you use? What “data partitioning allocation” did
	# you set? Explain your selection. (Hint: Take the lead from Week 2 lecture on
	# data distribution)
	
	
	
############################################################################################################
#Part 2

#preporcessing steop from previous step, type the following
#df = data_prep()



#split into test and training data
y = df['TargetB']
x = df.drop(['TargetB'], axis=1)







