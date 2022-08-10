'''
Logistic regression model Pima Diabetes dataset 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import random 

def load_data():
	'''
	Load data and give columns helpful names
	'''
	df = pd.read_csv('pima.csv',header=None)
	df.columns = ['preg','glucouse','bloodpressure','skinthickness','insulin','bmi','dbf','age','diabetes']
	scaler = MinMaxScaler()
	df.iloc[:,0:8] = scaler.fit_transform(df.iloc[:,0:8]) 
	return df

def logistic_regr_model(data_frame,expnum,modeltype):
	'''
	split the data and initialise a linear regression model that is either regular, ridge or lasso regression
	calculate auc_roc score, f1 score accuracy score, plot roc curve, cross validate
	'''
	X = data_frame.iloc[:,0:8].to_numpy()
	Y = data_frame.iloc[:,8].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.4,random_state=expnum)
	if modeltype == 0:
		lr = linear_model.LogisticRegression(penalty='none')
	if modeltype == 1:
		lr = linear_model.LogisticRegression(penalty='l1')
	if modeltype == 2:
		lr = linear_model.LogisticRegression(penalty='l2')
	lr.fit(xtrain,ytrain) # fit the model
	ypred = lr.predict_proba(xtest) # predict the probabilites of the values
	ypredfinal = np.around(ypred[:,1]) # round the values
	acc = accuracy_score(ypredfinal,ytest) # determine the accuracy score 
	aucscr = roc_auc_score(ypredfinal,ytest) # determine roc_auc score
	f1score = f1_score(ypredfinal,ytest) # determine the f1 score
	fpr,tpr,_ = roc_curve(ytest,ypred[:,1]) # roc curve
	plt.plot(fpr,tpr)
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('roc curve') # plot roc curve
	plt.show()
	# cross validation
	cv_results = cross_validate(lr,xtrain,ytrain,cv=10)
	print(cv_results['test_score'],' cross validation results')
	return acc,aucscr,f1score

def main():
	df = load_data()
	modeltype = 0
	max_exp = 5
	# 30 experinments
	acc_lst = np.empty(max_exp)
	auc_score_lst = np.empty(max_exp)
	f1score_lst = np.empty(max_exp)
	for exp in range(max_exp):
		acc_lst[exp],auc_score_lst[exp],f1score_lst[exp] = logistic_regr_model(df,exp,modeltype)
	print(acc_lst,'Accuracy Score 30 exp')
	print(acc_lst.mean(),'Accuracy Score mean',acc_lst.std(),'Accuracy Score std')
	print(auc_score_lst,'AUC 30 exp')
	print(auc_score_lst.mean(),'AUC Score mean',auc_score_lst.std(),'AUC Score std')
	print(f1score_lst,'F1 30 exp')
	print(f1score_lst.mean(),'F1 Score mean',f1score_lst.std(),'F1 Score std')

if __name__ == '__main__':
	main()