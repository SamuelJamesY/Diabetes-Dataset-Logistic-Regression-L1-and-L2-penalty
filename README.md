# Diabetes-Dataset-Logistic-Regression-L1-and-L2-penalty

Numpy, pandas and sklearn were used to develop a three logistic regression models; a regular logistic regression
a lasso logistic regresion model (L1) and ridge logistic regression model (L2). The models were built using a Pima Indian Diabetes dataset.
The logistic models sought to classify whether an individual had diabetes or not. 
  
  Eight predictors were used to help determine whether a patient had diabetes. These were pregnancy,
  glucouse, blood pressure, skin thickness, insulin levels, bmi, DBF and age. A train test split of 
  60/40 was used for this dataset. Sklearn metrics were used to determine the classification accuracy score 
  of the model, the auc_roc score and f1_score. Further an roc_auc curve was plotted as a helpful visualisation. 
  
  Further k-fold cross validation was undertaken with 10 folds, to gain further insight into the accuracy of the 
  classification score. For each model 30 experiments were run and the mean and standard deviation 
  of the classification accuracy score was recorded. 
