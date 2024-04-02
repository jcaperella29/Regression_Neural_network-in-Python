# Regression_Neural_network-in-Pythonrk-in
This is a Python script that produces a neural network for performing regression analysis.The sample data is regarding car sales but it  can be used to predict any continuous variable  
First needed modules are imported. Then scientific notation is removed from numpy arrays and the data is read in.
Next , the predictors and target variables are labeled and data is normalized.
Then the data is spilt into train and test and the model framework is built.
Next , the model parameters are tuned and the best parameter choices are ploted
Then the model is fit using the new parameters.
Then  both the predictions and orginal data are returned to orginal scale.
Next, the predictions and the test data are placed inside a dataframe
Then error for each prediction  is calculated in terms of absolute percent error.
The error for each prediction is then placed in the same dataframe as the predicitons and test data.
Finally ,that dataframe is exported as a csv.
