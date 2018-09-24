import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import K_Fold_Cross


class model:
    
    def __init__(self):
        pass
    
    def algo(self, train_data, test_data):
        
        # Instantiate object of class 'k_fold'
        k_cross = K_Fold_Cross.k_fold()
        plot, correct_K, correct_error = k_cross.cross(train_data)

        print("Estimated Error Rate = ",correct_error," with K= ",correct_K)
        plt.show(plot)

        #Separating the predictors and labels for train and test data
        train_x = np.array(train_data.iloc[:,:-1])
        train_y = train_data.iloc[:,-1]
        test_x = np.array(test_data.iloc[:,:-1])
        test_y = test_data.iloc[:,-1]

        #Predicting actual error rate for Test Set
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
        clf.fit(train_x,train_y)
        pred_y = clf.predict(test_x)

        actual_error_rate = np.square(np.subtract(test_y,pred_y)).mean()
        print("Actual Error Rate  = ",actual_error_rate)
        




data = pd.read_csv('student_scores.csv')

cross_validation = model()

#Dividing the train and test data
train = data.iloc[:20,:]
test = data.iloc[20:,:]

cross_validation.algo(train, test)