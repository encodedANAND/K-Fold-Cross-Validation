import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class k_fold:
    
    def __init__(self):
        pass
    
    
    def cross(self, df):
        k=2
        min_key = 2

        # Preparing plot between value of 'K' and estimated error
        fig = plt.figure()
        plt.xlabel('No. of splits')
        plt.ylabel('Error')
        
        while(k<=len(df)):
            self.n_splits = k
            n= (len(df) // self.n_splits)
            total_error = 0
            
            for i in range(self.n_splits):
            	# Preparing the train and validation data
                validation = df[(i*n):((i+1)*n)]
                train = pd.concat([df[:(i*n)],df[((i+1)*n):]])
                total_error += self.model(train ,validation)
            
            
            total_error = total_error/k

            if(k==2):
            	min_error = total_error
            else:
            	if(total_error<min_error):
            		min_error = total_error
            		min_key = k

            fig = plt.scatter(k,total_error,color='g')
            k= k+1
        
        return (fig,min_key,min_error)    


    
    def model(self, train, validation):

        train_x = np.array(train.iloc[:,:-1])
        train_y = train.iloc[:,-1]

        validation_x = np.array(validation.iloc[:,:-1])
        validation_y = validation.iloc[:,-1]
        

        #Making estimates from the validation set
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
        clf.fit(train_x,train_y)
        pred_y = clf.predict(validation_x)
        
        error = np.square(np.subtract(validation_y,pred_y)).mean()
        return error

