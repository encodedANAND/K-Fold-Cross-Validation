# K-Fold-Cross-Validation

K -Fold Cross Vaidation is one of the known Resampling method used for estimating the test error rate.In this technique, the data is divided into 'k' parts ,each time one of the part is considered as test set while rest as the train set.It is repeated for 'k' times and MSE(mean squared error) is estimated. K-fold estimate is calculated by averaging these estimates.

In **K_Fold_Cross**, a self defined k-fold cross validation class is defined which calculates estimate for large values of k and give the plot and 'k' with minimum error estimate.
In **Cross_Validation_Model**, a predictive model is defined for which we have to estimate the error. We will predict the value of 'k' for this class. This class will also predict the actual error estimate.
We have used **student_scores** dataset containing the data of scores gained by students based upon the numbers of hours studied by them.

On running the code, we can easily find that the estimated error is high for low values of 'k' but decreases sharply on increasing 'k'. After some point, it becomes almost constant. We can easily get a optimum value of 'k' around k=10.
