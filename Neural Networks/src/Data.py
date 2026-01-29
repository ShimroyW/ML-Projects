import numpy as np
from sklearn.datasets import fetch_openml

class Data_Generation():
    
    '-------------------------------- Initialize Class ------------------------------------'
    def __init__(self):
        pass
    
    
    '-------------------------------- Acquire image data ------------------------------------'
    def _image_data(self, dataset : str = 'mnist_784', version : int = 1, normalize : bool = True) -> np.ndarray:
        """
        Load MNIST dataset.
            
        Args:
            dataset   : select dataset.
            version   : select version of dataset.
            normalize : normalize data (make max. value equal to 1).

        Returns:
            X       : images of shape (70000, 784)
            y       : labels of shape (70000,)
        """
        
        print(f'Loading "{dataset}" dataset (this might take a minute on the initial run)...')
        
        mnist = fetch_openml(dataset, version = version, as_frame = False, parser = 'auto')
        X, y  = mnist.data, mnist.target.astype(int)

        if normalize:
            X = X / np.amax(X, axis = None)

        print(f'Loaded {X.shape[0]} samples with {X.shape[1]} features')
        
        return X, y
    
    
    '-------------------------------- Split Data ------------------------------------'
    def _data_splitting(self, X : np.ndarray, y : np.ndarray, mode : str, n_samples_train : float, 
                        n_samples_val : float, n_samples_test : float, seed : int = None, 
                        shuffle : bool = True) -> np.ndarray:

        """
        Split data into train, test and validation sets.
        
        Args:
             X               : data matrix X.
             y               : output vector y.
             mode            : mode for splitting -> ratio takes a percentage of the available rows and index splits the data untill the given index value.
             n_samples_train : in case of mode ratio give a value between 0 and 1 and in case of mode index give an index value between 0 and max nr. of rows of matrix X (or vector y)
             n_samples_val   : in case of mode ratio give a value between 0 and 1 and in case of mode index give an index value between 0 and max nr. of rows of matrix X (or vector y)
             n_samples_test  : in case of mode ratio give a value between 0 and 1 and in case of mode index give an index value between 0 and max nr. of rows of matrix X (or vector y)
             seed            : set seed for random variables.
             shuffle         : shuffle indices (True/False).

        Returns:
            X_train      : train set.
            X_val        : validation set.
            X_test       : test set.
            y_train      : train target function.
            y_val        : validation target function.
            y_test       : test target function.

        """
        
        n_samples_tot = X.shape[0]                
        indices       = np.arange(n_samples_tot)
        
        if seed:
            np.random.seed(seed)
        
        if shuffle:
            np.random.shuffle(indices)
        
                        
        # Split data set:
        match mode:
            case 'ratio':
                
                if n_samples_train + n_samples_val + n_samples_test < 1:
                    print('The ratio`s for the train, validation and test samples are smaller then 1. Please adjust these such that it becomes 1 when summed.')

                elif n_samples_train + n_samples_val + n_samples_test > 1:
                    print('The ratio`s for the train, validation and test samples are greater then 1. Please adjust these such that it becomes 1 when summed.')
                    
                if (n_samples_train > 0) & (n_samples_train <= 1):
            
                    if n_samples_train*n_samples_tot % 1 > 0:
                        print(f'For the given value of n_samples_train a remainder has been found of {n_samples_train*n_samples_tot % 1}. Therefore, the ratio has been adjusted to: {int(n_samples_tot*n_samples_train)/n_samples_tot}.')
                        
                    # Train size:
                    train_size = int(n_samples_tot * n_samples_train)                        
                    train_idx  = indices[:train_size]
                    
                if (n_samples_val > 0) & (n_samples_val <= 1):
            
                    if n_samples_val*n_samples_tot % 1 > 0:
                        print(f'For the given value of n_samples_val a remainder has been found of {n_samples_val*n_samples_tot % 1}. Therefore, the ratio has been adjusted to: {int(n_samples_tot*n_samples_val)/n_samples_tot}.')
                        
                    # Validation size:
                    val_size   = int(n_samples_tot * n_samples_val) 
                    val_idx    = indices[train_size : (train_size + val_size)]
                        
                if (n_samples_test > 0) & (n_samples_test <= 1):
            
                    if n_samples_test*n_samples_tot % 1 > 0:
                        print(f'For the given value of n_samples_test a remainder has been found of {n_samples_test*n_samples_tot % 1}. Therefore, the ratio has been adjusted to: {int(n_samples_tot*n_samples_test)/n_samples_tot}.')
                        
                    # Test size:
                    test_size  = int(n_samples_tot * n_samples_test) 
                    test_idx   = indices[(train_size + val_size):]
                    
            case 'index':
                
                if (n_samples_test > 0) & (n_samples_test <= n_samples_tot):                
                    test_idx  = indices[:n_samples_test]
                    
                else:
                    print('Please make sure that the amount of samples selected for the test set is smaller then the total amount of samples.')
                    
                    
                if (n_samples_val > 0) & (n_samples_val <= n_samples_tot):
                    val_idx   = indices[n_samples_test : (n_samples_test + n_samples_val)]
                    
                else:
                    print('Please make sure that the amount of samples selected for the validation set is smaller then the total amount of samples.')
                    
                    
                if (n_samples_train > 0) & (n_samples_train <= n_samples_tot):
                    train_idx = indices[(n_samples_test + n_samples_val):]
                    
                else:
                    print('Please make sure that the amount of samples selected for the validation set is smaller then the total amount of samples.')
                    
                    
            case _:

                print('Please select either the "ratio" or "columns" mode.')
                                        
        # Select data matrices for train, validation and test:
        X_train = X[train_idx]
        X_val   = X[val_idx]
        X_test  = X[test_idx]

        # Select target functions for train, validation and test:
        y_train = y[train_idx]
        y_val   = y[val_idx]
        y_test  = y[test_idx]

        return X_train, X_val, X_test, y_train, y_val, y_test