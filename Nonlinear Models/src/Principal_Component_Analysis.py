import numpy as np

class Principal_Component_Analysis():
    
    " ----------------------- Initialize Class ----------------------- "
    def __init__(self):
        pass
    
    
    " ----------------------- Adjust Data ----------------------- "
    def _data_adjustment(self, X : np.ndarray) -> np.ndarray:
        
        # Adjust for column averages te remove bias:
        X = X - np.matmul(np.ones((X.shape[0], 1)), X.mean(axis = 0).reshape(X.shape[1], 1).T)   

        return X
    
    
    " ----------------------- Singular Value Decomposition ----------------------- "
    def _singular_value_decomposition(self, X, mode : str = 'full', k : int = 1) -> np.ndarray:


        """
        Args:
             X    : data matrix X.
             mode : Set mode (full or truncated).
             k    : Set k-order if mode is truncated

        """

        # Select mode for computation:
        match mode:

            case 'full':
                print('Full Singular Value Decomposition.')

                if X.shape[1] >  np.linalg.matrix_rank(X):
                    print('Note: Matrix is not full rank! Therefore k is adjusted to equal rank of matrix X.')

                k = np.linalg.matrix_rank(X)

            case 'truncated':
                print('k-truncated Singular Value Decomposition.')

                if k > np.linalg.matrix_rank(X):
                    k = np.linalg.matrix_rank(X)
                    print('Order of k higher then rank of matrix X -> k is adjusted to equal rank of matrix X.')

                elif k == np.linalg.matrix_rank(X):
                    print('Order of k equals rank of X -> Full Singular Value Decomposition.')


            case _ :
                print('Please select either full or truncated for mode. k-order is set to default value of 1.')
                k = 1

        # Compute eigenvalues of (X^T)X:
        eigenvalues, eigenvectors = np.linalg.eig(np.matmul(X.transpose(), X))
        
        # Sort eigenvalues -> highest value to lowest value:
        eigenvectors    = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        eigenvalues     = eigenvalues[np.argsort(eigenvalues)[::-1]]
        
        # Generate diagonal Sigma matrices:
        Sigma     = np.zeros((k, k), dtype = float)
        Sigma_inv = np.zeros((k, k), dtype = float)

        # Fill Sigma matrices up to k-orer
        np.fill_diagonal(Sigma, np.sqrt(eigenvalues[:k]))
        np.fill_diagonal(Sigma_inv, 1/np.sqrt(eigenvalues[:k]))

        # Create matrix V with op to k-largest eigenvectors:
        V = eigenvectors[:, :k]

        # Create matrix U:
        U = np.matmul(np.matmul(X, V), Sigma_inv)

        return U, Sigma, V
    
    
    " ----------------------- Outlier Analysis ----------------------- "
    def _outlier_analysis(self, X : np.ndarray, magnitude : float = 3) -> np.ndarray:

        # Outliers:
        rows_outliers = np.array([])

        for col_idx in range(X.shape[1]):

            # Compute radius:
            r = magnitude*np.std(X[:, col_idx])

            # Select indices for valid and invalid data:
            idx_valid   = np.where(np.abs(X[:, col_idx]) <= r)
            idx_invalid = np.where(np.abs(X[:, col_idx]) >  r)

            if len(idx_invalid[0]) > 0:
                rows_outliers = np.append(rows_outliers, idx_invalid[0])

        # Get indicies of rows with outliers:
        rows_outliers = np.sort(np.unique(rows_outliers)).astype(int)

        return rows_outliers