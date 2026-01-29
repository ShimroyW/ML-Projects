import numpy as np
from sklearn.model_selection import KFold


class Train_Model():
    
    " --------------------------- Initialize Class ----------------------- "
    def __init__(self):
        pass
    
    
    " -------------------------- Train Epoch ---------------------------- "
    def _train_epoch(self, net, X : np.ndarray, y : np.ndarray, 
                    loss_fn, lr : float, batch_size : int) -> float:

        """
        Train for one epoch.

        Returns:
            average loss, average accuracy for the epoch
        """

        # Initialize params:
        total_loss    = 0
        total_correct = 0
        n_samples     = 0

        for X_batch, y_batch in self._batch_creation(X, y, batch_size, shuffle = True):

            # Forward pass
            logits         = net.forward(X_batch)
            loss, probs    = loss_fn.forward(logits, y_batch)

            # Compute accuracy
            predictions    = np.argmax(probs, axis=1)
            total_correct += np.sum(predictions == y_batch)
            total_loss    += loss * len(y_batch)
            n_samples     += len(y_batch)

            # Backward pass
            dlogits        = loss_fn.backward()
            net.backward(dlogits)

            # Update weights
            net.optimizer_step(lr)

        return total_loss / n_samples, total_correct / n_samples
    
    
    " -------------------------- Evaluate ---------------------------- "
    def _evaluate(self, net, X : np.ndarray, y : np.ndarray, loss_fn, 
                 batch_size : int = 256) -> float:

        """
        Evaluate network on data.

        Returns:
            average loss, accuracy
        """

        # Initialize params:
        total_loss         = 0
        total_correct      = 0
        n_samples          = 0

        for X_batch, y_batch in self._batch_creation(X, y, batch_size, shuffle=False):
            logits         = net.forward(X_batch)
            loss, probs    = loss_fn.forward(logits, y_batch)

            predictions    = np.argmax(probs, axis=1)
            total_correct += np.sum(predictions == y_batch)
            total_loss    += loss * len(y_batch)
            n_samples     += len(y_batch)

        return total_loss / n_samples, total_correct / n_samples
    
    
    '-------------------------------- Create batches ------------------------------------'
    def _batch_creation(self, X : np.ndarray, y : np.ndarray, batch_size : int, shuffle : bool = True):
        """
            Create mini-batches from data.

        Args:
             X           : data matrix X.
             y           : output vector y.
             batch_size  : size of batch
             shuffle     : shuffle indices (True/False)

        Returns:
            /
        """

        n_samples_tot = X.shape[0]
        indices       = np.arange(n_samples_tot)

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples_tot, batch_size):

            end_idx       = min(start_idx + batch_size, n_samples_tot)
            batch_indices = indices[start_idx:end_idx]

            yield X[batch_indices], y[batch_indices]
            
            
    " -------------------------- Train full set ---------------------------- "
    def _train(self, net, X_train : np.ndarray, y_train : np.ndarray, X_val : np.ndarray, y_val : np.ndarray,
              epochs : int, lr : float, batch_size : float, loss_fn = None, verbose : bool = True) -> dict:
        """
        Full training loop with validation.

        Returns:
            history dict with train/val loss and accuracy
        """

        if loss_fn is None:
            print('Please select a loss function')
            history = {}
        
        else:
            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss'  : [], 'val_acc'  : []
            }

            for epoch in range(epochs):
                train_loss, train_acc = self._train_epoch(net, X_train, y_train, loss_fn, lr, batch_size)
                val_loss, val_acc     = self._evaluate(net, X_val, y_val, loss_fn)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if verbose:
                    print(f"Epoch {epoch+1:2d}/{epochs} - "
                          f"Train Loss : {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss   : {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return history