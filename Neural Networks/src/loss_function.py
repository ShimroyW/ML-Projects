import numpy as np

class SoftmaxCrossEntropyLoss():
    
    """
        Combined Softmax + Cross-Entropy loss for numerical stability.
    """

    " --------------------------- Initialize Class ----------------------- "
    def __init__(self):
        self.cache = None

    " --------------------------- Forward Pass ----------------------- "
    def forward(self, logits : np.ndarray, y_true : np.ndarray) -> np.ndarray:
        
        """
            Compute softmax cross-entropy loss.
        
        Args:
            logits: raw scores (batch_size, num_classes)
            y_true: true labels as integers (batch_size,)
            
        Returns:
            loss: scalar loss value
            probs: softmax probabilities
        """
        
        batch_size = logits.shape[0]

        # Numerically stable softmax
        logits_shifted = logits - np.max(logits, axis = 1, keepdims = True)
        exp_logits     = np.exp(logits_shifted)
        probs          = exp_logits / np.sum(exp_logits, axis = 1, keepdims = True)

        # Cross-entropy loss
        
        # -> Why clipping again?
        
        log_probs = np.log(np.clip(probs, 1e-15, 1.0))
        loss      = -np.mean(log_probs[np.arange(batch_size), y_true.astype(int)])

        self.cache = (probs, y_true, batch_size)
        
        return loss, probs
    
    " --------------------------- Backward Pass ----------------------- "
    def backward(self):
        
        """
            Gradient of combined softmax + cross-entropy.
            Simplifies to (probs - one_hot_labels) / batch_size
        """
        
        probs, y_true, batch_size = self.cache
        
        dx = probs.copy()
        dx[np.arange(batch_size), y_true.astype(int)] -= 1
        dx /= batch_size
        
        return dx
    
    
class MSELoss():
    
    """
        Mean Squared Error loss for regression tasks.
        Loss = (1/n) * sum((y_pred - y_true)^2)
    """
    
    " --------------------------- Initialize Class ----------------------- "
    def __init__(self):
        self.cache = None
        
    " --------------------------- Forward Pass ----------------------- "
    def forward(self, y_pred, y_true):
        """
            Compute MSE loss.
        """
        
        y_pred = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
        y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
        
        self.cache = (y_pred, y_true)
        
        return np.mean((y_pred - y_true) ** 2)

    " --------------------------- Backward Pass ----------------------- "
    def backward(self):
        
        """
            Gradient: d(MSE)/d(y_pred) = 2(y_pred - y_true) / n
        """
        
        y_pred, y_true = self.cache
        
        return 2 * (y_pred - y_true) / y_pred.shape[0]
