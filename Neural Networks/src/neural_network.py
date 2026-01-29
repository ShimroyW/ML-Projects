import numpy as np

class Linear():
    
    " --------------------------- Initialize Class ----------------------- "
    def __init__(self, features_in : int, features_out : int):
        self.features_in  = features_in
        self.features_out = features_out
        self.weight       = np.random.randn(features_out, features_in)
        self.bias         = np.random.randn(features_out)
        self.cache        = None
        self.weight_grad  = None
        self.bias_grad    = None
        
    " --------------------------- Initialize parameters ----------------------- "
    def init_params(self, std : float = 0.1) -> None:
        
        """ 
            Initialize layer parameters with He/Xavier-like initialization.
        """
        
        # Using scaled initialization for better convergence
        self.weight = std * np.random.randn(*self.weight.shape) / np.sqrt(self.features_in)
        self.bias   = np.zeros(self.features_out)

    " --------------------------- Forward Pass ----------------------- "
    def forward(self, x : np.ndarray) -> np.ndarray:
        
        """
            Forward pass: y = xW^T + b
        """
        
        self.cache = x
        
        return x @ self.weight.T + self.bias
    
    
    " --------------------------- Backward Pass ----------------------- "
    def backward(self, d_upstream):
        """ 
            Backward pass: compute gradients and return dx.
        """
        
        x  = self.cache
        dx = d_upstream @ self.weight
        
        self.weight_grad = d_upstream.T @ x
        self.bias_grad   = np.sum(d_upstream, axis = 0)
        
        return dx

class NeuralNet():
    """
        Neural network container for layers.
    """
    
    " --------------------------- Initialize Class ----------------------- "
    def __init__(self, layers : int):
        self.layers = layers
        self.reset_params()

    " --------------------------- Reset Params ----------------------- "
    def reset_params(self, std : float = 1.0):
        
        """
            Reset all learnable parameters.
        """
        
        for layer in self.layers:
            if hasattr(layer, 'init_params'):
                layer.init_params(std = std)
                
    " --------------------------- Forward Pass ----------------------- "
    def forward(self, x : np.ndarray) -> np.ndarray:
        
        """
            Forward pass through all layers.
        """
        
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    
    " --------------------------- Backward Pass ----------------------- "
    def backward(self, d_upstream : np.ndarray) -> np.ndarray:
        
        """
            Backward pass through all layers (reversed).
        """
        
        dx = d_upstream
        
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
            
        return dx

    " --------------------------- Optimizer ----------------------- "
    def optimizer_step(self, lr : float):
        
        """
            Update weights using gradient descent.
        """
        
        for layer in self.layers:
            if hasattr(layer, 'weight') and layer.weight_grad is not None:
                layer.weight -= lr * layer.weight_grad
                
            if hasattr(layer, 'bias') and layer.bias_grad is not None:
                layer.bias -= lr * layer.bias_grad
