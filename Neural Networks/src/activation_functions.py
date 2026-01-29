import numpy as np

class ReLU():
    
    """
        ReLU activation: y = max(0, x)
    """

    
    " --------------------------- Initialize Class ----------------------- "
    def __init__(self):
        self.cache = None

        
    " --------------------------- Forward Pass ----------------------- "
    def forward(self, x : np.ndarray) -> np.ndarray:
        
        self.cache = x
        
        return np.maximum(x, 0)

    
    " --------------------------- Backward Pass ----------------------- "
    def backward(self, d_upstream : np.ndarray) -> np.ndarray:
        
        x          = self.cache
        dx         = d_upstream.copy()
        dx[x <= 0] = 0
        
        return dx


class Sigmoid():
    """
        Sigmoid activation: y = 1 / (1 + exp(-x))
    """

    " --------------------------- Initialize Class ----------------------- "
    def __init__(self):
        self.cache = None

    " --------------------------- Forward Pass ----------------------- "
    def forward(self, x : np.ndarray, xmax : float = 500, xmin : float = -500) -> np.ndarray:
        
        # -> Why clipping at these values?
        
        y = 1.0 / (1.0 + np.exp(-np.clip(x, xmin, xmax)))  # Clip for numerical stability
        self.cache = y
        
        return y

    " --------------------------- Backward Pass ----------------------- "
    def backward(self, d_upstream : np.ndarray) -> np.ndarray:
        
        y = self.cache
        
        return y * (1 - y) * d_upstream


class HyerbolicTangent():
    
    """
        Hyperbolic Tangent activation: y = tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    """
    
    " --------------------------- Initialize Class ----------------------- "
    def __init__(self):
        self.cache = None
    
    " --------------------------- Forward Pass ----------------------- "
    def forward(self, x : np.ndarray) -> np.ndarray:
                
        y = np.tanh(x)
        self.cache = y
        
        return y
    
    
    " --------------------------- Backward Pass ----------------------- "
    def backward(self, d_upstream : np.ndarray) -> np.ndarray:
        
        y = self.cache
        
        return (1 - (y * y) ) * d_upstream