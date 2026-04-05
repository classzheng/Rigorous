import numpy as np
import math_lib as ml

class Param:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = None
    
    def zero_grad(self):
        self.grad = None
    
    def __repr__(self):
        return f"Param(data.shape={self.data.shape}, has_grad={self.grad is not None})"

class Linear:
    def __init__(self, in_dim, out_dim, init='he'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init = init
        self.W = None
        self.b = None
        self.x = None
    
    def __call__(self, x):
        self.x = x
        if self.W is None:
            self._init_weights()
        return x @ self.W.data + self.b.data
    
    def _init_weights(self):
        if self.init == 'he':
            self.W = Param(ml.he((self.in_dim, self.out_dim)))
        elif self.init == 'xavier':
            self.W = Param(ml.xavier((self.in_dim, self.out_dim)))
        elif self.init == 'rand':
            self.W = Param(ml.rand((self.in_dim, self.out_dim)))
        else:
            self.W = Param(ml.const((self.in_dim, self.out_dim), 0.01))
        self.b = Param(ml.const((self.out_dim,), 0))
    
    def backward(self, dz):
        gW = self.x.T @ dz
        gb = dz.sum(axis=0)
        gx = dz @ self.W.data.T
        
        if self.W.grad is None:
            self.W.grad = gW
        else:
            self.W.grad += gW
        
        if self.b.grad is None:
            self.b.grad = gb
        else:
            self.b.grad += gb
        
        return gx
    
    @property
    def params(self):
        return [self.W, self.b]

class ReLU:
    def __init__(self):
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return ml.relu_forward(x)
    
    def backward(self, dz):
        return ml.relu_backward(dz, self.x)
    
    @property
    def params(self):
        return []

class Sigmoid:
    def __init__(self):
        self.x = None
        self.out = None
    
    def __call__(self, x):
        self.x = x
        self.out = ml.sigmoid_forward(x)
        return self.out
    
    def backward(self, dz):
        return ml.sigmoid_backward(dz, self.x)
    
    @property
    def params(self):
        return []

class Tanh:
    def __init__(self):
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return ml.Tanh.forward(x)
    
    def backward(self, dz):
        return ml.Tanh.backward(dz, self.x)
    
    @property
    def params(self):
        return []

class Softmax:
    def __init__(self, axis=-1):
        self.axis = axis
        self.x = None
        self.out = None
    
    def __call__(self, x):
        self.x = x
        self.out = ml.softmax_forward(x, self.axis)
        return self.out
    
    def backward(self, dz):
        return dz
    
    @property
    def params(self):
        return []

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def __call__(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x
    
    def backward(self, dz):
        return dz * self.mask if self.mask is not None else dz
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True
    
    @property
    def params(self):
        return []

class MSELoss:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return ml.mse_forward(y_pred, y_true)
    
    def backward(self):
        return ml.mse_backward(self.y_pred, self.y_true)

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.logits = None
        self.y_true = None
    
    def __call__(self, logits, y_true):
        self.logits = logits
        self.y_true = y_true
        loss, self.probs = ml.cross_entropy_forward(logits, y_true)
        return loss
    
    def backward(self):
        return ml.cross_entropy_backward(self.probs, self.y_true)

class Network:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.params = []
        self._collect_params()
    
    def add(self, layer):
        self.layers.append(layer)
        self._collect_params()
    
    def _collect_params(self):
        self.params = []
        for layer in self.layers:
            for p in layer.params:
                if p is not None:
                    self.params.append(p)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward(self, x):
        return self(x)
    
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
    
    def backward(self, loss):
        dz = loss.backward()
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                dz = layer.backward(dz)
        return dz
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
    
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
    
    def summary(self):
        print("=" * 60)
        print("网络结构")
        print("=" * 60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_info = f"Layer {i}: {layer.__class__.__name__}"
            if hasattr(layer, 'in_dim'):
                layer_info += f" ({layer.in_dim} -> {layer.out_dim})"
                if layer.W is not None:
                    params = layer.W.data.size + layer.b.data.size
                    total_params += params
                    layer_info += f", 参数: {params:,}"
            print(layer_info)
        print("=" * 60)
        print(f"总参数数量: {total_params:,}")
        print("=" * 60)

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) if p is not None else None for p in params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            if self.momentum > 0:
                self.v[i] = self.momentum * self.v[i] - self.lr * p.grad
                p.data += self.v[i]
            else:
                p.data -= self.lr * p.grad

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) if p is not None else None for p in params]
        self.v = [np.zeros_like(p.data) if p is not None else None for p in params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop:
    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.s = [np.zeros_like(p.data) if p is not None else None for p in params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            self.s[i] = self.decay * self.s[i] + (1 - self.decay) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (np.sqrt(self.s[i]) + self.eps)

class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step % self.step_size == 0:
            self.optimizer.lr *= self.gamma
    
    def get_lr(self):
        return self.optimizer.lr

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.current_step = 0
        self.base_lr = optimizer.lr
    
    def step(self):
        self.current_step += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + np.cos(np.pi * self.current_step / self.T_max)) / 2
    
    def get_lr(self):
        return self.optimizer.lr

def numerical_gradient_check(model, x, y, loss_fn, epsilon=1e-5):
    original_grads = [p.grad.copy() if p.grad is not None else None for p in model.params]
    
    model.zero_grad()
    
    output = model(x)
    loss = loss_fn(output, y)
    
    model.backward(loss_fn)
    
    max_rel_error = 0
    for i, p in enumerate(model.params):
        if p.grad is None:
            continue
        
        grad_numeric = np.zeros_like(p.data)
        for idx in np.ndindex(p.data.shape):
            old_val = p.data[idx]
            
            p.data[idx] = old_val + epsilon
            output_plus = model(x)
            loss_plus = loss_fn(output_plus, y)
            
            p.data[idx] = old_val - epsilon
            output_minus = model(x)
            loss_minus = loss_fn(output_minus, y)
            
            p.data[idx] = old_val
            grad_numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        rel_error = np.abs(p.grad - grad_numeric) / (np.abs(p.grad) + np.abs(grad_numeric) + 1e-8)
        max_rel_error = max(max_rel_error, np.max(rel_error))
        
        print(f"参数 {i}: 最大相对误差 = {np.max(rel_error):.6e}")
    
    for i, p in enumerate(model.params):
        if original_grads[i] is not None:
            p.grad = original_grads[i]
    
    return max_rel_error < 1e-5

def create_mlp(input_dim, hidden_dims, output_dim, activation='relu'):
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(Linear(prev_dim, hidden_dim))
        if activation == 'relu':
            layers.append(ReLU())
        elif activation == 'sigmoid':
            layers.append(Sigmoid())
        elif activation == 'tanh':
            layers.append(Tanh())
        prev_dim = hidden_dim
    
    layers.append(Linear(prev_dim, output_dim))
    
    return Network(layers)