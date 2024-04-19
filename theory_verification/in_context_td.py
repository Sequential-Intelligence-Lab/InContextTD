import numpy as np

def stack_four(A, B, C, D):
    top = np.concatenate([A, B], axis=1)
    bottom = np.concatenate([C, D], axis=1)
    return np.concatenate([top, bottom], axis=0)

class TFLayer:
    def __init__(self, d, n, semi_grad = True):
        self.d = d
        self.n = n
        self.P = np.zeros((2 * d + 1, 2 * d + 1)) 
        self.P[-1, -1] = 1
        self.M = np.eye(n + 1)
        self.M[-1, -1] = 0
        I = np.eye(d)
        O = np.zeros((d, d))
        self.M1 = stack_four(-I, I, O, O)
        self.M2 = stack_four(I, O, O, O)
        self.C = np.eye(d) 
        self.B = stack_four(self.C.T, O, O, O)
        self.A = self.B @ self.M1
        self.Q = np.zeros(self.P.shape)
        self.Q[:2*d, :2*d] = self.A
        if not semi_grad:
            self.Q = self.Q.T
    
    def forward(self, Z):
        next_Z = Z + 1.0 / self.n * self. P @ Z @ self.M @ Z.T @ self.Q @ Z
        return next_Z

class Transformer:
    def __init__(self, l, d, n, semi_grad = True):
        self.n = n
        self.layers = [TFLayer(d, n, semi_grad) for _ in range(l)]
        self.Cs = [layer.C for layer in self.layers]
    
    def forward(self, Z):
        v = []
        for layer in self.layers:
            Z  = layer.forward(Z)
            v.append(-Z[-1, -1])
        return v, Z

class Prompt:
    def __init__(self, d, n, gamma):
        self.n = n
        self.gamma = gamma
        self.phi = np.concatenate([np.random.randn(d, 1) for _ in range(n+1)], axis=1)
        self.phi_prime = [np.random.randn(d, 1) for _ in range(n)]
        self.phi_prime.append(np.zeros((d, 1)))
        self.phi_prime = gamma * np.concatenate(self.phi_prime, axis=1)
        self.r = [np.random.randn(1)[0] for _ in range(self.n)]
        self.r.append(0)
        self.r = np.array(self.r)
        self.r = np.reshape(self.r, (1, -1))
    
    def z(self):
        return np.concatenate([self.phi, self.phi_prime, self.r], axis=0)
    
    def td_update(self, w, C, semi_grad = True):  
        u = 0
        for j in range(self.n):
            td_error = self.r[0, j] + w.T @ self.phi_prime[:, [j]] - w.T @ self.phi[:, [j]]
            if semi_grad:
                u += td_error * self.phi[:, [j]]
            else:
                u += td_error * (self.phi[:, [j]] - self.phi_prime[:, [j]])
        u /= self.n
        u = C @ u
        w += u
        v = w.T @ self.phi[:, [-1]]
        return w, v

def g(pro, tf, phi, phi_prime, r):
    pro.phi[:, [-1]] = phi
    pro.phi_prime[:, [-1]] = phi_prime
    pro.r[-1, -1] = r
    _, Z = tf.forward(pro.z())
    return Z


def verify(d, n, l, semi_grad = True):
    gamma = 0.9
    tf = Transformer(l, d, n, semi_grad=semi_grad)
    pro = Prompt(d, n, gamma)
    # for _ in range(5):
    #     phi = np.random.randn(d, 1)
    #     phi_prime = np.random.randn(d, 1)
    #     r = np.random.randn()
    #     g(pro, tf, phi, phi_prime, r)
    tf_value, _ = tf.forward(pro.z())
    w = np.zeros((d, 1))
    td_value = []
    for i in range(l):
        w, v = pro.td_update(w, tf.Cs[i], semi_grad=semi_grad)
        td_value.append(v)
    td_value = np.array(td_value).flatten()
    print(tf_value - td_value)
# 
if __name__ == '__main__':
    verify(4, 9, 5)
    # verify residual gradient
    verify(4,9, 1, semi_grad=False)



