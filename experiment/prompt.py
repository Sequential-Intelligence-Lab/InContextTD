import torch

class Prompt:
    def __init__(self, 
                 d: int, 
                 n: int,
                 gamma: float, 
                 w: torch.Tensor = None,
                 noise: float = 0.0):
        '''
        d: feature dimension
        n: context length
        gamma: discount factor
        w: weight vector (optional)
        noise: reward noise level
        '''
        self.d = d
        self.n = n
        self.gamma = gamma

        self.phi = torch.randn(d, n+1)

        phi_prime = torch.randn(d, n)
        phi_prime = torch.concat([phi_prime, torch.zeros((d, 1))], dim=1)
        self.phi_prime = gamma * phi_prime

        if w:
            self.w = w
        else:
            self.w = torch.randn((d, 1))

        r = self.w.t() @ self.phi - self.w.t() @ self.phi_prime
        r += noise * torch.randn((1, n+1)) # add random noise
        r[0, -1] = 0
        self.r = r

    def z(self):
        return torch.cat([self.phi, self.phi_prime, self.r], dim=0)

    def td_update(self, 
                  w: torch.Tensor, 
                  C: torch.Tensor = None):
        '''
        w: weight vector
        C: preconditioning matrix
        '''
        u = 0
        for j in range(self.n):
            target = self.r[0, j] + w.t() @ self.phi_prime[:, [j]]
            td_error = target - w.t() @ self.phi[:, [j]]
            u += td_error * self.phi[:, [j]]
        u /= self.n
        if C:
            u = C @ u # apply conditioning matrix
        new_w = w + u 
        v = new_w.t() @ self.phi[:, [-1]]
        return new_w, v.item()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    weight_diffs = []
    value_diffs = []
    d = 3
    n = 20
    gamma = 0.9
    pro = Prompt(d, n, gamma)
    true_w = pro.w
    true_v = true_w.t() @ pro.phi[:, [-1]]
    w = torch.zeros((d, 1))
    for _ in range(100):
        w, v = pro.td_update(w)
        weight_diffs.append(torch.norm(w - true_w).item())
        value_diffs.append(abs((v - true_v).item()))
    
    plt.plot(weight_diffs, label='Weight Difference')
    plt.plot(value_diffs, label='Value Difference')
    plt.legend()
    plt.show()
    plt.close()
