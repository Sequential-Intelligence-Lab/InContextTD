import torch
#from experiment.boyan import BoyanChain
import numpy as np

class Feature:
    def __init__(self, d:int, s:int):
        '''
        d: dimension of the feature vector
        s: number of states
        
        '''
        self.d = d
        self.s= s
        self.phi = np.random.randn(s,d)

    def get_feature(self, s:int):
        return self.phi[s]

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
    
class MDP_Prompt:
    def __init__(self,
                 mdp: BoyanChain,
                 features: Feature,
                 n: int,
                 gamma: float):
        '''
        mdp: an instance of a BoyanChain MDP
        features: the features
        n: context length
        
        '''
        self.mdp = mdp
        self.features = features
        self.n = n
        self.gamma = gamma


        # sample from initial state distribution
        s = mdp.reset()
        for _ in range(self.n-1):
            s_prime, r = mdp.step(s)
            column = np.concatenate([features.get_feature(s), features.get_feature(s_prime),r], axis=1).reshape(-1,1)

        self.z_0 = torch.tensor(np.transpose(context))

        
if __name__ == '__main__':
    d = 3
    s = 5
    n = 20
    gamma = 0.9
    feat = Feature(d, s)
    print(feat.phi)
    print(feat.get_feature(2))
    import pdb; pdb.set_trace()
