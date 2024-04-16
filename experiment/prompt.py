import torch
from MRP.boyan import BoyanChain
import numpy as np
from collections import deque


class Feature:
    def __init__(self, d: int, s: int):
        '''
        d: dimension of the feature vector
        s: number of states

        '''
        self.d = d
        self.s = s
        self.phi = np.random.uniform(low=-1, high=1, size=(s, d))

    def __call__(self, s: int):
        return self.phi[s]


class Prompt:
    def __init__(self,
                 d: int,
                 n: int,
                 gamma: float,
                 w: torch.Tensor = None,
                 noise: float = 0.1):
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
            u = C @ u  # apply conditioning matrix
        new_w = w + u
        v = new_w.t() @ self.phi[:, [-1]]
        return new_w, v.item()


class MDPPrompt:
    def __init__(self,
                 d: int,
                 n: int,
                 gamma: float,
                 mdp: BoyanChain,
                 feature_fun: Feature):
        '''
        d: feature dimension
        n: context length
        gamma: discount factor
        mdp: an instance of a BoyanChain MDP
        feature_fun: a function that returns the feature vector of a state  
        '''
        self.d = d
        self.n = n
        self.gamma = gamma
        self.mdp = mdp
        self.feature_fun = feature_fun

    def reset(self):
        self.feature_window = deque(maxlen=self.n+2)
        self.reward_window = deque(maxlen=self.n+1)
        # populates the feature and rewards
        self.s = self.mdp.reset()
        self.feature_window.append(self.feature_fun(self.s))
        for _ in range(self.n+1):
            s_prime, r = self.mdp.step(self.s)
            self.feature_window.append(self.feature_fun(s_prime))
            self.reward_window.append(r)
            self.s = s_prime

        self._store_data()
        
        return self.z()
    
   


    def step(self):
        # step the MDP
        s_prime, r = self.mdp.step(self.s)
        self.feature_window.append(self.feature_fun(s_prime))
        self.reward_window.append(r)
        self.s = s_prime

        self._store_data()
        return self.z(), r
    
    def _store_data(self):
        features = np.array(self.feature_window, dtype=np.float32)
        rewards = np.array(self.reward_window, dtype=np.float32)
        self.phi = torch.from_numpy(features[:self.n]).T
        self.phi_prime = self.gamma*torch.from_numpy(features[1:self.n+1]).T
        self.r = torch.from_numpy(rewards[:self.n]).unsqueeze(0)
        self._context = torch.concat([self.phi, self.phi_prime, self.r], dim=0)
        self._query = torch.from_numpy(features[self.n+1]).reshape(self.d, 1)

    def context(self):
        return self._context
    
    def query(self):
        return self._query

    def z(self):
        query_col = torch.concat([self._query, torch.zeros((self.d+1, 1))], dim=0)
        return torch.concat([self._context, query_col], dim=1)

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
            u = C @ u  # apply conditioning matrix
        new_w = w + u
        v = new_w.t() @ self.phi[:, [-1]]
        return new_w, v.item()

class MDP_Prompt_Generator:
    def __init__(self,
                 mdp: BoyanChain,
                 features: Feature,
                 n: int,
                 eval_len: int,
                 gamma: float):
        '''
        mdp: an instance of a BoyanChain MDP
        features: the features
        n: context length
        eval_len: # of training samples we want this MDP to generate
        gamma: discount factor
        '''
        self.mdp = mdp
        self.features = features
        self.n = n
        self.eval_len = eval_len
        self.gamma = gamma
        self.slide_idx = 0

        try:
            assert eval_len > 0  # eval_len must be greater than 0
        except AssertionError as e:
            e.args += ("We must use this MDP to generate at least 1 training sample", 42)
            raise e

        rows = []
        # sample from initial state distribution
        s = mdp.reset()

        # unroll the MDP n + eval_len + 1 step
        for _ in range(self.n + self.eval_len+1):
            s_prime, r = mdp.step(s)
            row = np.concatenate([self.features.get_feature(
                s), self.gamma*self.features.get_feature(s_prime), [r]])
            rows.append(row)
            s = s_prime

        self.full_seq = np.stack(rows, axis=-1)
        self.next_prompt()

    # return the prompt as a tensor
    def z(self) -> torch.Tensor:
        return self.z_0

    # return the context as a tensor
    def context(self) -> torch.Tensor:
        return self.z_0[:, :-1]

    # returns the features of the query as a numpy array
    def query_features(self) -> np.ndarray:
        return self.z_0[:self.features.d, -1:].detach().numpy().T

    # returns the reward of the query as a tensor
    def query_state_reward(self) -> torch.Tensor:
        return self.query_reward

    def next_prompt(self):
        try:
            assert self.slide_idx+self.n+1 < self.full_seq.shape[1]
        except AssertionError as e:
            e.args += ("You cannot generate more than {eval_len} prompts using this MDP.".format(
                eval_len=self.eval_len), 42)
            raise e
        # update Z_0
        self.z_0 = torch.tensor(
            self.full_seq[:, self.slide_idx: self.slide_idx+self.n+1], dtype=torch.float32)
        # get the reward of the query before we zero it out to generate the prompt
        self.query_reward = self.z_0[-1, -1].detach().clone()
        self.z_0[self.features.d:, -1] = 0

        # slide the window by 1 for the next prompt
        self.slide_idx += 1


if __name__ == '__main__':
    d = 3
    s = 10
    n = 6
    eval_len = 3
    gamma = 0.9
    feat = Feature(d, s)
    bc = BoyanChain(s, gamma)
    mdp_prompt = MDPPrompt(d, n, gamma, bc, feat)
    Z_0 = mdp_prompt.reset()
    print(Z_0)
    Z_1, r =  mdp_prompt.step()
    print(Z_1)
    print(r)
    
    # mdp_prompt = MDP_Prompt_Generator(bc, feat, n, eval_len, gamma)

    # print("Features")
    # print(feat.phi)
    # print("Z_0")
    # print(mdp_prompt.z())
    # print("Context")
    # print(mdp_prompt.context())
    # print("Full Sequence")
    # print(mdp_prompt.full_seq)
    # for i in range(eval_len-1): # -1 is correct here since we already generate the first prompt upon initialization
    #     print( f"Prompt {i+1}")
    #     mdp_prompt.next_prompt()
    #     print("Z_0 after sliding")
    #     print(mdp_prompt.z())
    #     print("Query Features")
    #     print(mdp_prompt.query_features())
    #     print("Query Reward")
    #     print(mdp_prompt.query_state_reward())

    # assert mdp_prompt.z_0.shape == (2*d+1, n+1)
