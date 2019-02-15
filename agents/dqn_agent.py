import random
import numpy as np
import torch
from torch import optim
from torch.nn import functional
from torchvision import transforms
from agents.nets.reward_estimator import DQN
from agents.utils.epsilon import Epsilon
from agents.utils.replay_memory import ReplayMemory, Transition

class DQNAgent(object):
    def __init__(self,
                 state_shape,
                 lr=0.0001,
                 batch_size=2048,
                 gamma=0.7,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=20000,
                 num_of_actions=2):
        self.step = 0
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = Epsilon(eps_start, eps_end, eps_decay)
        self.qnet = DQN(state_shape, num_actions=num_of_actions).to(device)
        self.target_net = DQN(state_shape, num_actions=num_of_actions).to(device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.qnet.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.max_reward = 1.
        self.state_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def act(self, obs):
        state = self.state_transforms(obs).unsqueeze(0)
        sample = random.random()
        eps_th = self.epsilon()
        # print(eps_th)
        if sample > eps_th:
            with torch.no_grad():
                val, idx = self.qnet(state).max(1)
                return idx.item()
        return np.random.randint(6)

    def update_qnet(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        state_batch = torch.stack(tuple([self.state_transforms(state) for state in batch.state]))
        action_batch = torch.stack(tuple([torch.tensor(action, device=self.device, dtype=torch.long, requires_grad=False)
                                        for action in batch.action]))
        reward_batch = torch.stack(tuple([torch.tensor(reward, device=self.device, dtype=torch.float32, requires_grad=False)
                                        for reward in batch.reward]))

        q = self.qnet(state_batch)
        state_action_values = q.gather(1, action_batch.view(self.batch_size, 1))

        expected_state_action_values = self.future_reward_estimate(batch) + reward_batch

        loss = functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnet.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

    def future_reward_estimate(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = torch.stack(tuple([self.state_transforms(s) for s in batch.next_state if s is not None]))
        next_state_values = torch.zeros(self.batch_size, device=self.device, requires_grad=False)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        return next_state_values * self.gamma

    def update_target_net(self):
        self.target_net.load_state_dict(self.qnet.state_dict())

    def push_transition(self, state, action, next_state, reward):
        # print('transition state {} action {} next_state {} reward {}'.format(0, action, 0, reward))
        self.memory.push(state, action, next_state, reward)

    def save_target_net(self, id):
        torch.save(self.target_net.state_dict(), 'trained_models/target_net_{}'.format(str(id)))

    def save_qnet(self, id):
        torch.save(self.qnet.state_dict(), 'trained_models/qnet_{}'.format(str(id)))
