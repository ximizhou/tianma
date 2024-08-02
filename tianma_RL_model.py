import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
from collections import namedtuple
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from sum_tree import SumTree
from RL.binary_recommend_env import BinaryRecommendEnv
from RL.enumerated_recommend_env import EnumeratedRecommendEnv
from RL.evaluate import evaluate_dqn
from gcn import GraphEncoder
import time
import warnings

warnings.filterwarnings("ignore")

# Mapping of dataset names to environment classes
ENVIRONMENTS = {
    "LAST_FM": BinaryRecommendEnv,
    "LAST_FM_STAR": BinaryRecommendEnv,
    "YELP": EnumeratedRecommendEnv,
    "YELP_STAR": BinaryRecommendEnv
}

# Feature type for each dataset
FEATURE_TYPES = {
    "LAST_FM": 'feature',
    "LAST_FM_STAR": 'feature',
    "YELP": 'large_feature',
    "YELP_STAR": 'feature'
}

# Define the transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_cand'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.priority_max = 0.1
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def push(self, *args):
        data = Transition(*args)
        priority = (np.abs(self.priority_max) + self.epsilon) ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size):
        batch_data = []
        indices = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data, is_weight = self.tree.get(s)
            batch_data.append(data)
            priorities.append(p)
            indices.append(idx)

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return indices, batch_data, is_weights

    def update(self, indices, errors):
        self.priority_max = max(self.priority_max, max(np.abs(errors)))
        for i, idx in enumerate(indices):
            priority = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100):
        super(DQN, self).__init__()
        self.fc2_value = nn.Linear(hidden_size, hidden_size)
        self.out_value = nn.Linear(hidden_size, 1)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)
        self.out_advantage = nn.Linear(hidden_size, 1)

    def forward(self, state, action, choose_action=True):
        value = self.out_value(F.relu(self.fc2_value(state))).squeeze(dim=2)
        if choose_action:
            state = state.repeat(1, action.size(1), 1)
        state_action = torch.cat((state, action), dim=2)
        advantage = self.out_advantage(F.relu(self.fc2_advantage(state_action))).squeeze(dim=2)

        if choose_action:
            qsa = advantage + value - advantage.mean(dim=1, keepdim=True)
        else:
            qsa = advantage + value

        return qsa

class DQNAgent:
    def __init__(self, device, memory, state_size, action_size, hidden_size, gcn_net, learning_rate, l2_norm,
                 padding_id, eps_start=0.9, eps_end=0.1, eps_decay=0.0001, tau=0.01):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.policy_net = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(), self.gcn_net.parameters()), lr=learning_rate,
                                    weight_decay=l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.padding_id = padding_id
        self.tau = tau

    def select_action(self, state, candidates, action_space, is_test=False, is_last_turn=False):
        state_emb = self.gcn_net([state])
        candidates = torch.LongTensor([candidates]).to(self.device)
        candidates_emb = self.gcn_net.embedding(candidates)
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if is_test or sample > eps_threshold:
            if is_test and (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1]
            with torch.no_grad():
                action_values = self.policy_net(state_emb, candidates_emb)
                print(sorted(list(zip(candidates[0].tolist(), action_values[0].tolist())), key=lambda x: x[1], reverse=True))
                action = candidates[0][action_values.argmax().item()]
                sorted_actions = candidates[0][action_values.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist()
        else:
            shuffled_candidates = action_space[0] + action_space[1]
            random.shuffle(shuffled_candidates)
            return torch.tensor(shuffled_candidates[0], device=self.device, dtype=torch.long), shuffled_candidates

    def update_target_model(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def optimize_model(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        self.update_target_model()

        indices, transitions, is_weights = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))
        action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device)
        action_emb_batch = self.gcn_net.embedding(action_batch)
        reward_batch = torch.FloatTensor(np.concatenate(batch.reward).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_candidates = [c for s, c in zip(batch.next_state, batch.next_cand) if s is not None]
        next_state_emb_batch = self.gcn_net(non_final_next_states)
        next_candidates_batch = self.padding(non_final_next_candidates)
        next_candidates_emb_batch = self.gcn_net.embedding(next_candidates_batch)

        q_eval = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False)
        best_actions = torch.gather(input=next_candidates_batch, dim=1,
                                    index=self.policy_net(next_state_emb_batch, next_candidates_emb_batch).argmax(dim=1).view(
                                        len(non_final_next_states), 1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)
        q_target = torch.zeros((batch_size, 1), device=self.device)
        q_target[non_final_mask] = self.target_net(next_state_emb_batch, best_actions_emb, choose_action=False).detach()
        q_target = reward_batch + gamma * q_target

        errors = (q_eval - q_target).detach().cpu().squeeze().numpy()
        self.memory.update(indices, errors)

        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.gcn_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def padding(self, candidates, max_len=25):
        padded_candidates = [c + [self.padding_id] * (max_len - len(c)) for c in candidates]
        return torch.LongTensor(np.array(padded_candidates).astype(int)).to(self.device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LAST_FM', help='Name of dataset')
    parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--max_turn', type=int, default=15, help='Max number of turns')
    parser.add_argument('--episode_length', type=int, default=500, help='Number of turns per episode')
    parser.add_argument('--memory_size', type=int, default=200, help='Size of replay memory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DQN training')
    parser.add_argument('--hidden_size', type=int, default=100, help='Size of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='L2 regularization coefficient')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--eval_num', type=int, default=1, help='Number of evaluations')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--log_dir', type=str, default='log', help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save models')
    parser.add_argument('--prioritized', action='store_true', help='Use prioritized experience replay')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_class = ENVIRONMENTS[args.dataset]
    env = env_class(args.seed)

    state_size = env.state_size
    action_size = env.action_size

    gcn_net = GraphEncoder(env.num_users, env.num_items, env.num_relations, state_size, action_size).to(device)

    memory = PrioritizedReplayMemory(args.memory_size) if args.prioritized else ReplayMemory(args.memory_size)
    agent = DQNAgent(device, memory, state_size, action_size, args.hidden_size, gcn_net, args.learning_rate, args.l2_norm, env.padding_id)

    for episode in range(args.num_epoch):
        state, info = env.reset()
        for t in range(args.episode_length):
            action, sorted_actions = agent.select_action(state, info['candidates'], info['action_space'], is_last_turn=(t == args.max_turn - 1))
            next_state, reward, done, info = env.step(action.item())
            if done:
                next_state = None
            agent.memory.push(state, action, next_state, reward, info['candidates'])
            state = next_state
            agent.optimize_model(args.batch_size, args.gamma)
            if done:
                break

        if episode % args.eval_num == 0:
            evaluate_dqn(env, agent, args.max_turn)
            torch.save(agent.policy_net.state_dict(), os.path.join(args.model_dir, f'policy_net_{episode}.pth'))
            torch.save(agent.target_net.state_dict(), os.path.join(args.model_dir, f'target_net_{episode}.pth'))
