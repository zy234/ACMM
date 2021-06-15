import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import random
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'attention_mask', 'mask_num'))

BATCH_SIZE = 48
PAD_SIZE = 256

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """保存变换"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Envirment(nn.Module):
    def __init__(self, encoder, cls):
        super(Envirment, self).__init__()

        self.encoder = encoder
        self.cls = cls

    def forward(self, input_ids, attention_mask, y, predict=False):
        # 如果传入的是一个句子
        if len(input_ids.size()) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            y = y.unsqueeze(0)

        bs, seqlen = input_ids.size()

        encoder_out = self.encoder(input_ids, attention_mask)
        logits = self.cls(encoder_out['last_hidden_state'])
        loss = F.cross_entropy(logits.view(-1, 5), y.view(-1), reduction='none').view(bs, seqlen)

        if predict:
            return logits

        return loss.mean(dim=-1), encoder_out['last_hidden_state']


class A2CModel(nn.Module):
    def __init__(self):
        super(A2CModel, self).__init__()
        self.embedding = nn.Embedding(30522, 48)
        self.h_state = nn.Linear(768, 48)
        self.lstm = nn.LSTM(48, 24, batch_first=True)
        self.fc1 = nn.Linear(48, 24)
        self.fc2 = nn.Linear(24, 12)
        self.action = nn.Linear(12, 2)
        self.value = nn.Linear(12, 1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, hidden_state=None):
        if hidden_state is not None:
            hidden_state = self.dropout(hidden_state)
            x = F.relu(self.h_state(hidden_state))
        else:
            x = self.embedding(x)
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        x, _ = self.lstm(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = x.squeeze()

        action_probs = F.softmax(self.action(x), dim=-1)
        state_values = self.value(x)

        state_values = state_values.squeeze().mean(dim=-1)
        return action_probs, state_values

    def get_new_ids(self, input_ids, seqlens, model):

        action_probs, _ = model(input_ids)

        # m = Categorical(action_probs)
        # actions = m.sample()
        # 按照分布选择mask 索引
        total_mask_num = 0

        mask_probs = action_probs[:, :, 0]
        # mask_probs = torch.softmax(mask_probs, dim=-1)
        for index, seqlen in enumerate(seqlens):
            mask_num = int(seqlen * 0.2)
            total_mask_num += mask_num

            total_prob = mask_probs[index][:seqlen].sum()
            mask_prob = mask_probs[index][:seqlen] / total_prob

            mask_index = torch.arange(seqlen)
            mask_pos = np.random.choice(mask_index, mask_num, replace=False, p=mask_prob.detach().cpu().numpy())
            input_ids[index][mask_pos] = 103

        return input_ids, total_mask_num


def pad_func(*input):

    res = []
    for element in input:
        if len(element) > PAD_SIZE:
            element = element[:PAD_SIZE]
            res.append(element)
        else:
            padd = torch.zeros(PAD_SIZE - len(element)).type_as(element)
            element = torch.cat((element, padd))
            res.append(element)

    return res

def RLtrainIters_v2(env, state, ActorCriticModel, attention_mask, y, seqlen, memory):
    '''
    :param env: bert model
    :param state: input ids
    :param ActorCriticModel: A2Cmodel
    :param num_episodes: seqlen
    :param gamma:
    :param optimiezer: optimizer
    :param scheduler:
    :return:
    '''
    ActorCriticModel.to(device)

    with torch.no_grad():
        base_score, hidden_state = env(state, attention_mask, y)

    # 假设传入一个句子, 真实长度
    state = state[:seqlen]
    attention_mask = attention_mask[:seqlen]
    y = y[:seqlen]
    hidden_state = hidden_state.squeeze()[:seqlen].detach()

    action_probs, state_value = ActorCriticModel(state, hidden_state=hidden_state)  # 传入一个句子 seqlen, 2; 1
    # 多次采样action 及对应的log prob, 将其存入memory
    mask_probs = action_probs[:, 0]
    mask_num = int(seqlen * 0.2)

    total_prob = mask_probs[:seqlen].sum()
    mask_prob = mask_probs[:seqlen] / total_prob
    mask_index = torch.arange(seqlen)

    for i in range(1):
        # m = Categorical(action_probs)
        # action = m.sample()
        mask_pos = np.random.choice(mask_index, mask_num, replace=False, p=mask_prob.detach().cpu().numpy())

        action = torch.ones_like(state)
        action[mask_pos] = 0
        action = torch.where(y == 0, action, torch.tensor(1).type_as(action))
        action[0] = 1
        action[seqlen - 1] = 1
        # next state
        next_state = state.clone()
        next_state = torch.where(action == 0, torch.tensor(103).type_as(next_state), next_state)
        # new state score
        with torch.no_grad():
            new_score, _ = env(next_state, attention_mask, y)
        # reward
        # mask_num = mask_pos.sum().unsqueeze(0)
        # print(mask_num / seqlen)
        reward = new_score / base_score - 1
        if reward < 0:
            reward = torch.tensor(0).type_as(new_score)
        rewards = torch.zeros_like(mask_prob)
        rewards[:] = reward

        # 存入memory
        # if len(save_state) < PAD_SIZE:

        s_state, s_action, s_rewards, s_attention_mask = pad_func(state, action, rewards, attention_mask)
        memory.push(s_state, s_action, s_rewards, s_attention_mask, torch.tensor([mask_num]))


def RL_optimize(ActorCriticModel, env, memory, optimizer, pri=False):
    # 模型优化
    if len(memory) < BATCH_SIZE:
        return 0, 0
    # 取出一个batch的数据
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # 转化为batch tensor
    state = torch.cat(batch.state).view(-1, PAD_SIZE)
    action = torch.cat(batch.action).view(-1, PAD_SIZE)
    reward = torch.cat(batch.reward).view(-1, PAD_SIZE)
    attention_mask = torch.cat(batch.attention_mask).view(-1, PAD_SIZE)
    mask_num = torch.cat(batch.mask_num)

    with torch.no_grad():
        hidden_state = env.encoder(state, attention_mask)['last_hidden_state']
    action_probs, state_value = ActorCriticModel(state)
    # advantage
    state_values = torch.where(reward == 0, reward, state_value.unsqueeze(1).repeat(1, PAD_SIZE))
    advantage = reward - state_values
    # loss function
    critic_loss = F.smooth_l1_loss(state_values, reward.detach())
    # action对应的log prob
    m = Categorical(action_probs)
    log_probs = m.log_prob(action)
    actor_loss = (-log_probs * advantage.detach()).mean()
    loss = critic_loss + actor_loss

    # 更新critic
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = reward.sum()
    mask_num =mask_num.sum().item()

    return total_reward, mask_num