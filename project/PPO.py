import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.max_buffer_size = 100000
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        
    
    
    def clear(self):
        length = len(self.rewards)
        if(length > self.max_buffer_size):
            self.actions = self.actions[length-self.max_buffer_size:]
            self.states[:] = self.states[length-self.max_buffer_size:]
            self.logprobs[:] = self.logprobs[length-self.max_buffer_size:]
            self.rewards[:] = self.rewards[length-self.max_buffer_size:]
            self.state_values[:] = self.state_values[length-self.max_buffer_size:]
            self.is_terminals[:] = self.is_terminals[length-self.max_buffer_size:]
    
    def get_batches(self, batch_size, rewards):
        data_size = len(self.rewards)
        action_batch = []
        states_batch = []
        logprobs_batch = []
        rewards_batch = []
        state_values_batch = []
        is_terminals_batch = []
        idx_arr = np.random.choice(range(0,data_size), size= batch_size)
        for idx in idx_arr:
            action_batch.append(self.actions[idx])
            states_batch.append(self.states[idx])
            logprobs_batch.append(self.logprobs[idx])
            rewards_batch.append(rewards[idx])
            state_values_batch.append(self.state_values[idx])
            is_terminals_batch.append(self.is_terminals[idx])
        
        return action_batch, states_batch, logprobs_batch, rewards_batch, state_values_batch, is_terminals_batch
            


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, max_steer):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.max_steer = max_steer
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init ** 2).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 512),
                            nn.Tanh(),
                            nn.Linear(512, 256),
                            nn.Tanh(),
                            nn.Linear(256, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.max_steer*self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action = action.clamp(-self.max_steer,self.max_steer)
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def act_eval(self, state):
        if self.has_continuous_action_space:
            state = torch.FloatTensor(state).to(device)
            action_mean = self.max_steer * self.actor(state)
            action_mean = action_mean.clamp(-self.max_steer, self.max_steer)
        return action_mean.detach()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.max_steer*self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, max_steer,action_std_init=0.6, batch_size = 64):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,max_steer).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,max_steer).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state_tensor)

            self.buffer.states.append(state_tensor.cpu())
            self.buffer.actions.append(action.cpu())
            self.buffer.logprobs.append(action_logprob.cpu())
            self.buffer.state_values.append(state_val.cpu())

            return action.detach().cpu().numpy().flatten()
        else: 
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()
    


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalize rewards
        # rewards = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

        old_action_batch, old_states_batch, old_logprobs_batch, rewards_batch, state_values_batch, is_terminals_batch = self.buffer.get_batches(self.batch_size, rewards=rewards)
        
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        old_states = torch.squeeze(torch.stack(old_states_batch, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(old_action_batch, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(old_logprobs_batch, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(state_values_batch, dim=0)).detach().to(device)

        

        # Compute advantages
        advantages = rewards_batch.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate actions
            logprobs, state_values_new, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values_new = torch.squeeze(state_values_new)
            # Compute ratios
            ratios = torch.exp(logprobs - old_logprobs)
            # Compute PPO loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values_new, rewards_batch) - 0.01 * dist_entropy
            # Optimize
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

