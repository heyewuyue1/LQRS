import random
import torch.autograd
from preprocessor.sparkplanpreprocessor import SparkPlanPreprocessor
from tree_conv.net import ActorNet, CriticNet
from config import PPOConfig, EnvironmentConfig
import torch.optim as optim
import torch.nn.functional as F
import torch
from utils.logger import setup_custom_logger
from utils import rl_utils
import numpy as np
import os

logger = setup_custom_logger("LQRS")

class LQRS:
    def __init__(self, deterministic):
        self.transition_dict = {
            'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []
        }
        self.preprocessor = SparkPlanPreprocessor()
        self.lead_dim = EnvironmentConfig.table_num - 2
        self.hint_dim = EnvironmentConfig.table_num
        self.actor = ActorNet(self.preprocessor.feature_dim, 2 + self.lead_dim + 1)
        self.critic = CriticNet(self.preprocessor.feature_dim)
        if PPOConfig.optimizer == 'Adam':
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPOConfig.actor_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=PPOConfig.critic_learning_rate)
        if PPOConfig.device == 'cuda':
            self.actor.cuda()
            self.critic.cuda()
        self.episode_cnt = 0
        self.return_list = []
        self.last_execution_time = -1
        self.last_num_shuffles = 0
        self.act_list = self.generate_act_dict(EnvironmentConfig.table_num)
        self.use_shaping_reward = PPOConfig.use_shaping_reward
        # testing mode
        self.deterministic = deterministic
        if self.deterministic:
            self.load(PPOConfig.save_name)
            logger.info(f"LQRS entered test mode, load ckpt from {PPOConfig.save_name}")
        else:
            if os.path.exists('/home/xxxxxxxx/LQRS_release/models/checkpoints/' + PPOConfig.save_name + '_actor'):
                self.load(PPOConfig.save_name)
                logger.info(f"Found existing checkpoint at {PPOConfig.save_name}, loading it")
            else:
                logger.info(f"No existing checkpoint found at {PPOConfig.save_name}, starting fresh")
            logger.info(f"LQRS entered train mode, saving ckpt to {PPOConfig.save_name}")

    def generate_act_dict(self, table_num):
        act_list = []
        act_list.append(('cbo', (1,)))
        act_list.append(('cbo', (0,)))
        for i in range(table_num - 2):
            act_list.append(('lead', (i + 2,)))
        act_list.append(('no-op', ()))
        logger.info(f"Generated act_dict for {table_num} tables, dim: {len(act_list)}")
        return act_list
    
    def generate_mask(self, current_table_num, runtime, trigger_time):
        mask = torch.zeros(2 + self.lead_dim + 1)
        logger.debug(f"runtime: {runtime}, trigger_time: {trigger_time}")

        # mask in cbo
        if (not runtime) and trigger_time == 1:
            mask[0] = 1
            mask[1] = 1
            return mask
        if (self.episode_cnt >= 480 and runtime) or self.episode_cnt >= 960 or self.deterministic:
            for i in range(2, current_table_num):
                mask[i] = 1
                
        mask[-1] = 1
        return mask

    def act(self, plan, executed_row_counts, runtime, trigger_time, num_shuffles):
 
        feature_vector = self.preprocessor.transform(plan, executed_row_counts)
        
        current_table_num = plan.count("Join") + 1

        action_probs = torch.exp(self.actor([feature_vector])).cpu().detach().squeeze()
        action_probs = torch.clamp(action_probs, min=1e-8, max=1e8)

        action_probs /= action_probs.sum()

        mask = self.generate_mask(current_table_num, runtime, trigger_time)

        masked_probs = action_probs * mask

        if self.deterministic:
            action_index = masked_probs.argmax().item()  # get action with most chances
        else:
            action_index = torch.multinomial(masked_probs, num_samples=1).item()

        selected_action = self.act_list[action_index]
        assert selected_action[1] == () or selected_action[1][0] < current_table_num

        # append to transition_dict
        # self.transition_dict after first append
        # {
        #     "states": [(s1, e1)],
        #     "actions": [(l1,r1)],
        #     "next_states": [None],
        #     "rewards": [0],
        #     "dones": [0]
        # }
        #
        # self.transition_dict after second append
        # {
        #     "states": [(s1, e1), (s2, e2)],
        #     "actions": [(l1, r1), (l1, r1)],
        #     "next_states": [(s2, e2), None],
        #     "rewards": [r2, 0],
        #     "dones": [0, 0]
        # }
        self.transition_dict["states"].append((plan, executed_row_counts))
        self.transition_dict["actions"].append(action_index)
        self.transition_dict["next_states"].append(None)
        self.transition_dict["rewards"].append(0)
        self.transition_dict["dones"].append(0)
        if len(self.transition_dict["next_states"]) > 1:
            self.transition_dict["next_states"][-2] = (plan, executed_row_counts)
            if self.use_shaping_reward:
                self.transition_dict["rewards"][-2] = self.calculate_shaping_reward(num_shuffles)
                self.last_num_shuffles = num_shuffles
            else:
                self.transition_dict["rewards"][-2] = 0
        return selected_action

    def replay_if_can(self, last_execution_time, plan_time):
        self.last_num_shuffles = 0
        if last_execution_time == self.last_execution_time: 
            return 0
        if len(self.transition_dict["states"]) == 0:
            logger.debug("Can't replay now because transition_dict is empty")
            return 0
        logger.info("Finished an episode, try to replaying...")
        logger.info(f"Optimization Cost: {plan_time}ms")
        # Log actions
        action_list = []
        for i in self.transition_dict["actions"]:
            if self.act_list[i][0] == 'no-op':
                action_list.append("No-op")
            elif self.act_list[i][0] == 'lead':
                action_list.append(f"Lead({self.act_list[i][1][0]})")
            elif self.act_list[i][0] == 'hint':
                action_list.append(f"Broadcast({self.act_list[i][1][0]})")
            elif self.act_list[i][0] == 'cbo':
                action_list.append(f"CBO On" if self.act_list[i][1][0] == 1 else "CBO Off")
        logger.info(f"Action Chain: {'->'.join(action_list)}")
        if self.deterministic:
            self.transition_dict = {
                'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []
            }
            return 0
        self.episode_cnt += 1
        # self.transition_dict after last append
        # {
        #     "states": [(s1, e1), (s2, e2),...,(sn, en)],
        #     "actions": [(l1, r1), (l1, r1),...,(ln,rn)],
        #     "next_states": [(s2, e2), (s3, e3),..., (sn, en), None],
        #     "rewards": [r2, r3, ...,  0],
        #     "dones": [0, 0,..., 0]
        # }

        # self.transition_dict after preprocss
        # {
        #     "states": [(s1, e1), (s2, e2),...,(sn, en)],
        #     "actions": [(l1, r1), (l1, r1),...,(ln,rn)],
        #     "next_states": [(s2, e2), (s3, e3),..., (sn, en), (s1, e1)],
        #     "rewards": [r2, r3, ..., latency],
        #     "dones": [0, 0,..., 1]
        # }
        self.transition_dict["next_states"][-1] = self.transition_dict["states"][0]
        if last_execution_time / 1e9 < 1:
            logger.warning(f"Last query execution failed/timeout, setting the punishment to {PPOConfig.fail_punish}")
            self.transition_dict["rewards"][-1] = PPOConfig.fail_punish
        else:
            self.transition_dict["rewards"][-1] = -np.sqrt(last_execution_time / 1e9)
        self.transition_dict['dones'][-1] = 1

        # Get action
        actions = torch.tensor(self.transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(PPOConfig.device)

        # Calculate return
        logger.info(f"Rewards: {self.transition_dict['rewards']}")
        return_ = self.transition_dict["rewards"][-1]
        for reward in list(reversed(self.transition_dict['rewards']))[1:]:
            return_ = return_ * PPOConfig.gamma + reward
        logger.info(f"Total reward of {self.episode_cnt}th episode: {return_}")
        self.return_list.append(return_)
        self.writer.add_scalar("Return/Episode", return_, global_step=self.episode_cnt)
        
        # Get reward
        rewards = torch.tensor(self.transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(PPOConfig.device)

        # Refresh last_execution_time
        self.last_execution_time = last_execution_time

        # Get states, maybe optimize states and next_states into a single list
        raw_states = self.transition_dict["states"]
        states = [self.preprocessor.transform(plan, erc) for plan, erc in raw_states]
        raw_next_states = self.transition_dict["next_states"]
        next_states = [self.preprocessor.transform(plan, erc) for plan, erc in raw_next_states]
        # and dones
        dones = torch.tensor(self.transition_dict['dones'], dtype=torch.float).view(-1, 1).to(PPOConfig.device)

        # Clear transition_dict before training to avoid errors in train affects next query
        self.transition_dict = {
            'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []
        }

        # Calculate td_target
        td_target = rewards + PPOConfig.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

         # Calculate advantage
        advantage = rl_utils.compute_advantage(PPOConfig.gamma, PPOConfig.lmbda, td_delta.cpu()).to(PPOConfig.device)
        old_log_probs = self.actor(states).gather(1, actions).detach()
        for i in range(PPOConfig.epochs):
            combined = list(zip(states, actions, old_log_probs, advantage, td_target))
            random.shuffle(combined)

            shuffled_states, shuffled_actions, shuffled_old_log_probs, shuffled_advantage, shuffled_td_target = zip(*combined)
            states = list(shuffled_states)
            actions = torch.tensor(shuffled_actions).view(-1,1).to(PPOConfig.device)
            old_log_probs = torch.tensor(shuffled_old_log_probs).view(-1,1).to(PPOConfig.device)
            advantage = torch.tensor(shuffled_advantage).view(-1,1).to(PPOConfig.device)
            td_target = torch.tensor(shuffled_td_target).view(-1,1).to(PPOConfig.device)
            
            log_probs = self.actor(states).gather(1, actions)
            ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -10, 10))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - PPOConfig.epsilon, 1 + PPOConfig.epsilon) * advantage  # The PPO clip
            probs = torch.exp(torch.clamp(log_probs, -10, 10))
            entropy = -torch.sum(probs * log_probs, dim=1).mean()

            actor_loss = torch.mean(-torch.min(surr1, surr2)) - PPOConfig.entropy_coef * entropy # PPO loss+entropy normalization
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        if self.episode_cnt % PPOConfig.save_interval== 0:
            self.save(PPOConfig.save_name)
        return 0 
    
    def calculate_shaping_reward(self, num_shuffles):
        return max(0, num_shuffles - self.last_num_shuffles) * PPOConfig.shuffle_punishment

    def load(self, name):
        self.critic.load_state_dict(torch.load(f"models/checkpoints/{name}_critic"))
        self.actor.load_state_dict(torch.load(f"models/checkpoints/{name}_actor"))

    def save(self, name):
        torch.save(self.critic.state_dict(), f"models/checkpoints/{name}_critic")
        torch.save(self.actor.state_dict(), f"models/checkpoints/{name}_actor")
