import random
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, x_space, vel_space, actions):
        self.actions = actions
        self.x_space = x_space
        self.vel_space = vel_space
        self.q = np.zeros((len(x_space) + 1, len(vel_space) + 1, len(actions)))

    def next_action(self, obs, epsilon):
        explore = np.random.binomial(1, epsilon)
        if explore:
            action = random.choice(self.actions)
        else:
            action = self.actions[np.argmax(self.q[obs])]
        return action

    def train_agent(self, env, episodes, epsilon, gamma, alpha):
        rewards_per_episode = []
        episode = 0
        while episode < episodes:
            print(episode)
            obs,_ = env.reset()
            print(obs)
            done = False
            total_reward = 0
            step_count = 0
            while not done:
                state = self.get_state(obs)
                new_epsilon = self.linear_decay(episode, episodes, epsilon, .2)
                new_alpha = alpha
                action = self.next_action(state, new_epsilon)
                action_idx = self.actions.index(action)
                real_action = np.array([action])
                obs, reward, done, _, _ = env.step(real_action)
                next_state = self.get_state(obs)
                self.q[state][action_idx] = self.q[state][action_idx] + new_alpha * (reward + gamma * np.argmax(self.q[next_state]) - self.q[state][action_idx])
                state = next_state
                total_reward += reward
                step_count += 1
                env.render()
            episode+=1
            rewards_per_episode.append(total_reward)
            print('total_reward', total_reward)
            print('episode: ', episode)
        return rewards_per_episode

    def test_agent(self, env, episodes=10):
        obs,_ = env.reset()
        done = False
        while not done:
                state = self.get_state(obs)
                action = self.actions[np.argmax(self.q[state])]
                action_idx = self.actions.index(action)
                real_action = np.array([action])
                obs, reward, done, _, _ = env.step(real_action)
                env.render()

    def get_state(self, obs):
        x, vel = obs
        x_bin = np.digitize(x, self.x_space)
        vel_bin = np.digitize(vel, self.vel_space)
        return x_bin, vel_bin

    def linear_decay(self, episode, episodes, start_epsilon, min_epsilon):
        decay_rate = (start_epsilon - min_epsilon) / episodes
        return max(min_epsilon, start_epsilon - decay_rate * episode)

    def exponential_decay(self, episode, episodes, start_epsilon, min_epsilon):
        decay_rate = -np.log(min_epsilon / start_epsilon) / episodes
        return max(min_epsilon, start_epsilon * np.exp(-decay_rate * episode))