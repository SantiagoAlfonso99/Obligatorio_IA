from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

class ContinuousMountainCarEnvExtended(Continuous_MountainCarEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = 1000
        self.actual_steps = 0
        
    def step(self, action):
        self.actual_steps += 1
        obs, reward, done, _, _ = super().step(action)
        if done:
            reward = 100
        else:
            reward = -1
        done = done or self.actual_steps >= self.max_steps # asegurarnos que el episodio no dure más de 1000 pasos
        return obs, reward, done, False, {}
    
    def reset(self, *args, **kwargs):
        self.actual_steps = 0
        return super().reset(*args, **kwargs)
    
    
