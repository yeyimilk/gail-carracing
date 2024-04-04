import numpy as np
import gym

class Env():
    """
    Test environment wrapper for CarRacing
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v1')
        self.env.seed(0)
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_repeat = 4
        self.img_stack = 4

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack

        stack_array = np.array(self.stack)
        stack_array = np.transpose(stack_array, axes=[1, 2, 0])
        stack_array = np.expand_dims(stack_array, axis=0)
        return stack_array

        # return np.array((self.stack))
        #return self.stack

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)

            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack

        stack_array = np.array(self.stack)
        stack_array = np.transpose(stack_array, axes=[1, 2, 0])
        stack_array = np.expand_dims(stack_array, axis=0)

        return stack_array, total_reward, done, die


    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
