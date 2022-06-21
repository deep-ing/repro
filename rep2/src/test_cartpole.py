import numpy as np

def test_agent():
    import gym
    from envs import CartPoleEnv
    # env = gym.make('CartPole-v1')
    env = CartPoleEnv()

    total_rewards = []
    len_episodes = []

    for i in range(100):
        env.reset()
        env.render(mode='rgb_array')
        done = False
        len_episode = 0
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render(mode='rgb_array')
            
            len_episode += 1
            total_reward += reward

            print(len_episode)
            if len_episode >= 200:
                break
        total_rewards.append(total_reward)
        len_episodes.append(len_episode)

    print('Reward (mean/std):', np.mean(total_rewards), np.std(total_rewards))
    print('Episode length (mean/std):', np.mean(len_episodes), np.std(len_episodes))

if __name__ == "__main__":
    test_agent()