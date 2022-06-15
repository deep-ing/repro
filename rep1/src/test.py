import numpy as np

def test_agent():
    from envs import SimpleMazeEnv
    from envs import MazeEnv
    from envs import CatcherEnv

    env_map = {1: SimpleMazeEnv, 2: MazeEnv, 3: CatcherEnv}
    user_input = int(input("====================\nChoose env (type the number):\n\
1) Simple Maze\n\
2) Maze\n\
3) Catcher\n\
====================\n>> "))
    try:
        env = env_map[user_input](rng=np.random.RandomState(0), render_mode='rgb_array')
    except:
        env = env_map[user_input](render_mode='rgb_array')

    total_rewards = []
    len_episodes = []

    for i in range(100):
        env.reset()
        env.render()
        done = False
        len_episode = 0
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            
            len_episode += 1
            total_reward += reward

            print(len_episode)
            if len_episode >= 50:
                break
        total_rewards.append(total_reward)
        len_episodes.append(len_episode)

    print('Reward (mean/std):', np.mean(total_rewards), np.std(total_rewards))
    print('Episode length (mean/std):', np.mean(len_episodes), np.std(len_episodes))

if __name__ == "__main__":
    test_agent()