

def test_envdebugging():
    from src.envs import SimpleMazeEnv
    from src.envs import MazeEnv

    env_map = {1: SimpleMazeEnv, 2: MazeEnv}
    user_input = int(input("====================\nChoose env (type the number):\n1) Simple Maze\n2) Maze\n====================\n>> "))
    env = env_map[user_input]()

    env.reset()
    env.render()
    user_input = None
    action_map = {'i': 0, 'k': 1, 'j': 2, 'l': 3}
    while user_input != 'stop':
        user_input = input('input i/k/j/l/stop: ')
        if user_input in action_map:
            action = action_map[user_input]
        else:
            break

        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
        env.render()


if __name__ == "__main__":
    test_envdebugging()