

def test_envdebugging():
    import os
    import sys
    PROJECT_PATH = os.getcwd()
    SOURCE_PATH = os.path.join(
        PROJECT_PATH,"src"
    )
    sys.path.append(SOURCE_PATH)
    from envs import SimpleMazeEnv

    env = SimpleMazeEnv()

    env.reset()
    env.render()
    user_input = None
    action_map = {'i': 0, 'k': 1, 'j': 2, 'l': 3}
    while user_input != 'stop':
        user_input = input('input i/k/j/l:')
        if user_input in action_map:
            action = action_map[user_input]
        else:
            break

        obs, reward, done, info = env.step(action)
        env.render()


if __name__ == "__main__":
    test_envdebugging()