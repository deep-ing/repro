def test_humanplay():
    import gym

    env = gym.make('CartPole-v1')

    env.reset()
    env.render()
    user_input = None
    action_map = {'j': 0, 'l': 1}
    while user_input != 'stop':
        user_input = input('input j/l/stop: ')
        if user_input in action_map:
            action = action_map[user_input]
        else:
            break

        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
        env.render()


if __name__ == "__main__":
    test_humanplay()