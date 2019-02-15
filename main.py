import argparse
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agents.dqn_agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e' ,'--epochs-count', type=int, default=1000, help='epochs count')
    parser.add_argument('-t' ,'--target_update', type=int, default=20, help='epochs between target network updates')
    parser.add_argument('-es' ,'--eps-start', type=float, default=0.9, help='random action start probability')
    parser.add_argument('-ee' ,'--eps-end', type=float, default=0.05, help='random action end probability')
    parser.add_argument('-ed' ,'--eps-decay', type=int, default=20000, help='random action probability decay')
    parser.add_argument('-bs' ,'--batch-size', type=int, default=256, help='batch_size')
    parser.add_argument('-g' ,'--gamma', type=float, default=0.7, help='gamma')
    parser.add_argument('-l' ,'--learning-rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--no-render', type=bool, help='display game')
    args = parser.parse_args()
    return args

def step_generator(env, agent):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        yield state, action, next_state, reward
        state = next_state

def main(args):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    agent = DQNAgent((3, 240, 256),
                     lr=args.learning_rate,
                     eps_start=args.eps_start,
                     eps_end=args.eps_end,
                     eps_decay=args.eps_decay,
                     batch_size=args.batch_size,
                     num_of_actions=7)

    for i_episode in range(args.epochs_count):

        total_reward = 0

        for step, (state, action, next_state, reward) in enumerate(step_generator(env, agent)):

            if not args.no_render:
                env.render()
            agent.push_transition(state, action, next_state, reward)

            total_reward += reward

        print('total reward: {}'.format(total_reward))
        agent.update_qnet()

        if i_episode % args.target_update == 0:
            agent.update_target_net()

        if i_episode != 0 and i_episode % 200 == 0:
            agent.save_target_net('models/model_{}_total_reward_{}'.format(i_episode, total_reward))

    env.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
