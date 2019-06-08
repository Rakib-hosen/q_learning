import numpy as np
import gym

#...................make a enviorment ......................
env = gym.make("MountainCar-v0")
env.reset()


LEARNING_RATE = 0.01
DISCOUNT = 0.9
EPISODES = 25000

#...................make a random q_table ...................

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)                                                  # size of the q_table
discrete_os_size_win = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE           # space between high and low obsarbation state
q_table = np.random.uniform(low= -2, high = 0 ,size =( DISCRETE_OS_SIZE + [env.action_space.n]))           # make a random q_table values 0-(-2) discretesize +action in list

#.........................convert the continus state to discrete state..................

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_size_win
    return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset())
done = False

while not done:
    action = np.argmax([discrete_state])
    new_state , reward , done , _=env.step(action)
    new_discrete_state = get_discrete_state( new_state )
    env.render()
    if not done:
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state,(action,)]
        new_q = (1 - LEARNING_RATE)*current_q+LEARNING_RATE(reward+ DISCOUNT * max_future_q)
        q_table[discrete_state,(action,)]= new_q
    elif new_state[0]>=env.goal_position:
        q_table[discrete_state,(action,)]=0
    
    discrete_state = new_discrete_state
env.close()
