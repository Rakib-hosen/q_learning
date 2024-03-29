import gym
import numpy as np

env = gym.make("MountainCar-v0")
#env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95           #How important future action/reward over current reward
EPISODES = 25000
SHOW_EVERY =2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size =(env.observation_space.high- env.observation_space.low)/  DISCRETE_OS_SIZE   
q_table = np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))

epsilon =0.9
START_EPSILON_DACAYING = 1
END_EPSILON_DACAYING = EPISODES//2
epsilon_decay_value = epsilon / (END_EPSILON_DACAYING - START_EPSILON_DACAYING)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    if episode % SHOW_EVERY ==0:
        print(episode)
        render = True
    else:
        render =False
        
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    while not done:
        
        if np.random.random()>epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state ,reward ,done ,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()
            
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)] =new_q
        elif new_state[0]>= env.goal_position:
            q_table[discrete_state +(action,)]=0
            print(f"made it in episode {episode}")
        
        discrete_state = new_discrete_state
    if END_EPSILON_DACAYING >=episode >= START_EPSILON_DACAYING:
        epsilon -=epsilon_decay_value
        #print(a,action,new_state , reward ,done ,_)



env.close()

