import pickle
import argparse
from collections import defaultdict
import random

NEGATIVE_REWARD = -10
POSITIVE_REWARD = 10
STEP_REWARD = -1
GAMMA = 0.95
THRESHOLD = 1e-4
P = 0
Q = 0

def in_bounds(x,y):
    return -1<x<4 and -1<y<4

def get_xy(position): #Converts a position (1–16) to its (x, y) position in the grid
    row = (position - 1) // 4  
    col = (position - 1) % 4
    return col, row

def is_inbetween(b1: int, b2: int, r: int) -> bool: #True if center of r is in between line connecting b1 and b2
    b1_x, b1_y = get_xy(b1)
    b2_x, b2_y = get_xy(b2)
    r_x , r_y = get_xy(r)
    return (r_y - b1_y + b1_x) * (b2_x - b1_x) == (b2_y - b1_y) * r_x if b1 != b2 else False

def get_transition(state: tuple, action : int, opp_policy : dict, p :float, q :float) -> list:
    b1, b2, r, ball_owner = state
    transitions = []
    delta_list = [(-1,0), (1,0) , (0,-1), (0,1)] #(x,y) for L,R,U,D
    r_prob_list = opp_policy.get(state, [0.25]*4)

    for i, r_prob in enumerate(r_prob_list):
        rx, ry = get_xy(r)
        rx_new = rx + delta_list[i][0]
        ry_new = ry + delta_list[i][1]
        new_r = 4 * ry_new + rx_new + 1 if in_bounds(rx_new, ry_new) else r
        
        if 0 <= action <=3:
            x, y = get_xy(b1)
            x_new = x + delta_list[action][0]
            y_new = y + delta_list[action][1]

            success_prob = 1-2*p if ball_owner == 1 else 1-p

            if not in_bounds(x_new, y_new):
                    transitions.append((r_prob, (b1, b2, new_r, 0), NEGATIVE_REWARD, True))  # fell out of bounds
                    continue
            
            new_b1 = 4 * y_new + x_new + 1 # convert new x,y to new position on grid
            
            if ball_owner == 1: #Movement Succeeds
                new_state =  (new_b1, b2, new_r ,1) 
                tackling = (new_b1 == new_r or (new_r == b1 and new_b1 == r))
                if tackling:
                    transitions.append((success_prob * r_prob * 0.5, new_state,STEP_REWARD, False))
                    transitions.append((success_prob * r_prob * 0.5, (new_b1, b2, new_r ,0)  , NEGATIVE_REWARD, True))
                else:
                    transitions.append((success_prob * r_prob, new_state, STEP_REWARD, False))

            else:
                new_state = (new_b1, b2, new_r, 2)
                transitions.append((success_prob * r_prob, new_state , STEP_REWARD, False))

            if ball_owner == 1: #Movement Fails
                new_state = (b1, b2, new_r, 0)
                transitions.append(((1-success_prob) * r_prob, new_state, NEGATIVE_REWARD, True)) #-1 reward, fail with ball, episode ends
            else:
                transitions.append(((1-success_prob) * r_prob, (b1, b2, new_r, ball_owner), STEP_REWARD, False)) # 0 reward for failing with no ball

        if 4 <= action <=7:
            x, y = get_xy(b2)
            x_new = x + delta_list[action-4][0]
            y_new = y + delta_list[action-4][1]

            success_prob = 1-2*p if ball_owner == 2 else 1-p

            if not in_bounds(x_new, y_new):
                    transitions.append((r_prob, (b1, b2, new_r, 0), NEGATIVE_REWARD, True))  # fell out of bounds
                    continue
            
            new_b2 = 4 * y_new + x_new + 1 # convert new x,y to new position on grid
            
            if ball_owner == 2: #Movement Succeeds
                new_state =  (b1, new_b2, new_r ,2) 
                tackling = (new_b2 == new_r or (new_r == b2 and new_b2 == r))
                if tackling:
                    transitions.append((success_prob * r_prob * 0.5, new_state , STEP_REWARD, False))
                    transitions.append((success_prob * r_prob * 0.5, (b1, new_b2, new_r ,0) , NEGATIVE_REWARD, True))
                else:
                    transitions.append((success_prob * r_prob, new_state, STEP_REWARD, False))

            else:
                new_state = (b1, new_b2, new_r, 1)
                transitions.append((success_prob * r_prob, new_state , STEP_REWARD, False))

            if ball_owner == 2: #Movement Fails
                new_state = (b1, b2, new_r, 0)
                transitions.append(((1-success_prob) * r_prob, new_state, NEGATIVE_REWARD, True)) #-1 reward, fail with ball, episode ends
            else:
                new_state = (b1, b2, new_r, 1)
                transitions.append(((1-success_prob) * r_prob, new_state, STEP_REWARD, False)) # 0 reward for failing with no ball
        
        if action == 8:
            b1_x, b1_y = get_xy(b1)
            b2_x, b2_y = get_xy(b2)

            pass_prob = q - 0.1 * max(abs(b1_x - b2_x), abs(b1_y - b2_y))
            pass_prob = pass_prob / 2 if is_inbetween(b1, b2, new_r) else pass_prob 

            transitions.append((pass_prob * r_prob, (b1, b2, new_r, 3 - ball_owner), 0, False)) #pass successful

            transitions.append(((1-pass_prob) * r_prob, (b1, b2, new_r, 0), NEGATIVE_REWARD, True)) #pass failed

        if action == 9:
            x,y = get_xy(b1) if ball_owner == 1 else get_xy(b2)
            shoot_prob =  q - 0.2 * (3 - x)

            shoot_prob = shoot_prob / 2 if new_r in [7,8,11,12] else shoot_prob

            transitions.append((shoot_prob * r_prob, (b1, b2, new_r, 3), POSITIVE_REWARD, True)) #shoot successful

            transitions.append(((1-shoot_prob) * r_prob, (b1, b2, new_r, 0), NEGATIVE_REWARD, True)) #shoot failed
    return transitions

def is_terminal(state): 
    return state[3] == 0 or state[3] == 3 #0 for possession lost, 3 for goal

def value_iteration(all_states, opp_policy, p, q):
    V = defaultdict(float) #Value function
    policy = {}
    iteration = 0

    while True:
        delta = 0 # tracks change in V[state] in current iteration
        

        for state in all_states:
            if is_terminal(state): #skip terminal states
                continue
            max_value = float('-inf')
            best_action = None

            for action in range(10): 
                transitions = get_transition(state, action, opp_policy ,p ,q)
                value = 0
                for prob, next_state, reward, done in transitions:
                    value += prob * (reward + GAMMA * V[next_state])
                if value > max_value:
                    max_value = value
                    best_action = action
            
            delta = max(delta, abs(V[state] - max_value))
            V[state] = max_value
            policy[state] = best_action
        
        iteration += 1
        # print(iteration)
        if delta < THRESHOLD:
            break
    print("Value iteration complete")
    return V, policy

def simulate(policy, start_state, opp_policy , p ,q):
    state = start_state
    path = [state]
    
    while not is_terminal(state):
        action = policy[state]
        transitions = get_transition(state, action, opp_policy, p, q)
        probs = [t[0] for t in transitions]
        
        selected_transition = random.choices(transitions, weights = probs, k = 1)[0]
        _, next_state, reward, done = selected_transition
        state = next_state
        path.append(state)

        if done:
            break

    return path

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=float, default=0.2, help="value of p")
    parser.add_argument('-q', type=float, default=0.7, help="value of q")
    parser.add_argument('-policy', type= str, default="random", help="select policy random/greedy/bus")
    args = parser.parse_args()

    # for i in range(20):
    #     choice = random.choices([1,0], weights = [0.6 , 0.4], k = 1)[0]
    #     print(f"{i}th result = {choice}")

    with open("all_states.pkl", "rb") as f:
        all_states = pickle.load(f)

    with open(f"pickle_policies/{args.policy}_policy.pkl", "rb") as f:
        opponent_policy = pickle.load(f)
    p = args.p
    q = args.q
    V, policy = value_iteration(all_states, opponent_policy, p , q) 
    
    wins= 0
    loss = 0
    matches = 10000
    for i in range(matches):
        simulations = simulate(policy, (5,9,8,1), opponent_policy, p, q)
        if simulations[-1][-1] == 3:
            wins += 1

    print(f"wins: {wins}, loss: {matches - wins}")



