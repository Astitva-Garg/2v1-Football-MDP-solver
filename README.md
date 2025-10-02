# 2v1 Football MDP – Half Field Offense

This project is an implementation of the **2v1 Half Field Offense football problem** using **Markov Decision Processes (MDPs)** and **Value Iteration**.  
It is based on an assignment problem from IIT Bombay’s CS747 course (Reinforcement Learning).

---

## Problem Description

You control two attackers **B1** and **B2** against one defender **R** on a **4x4 football half-pitch**.  
The objective is to maximize the probability of scoring a goal using **MDP planning**.

- The **state** is represented as:  
  `[B1_square, B2_square, R_square, ball_possession]`  
  where squares are numbered 1–16 row-wise and `ball_possession ∈ {1,2,0,3}` (B1, B2, lost, goal).

- **Action space (10 actions):**
  - B1 movement: Left, Right, Up, Down → `[0,1,2,3]`
  - B2 movement: Left, Right, Up, Down → `[4,5,6,7]`
  - Pass to teammate → `[8]`
  - Shoot on goal → `[9]`

- **Opponent policies:** (pre-computed probability distributions)
  - `greedy` → Defender moves towards the ball.  
  - `bus` → Defender shuffles near the goal (“park the bus”).  
  - `random` → Defender moves randomly.  

---

## Transition Function

The dynamics include **stochasticity** based on parameters `p` and `q`:

- **Movement:**  
  - With ball: success with probability `1-2p`, else lose possession.  
  - Without ball: success with probability `1-p`, else end of episode.  

- **Tackling:**  
  - If ball carrier and defender collide or swap squares → 50% chance of losing the ball.  

- **Passing:**  
  - Probability of success = `q - 0.1 * max(|x1-x2|, |y1-y2|)`  
  - Halved if defender lies between players.  

- **Shooting:**  
  - Probability of goal = `q - 0.2 * (3 - x)`  
  - Halved if defender is in front of the goal.  

Terminal states occur when:  
- Goal is scored (`ball_possession = 3`), or  
- Ball possession is lost (`ball_possession = 0`).  

---

## Requirements

```bash
python >= 3.8
```

Install dependencies:

```bash
pip install pickle argparse
```

---

## Usage

Run the script with command line arguments:  

```bash
python football_mdp.py -p <p_value> -q <q_value> -policy <opponent_policy>
```

### Example:
```bash
python football_mdp.py -p 0.3 -q 0.8 -policy greedy
```

Arguments:
- `-p` → movement failure parameter (default: 0.2, range: [0, 0.5])  
- `-q` → skill parameter for passing/shooting (default: 0.7, range: [0.6, 1])  
- `-policy` → opponent strategy (`random`, `greedy`, `bus`)  

---

## Outputs

- **Value Iteration** computes the optimal policy for attackers.  
- The script then **simulates matches** (default: 10,000 runs) from starting state `(5, 9, 8, 1)`.  
- Prints the number of wins (goals scored) vs losses (possession lost).  

Example output:
```
Value iteration complete
wins: 6785, loss: 3215
```

---

## Repository Structure

```
├── mdp.py               # Main solution file (your code)
├── all_states.pkl                 # Pickle file containing all possible states
├── pickle_policies/
│   ├── random_policy.pkl
│   ├── greedy_policy.pkl
│   └── bus_policy.pkl
└── README.md
```

---

## Extensions

- Generate graphs showing probability of winning vs parameters `p` and `q`.  
- Compare performance against different opponent strategies.  
- Explore TD-learning methods (TD(0), TD(λ)) as alternatives to value iteration.

---

## License

This project is released for **educational purposes** under the MIT License.

