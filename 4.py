import numpy as np

maze = [
    "S.#.....",
    ".##.###.",
    "....#...",
    ".####.#.",
    "......#.",
    ".####.#G",
    "....#...",
    "###...#.",
]
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
start, goal = (0, 0), (5, 7)

alpha, gamma = 0.1, 0.95
epsilon, eps_min, eps_decay = 1.0, 0.01, 0.995
episodes, max_steps = 2000, 200

Q = np.zeros((8, 8, 4))


def step(state, action):
    r = state[0] + moves[action][0]
    c = state[1] + moves[action][1]
    if r < 0 or r > 7 or c < 0 or c > 7 or maze[r][c] == "#":
        return state, -5, False
    if (r, c) == goal:
        return (r, c), 100, True
    return (r, c), -1, False


for ep in range(episodes):
    state = start
    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])
        next_state, reward, done = step(state, action)
        future = 0 if done else np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * future - Q[state][action])
        state = next_state
        if done:
            break
    epsilon = max(eps_min, epsilon * eps_decay)

state, path = start, [start]
while state != goal and len(path) < 64:
    state, _, _ = step(state, np.argmax(Q[state]))
    path.append(state)

print("reached goal:", state == goal)
print("steps:", len(path) - 1)
print(path)
