import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    returns_sum = np.zeros(n_states, dtype=float)
    returns_count = np.zeros(n_states, dtype=int)

    for episode in episodes:
        states = [s for s, _ in episode]
        rewards = [r for _, r in episode]

        returns = np.zeros(len(episode), dtype=float)
        G = 0.0
        for t in reversed(range(len(episode))):
            G = rewards[t] + gamma * G
            returns[t] = G

        visited = set()
        for t, s in enumerate(states):
            if s not in visited:
                visited.add(s)
                returns_sum[s] += returns[t]
                returns_count[s] += 1

    V = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]

    return V