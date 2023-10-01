import random

import numpy as np

# Epsilon greedy strategy
# The agent will visit new states with epsilon probability
EPSILON = 0.1

# Number of bandits
BANDITS = 3

# Number of iterations
EPISODES = 10_000


class Bandit:
    def __init__(self, probability):
        # Qk(a) stores the mean of the rewards
        # Long term reward of taking action `a` when agent is in state `s`
        self.q = 0
        # k means how many times action `a` (the bandit) was chosen in the past
        self.k = 0
        # The probability distribution
        self.probability = probability

    def get_reward(self) -> int:
        # Rewards can be +1 (win) or 0 (lose)
        if np.random.random() < self.probability:
            return 1
        return 0


class NArmedBandit:
    def __init__(self):
        self.bandits = [
            Bandit(probability=0.4),
            Bandit(probability=0.5),
            Bandit(probability=0.6),
        ]

    def run(self):
        for i in range(EPISODES):
            bandit = self.bandits[self.select_bandit()]
            reward = bandit.get_reward()
            self.update(bandit, reward)
            print(f"Iteration {i}: bandit {bandit.probability} with Q value {bandit.q}")

    def select_bandit(self) -> int:
        # This is the epsilon greedy strategy
        # With epsilon probability the agent will explore - otherwise it exploits
        if np.random.random() < EPSILON:
            # explore
            return random.randint(0, len(self.bandits) - 1)
        # exploit
        return self.get_bandit_max_q()

    def get_bandit_max_q(self) -> int:
        # Find the bandit with max Q(a) value for the greedy exploitation
        # We find and return the index of the bandit with max Q(a)
        return self.bandits.index(
            max(
                self.bandits,
                key=lambda b: b.q,
            )
        )

    def update(self, bandit, reward):
        bandit.k += 1
        bandit.q = bandit.q + (1 / (1 + bandit.k)) * (reward - bandit.q)

    # How many times was a given bandit chosen
    def show_statistics(self):
        for i in range(len(self.bandits)):
            print(f"Bandit {i} with k: {self.bandits[i].k}")


if __name__ == "__main__":
    bandit_problem = NArmedBandit()
    bandit_problem.run()
    bandit_problem.show_statistics()
