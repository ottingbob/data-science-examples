import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Empty cell
BLANK = " "
AI_PLAYER = "X"
HUMAN_PLAYER = "O"

TRAINING_EPOCHS = 8_000

# Probability to try and (explore) new strategies
TRAINING_EPSILON = 0.4

# These rewards might bias the AI to NOT always make the `best` move
# Since it is incentivised to NOT lose it may try to win at all costs
# where it won't prevent the other player from winning, it only focuses
# on it's chances of winning...
# Therefore these might need to be re-adjusted to not have it penalized
# so much for a loss
#
REWARD_WIN = 10
# We heavily penalize losses and ties in this strategy
REWARD_LOSE = -100
REWARD_TIE = -10


# Human and AI players
class Player(ABC):
    def __init__(self):
        pass

    # Prints the board as a 3x3 grid
    @staticmethod
    def show_board(board: List[str]):
        print("|".join(board[0:3]))
        print("|".join(board[3:6]))
        print("|".join(board[6:9]))

    @abstractmethod
    def reward(self, value, board):
        pass


class HumanPlayer(Player):
    def __init__(self):
        pass

    def reward(self, value, board):
        # raise NotImplementedError("Human should not get rewarded!")
        if value != REWARD_LOSE:
            print("Congrats you didn't lose")

    def make_move(self, board: List[str]) -> int:
        while True:
            try:
                self.show_board(board)
                move = input("Your next move (cell index 1-9):")
                move = int(move)

                # Make sure the move is valid
                if not (move - 1 in range(9)):
                    raise ValueError
            except ValueError:
                print("Invalid move; try again...")
            else:
                # Based on index 0
                return move - 1


class AIPlayer(Player):
    # These are the params needed for the Time-Difference Learning concept
    # It is an online-algorithm: meaning we can update Q(s,a) as soon as we
    #   know s'
    # If alpha = 1 then we end up with the Bellman-Equation
    # The learning rate controls how much of a difference between previous
    #   Q(s,a) value and the newly proposed Q(s',a') value s taken into account
    def __init__(
        self,
        epsilon=TRAINING_EPSILON,
        alpha=0.3,
        gamma=0.9,
    ):
        # This is the epsilon parameter of the model which is the probability
        # of exploration
        self.epsilon = epsilon
        # This is the learning rate
        self.alpha = alpha
        # Gamma is the discount parameter for future reward. Rewards are better
        # NOW than rewards in the future
        self.gamma = gamma

        # Previous move during the game
        self.move = None
        # Board in the previous iteration
        self.board = (BLANK,) * 9

        # Q(s,a) function is a neural network in this implementation.
        # This is how we eliminate the huge data structure that we used
        #   previously which was a dict
        # It has a single hidden layer with 32 neurons - the output is 1 value
        #   which will be the index of the optimal move
        # The input is 36 neurons = 3 x 3 = 9 x 4 = 36
        #
        # Q: SxA -> R
        # Return a value for `s` state and `a` action (s,a) pair
        model = Sequential()
        model.add(Dense(32, input_dim=36, activation="relu"))
        model.add(Dense(1, activation="relu"))
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.q = model

    # These are the available or empty cells on the grid / board
    def available_moves(self, board: List[str]):
        return [i for i in range(9) if board[i] == BLANK]

    # Encode the (s,a) pair as a vector to be in the correct format to be
    # input into the neural network
    def encode_input(self, board: List[str], action: int):
        # (s,a) pair represented as a 1D array
        vector_representation = []

        # One-hot encoding for the 3 states
        # [1,0,0] -> means the given cell has X ticker
        # [0,1,0] -> means the given cell has 0 ticker
        # [0,0,1] -> means the given cell has no ticker / blank / empty
        #
        # Every single cell on the board (9 cells) has 3 values because of
        # this representation
        # Therefore there are 9 x 3 = 27 values
        for cell in board:
            for ticker in [AI_PLAYER, HUMAN_PLAYER, BLANK]:
                if cell == ticker:
                    vector_representation.append(1)
                else:
                    vector_representation.append(0)

        # One-hot encoding of the action - array with size 9
        # [1,0,0,0,0,0,0,0,0] -> means putting marker to the first cell
        # [0,1,0,0,0,0,0,0,0] -> means putting marker to the second cell
        # ...
        for move in range(9):
            if action == move:
                vector_representation.append(1)
            else:
                vector_representation.append(0)

        # All together there are 27 + 9 = 36 values
        # Where the input layer has 36 neurons
        return np.array([vector_representation])

    # Make a random move with epsilon probability (exploration) or pick the
    # action with the highest Q value (exploitation)
    def make_move(self, board: List[str]) -> int:
        self.board = tuple(board)
        # Get the available moves
        # TODO: Shouldn't we pass in `self.board` here ??
        actions = self.available_moves(board)
        # Action with epsilon probability
        if random.random() < self.epsilon:
            # This is in index (0-8 board cell related index)
            self.move = random.choice(actions)
            return self.move

        # Take the action with the highest Q value
        q_values = [self.get_q(self.board, a) for a in actions]
        max_q_value = max(q_values)
        # TODO: Couldn't I just use this:
        # max_q_value = max(actions, key=lambda a: self.get_q(self.board, a))

        # If multiple best actions, choose one at random
        # TODO: Again this looks super naive...
        if q_values.count(max_q_value) > 1:
            best_actions = [
                i for i in range(len(actions)) if q_values[i] == max_q_value
            ]
            best_move = actions[random.choice(best_actions)]
        else:
            best_move = actions[q_values.index(max_q_value)]

        # Store the best move
        self.move = best_move
        return self.move

    # Q(s,a) -> Q value for (s,a) pair
    # We have input (s,a) representation and teh neural network will make a
    # prediction and return the Q value.
    # The Q value will be learned during training
    def get_q(self, state: List[str], action: int):
        return self.q.predict([self.encode_input(state, action)], batch_size=1)

    # Evaluate a given state
    # Update the Q(s,a) table regarding `s` state and `a` action
    def reward(self, value, board: List[str]):
        # If we have a valid best move
        if self.move:
            # Use the Time-Difference Learning formula
            # The state is the board itself
            # The action is the index of the best move
            prev_q = self.get_q(self.board, self.move)
            # Calculate Q' value
            max_q_new = max(
                [self.get_q(tuple(board), a) for a in self.available_moves(self.board)]
            )
            # Now train the neural network with the new (s,a) and Q value
            # This allows the neural network to "remember" the given (s,a) pair
            q_value = prev_q + self.alpha * ((value + self.gamma * max_q_new) - prev_q)
            self.q.fit(
                self.encode_input(self.board, self.move),
                q_value,
                epochs=3,
                verbose=0,
            )

        # We reinitialize these since reward happens on a game over /
        # terminal state
        self.move = None
        self.board = None


class TicTacToe:
    def __init__(self, player_1: Player, player_2: Player):
        self.player_1 = player_1
        self.player_2 = player_2
        self.first_player_turn = random.choice([True, False])
        self.board = [BLANK] * 9

    def play(self):
        # This is the game loop
        # TODO: This conditional logic is CRINGE
        while True:
            # First player make a move
            if self.first_player_turn:
                player = self.player_1
                other_player = self.player_2
                player_tickers = (AI_PLAYER, HUMAN_PLAYER)
            else:
                player = self.player_2
                other_player = self.player_1
                player_tickers = (HUMAN_PLAYER, AI_PLAYER)

            # Check the state of the game - win / lose / draw
            game_over, winner = self.is_game_over(player_tickers)

            # Game is over then handle rewards
            if game_over:
                self.handle_game_over(
                    winner=winner,
                    player=player,
                    other_player=other_player,
                    player_tickers=player_tickers,
                )
                break

            # Flip to the next players turn
            self.first_player_turn = not self.first_player_turn

            # Handle the current players best move
            # Based on Q(s,a) for the AI player
            move = player.make_move(self.board)
            self.board[move] = player_tickers[0]

    def handle_game_over(
        self,
        winner,
        player,
        other_player,
        player_tickers,
    ):
        player.show_board(self.board)
        if winner == player_tickers[0]:
            print(f"\n{player.__class__.__name__} won!")
            player.reward(REWARD_WIN, self.board)
            other_player.reward(REWARD_LOSE, self.board)
        elif winner == player_tickers[1]:
            print(f"\n{other_player.__class__.__name__} won!")
            player.reward(REWARD_LOSE, self.board)
            other_player.reward(REWARD_WIN, self.board)
        # Here is the tie which is what we should actually check
        # for initially
        else:
            print("TIE!")
            player.reward(REWARD_TIE, self.board)
            other_player.reward(REWARD_TIE, self.board)

    def is_game_over(self, player_tickers):
        # Consider both players
        # X and O players are the tickers
        for player_ticker in player_tickers:
            # Check the horizontal dimensions (rows)
            for i in range(3):
                if (
                    self.board[3 * i + 0] == player_ticker
                    and self.board[3 * i + 1] == player_ticker
                    and self.board[3 * i + 2] == player_ticker
                ):
                    return True, player_ticker
            # Check the vertical dimensions (columns)
            for i in range(3):
                if (
                    self.board[i + 0] == player_ticker
                    and self.board[i + 3] == player_ticker
                    and self.board[i + 6] == player_ticker
                ):
                    return True, player_ticker
            # Check diagonal 1
            if (
                self.board[0] == player_ticker
                and self.board[4] == player_ticker
                and self.board[8] == player_ticker
            ):
                return True, player_ticker
            # Check diagonal 2
            if (
                self.board[2] == player_ticker
                and self.board[4] == player_ticker
                and self.board[6] == player_ticker
            ):
                return True, player_ticker

        # Check for a draw
        if self.board.count(BLANK) == 0:
            return True, None
        return False, None


if __name__ == "__main__":
    ai_player_1 = AIPlayer(epsilon=TRAINING_EPSILON)
    ai_player_2 = AIPlayer(epsilon=TRAINING_EPSILON)
    print("Training the AI player(s)...")
    for i in range(TRAINING_EPOCHS):
        game = TicTacToe(
            player_1=ai_player_1,
            player_2=ai_player_2,
        )
        game.play()

    print("\nTraining is DONE!")
    # Epsilon=0 means no exploration - it will only use the Q(s,a) function
    # to make the moves
    ai_player_1.epsilon = 0
    human_player = HumanPlayer()
    game = TicTacToe(ai_player_1, human_player)
    game.play()
