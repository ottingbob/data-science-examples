import random
from abc import ABC, abstractmethod
from typing import List

# Empty cell
BLANK = " "
AI_PLAYER = "X"
HUMAN_PLAYER = "O"

TRAINING_EPOCHS = 40_000

# Probability to try and (explore) new strategies
TRAINING_EPSILON = 0.4
REWARD_WIN = 10
REWARD_LOSE = -10
REWARD_TIE = -1


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
        default_q=1,
    ):
        # This is the epsilon parameter of the model which is the probability
        # of exploration
        self.epsilon = epsilon
        # This is the learning rate
        self.alpha = alpha
        # Gamma is the discount parameter for future reward. Rewards are better
        # NOW than rewards in the future
        self.gamma = gamma
        # If the given move at the given state is not defined yet, we use the
        # default Q value
        self.default_q = default_q
        # Q(s,a) function is a dict in this implementation. This is the Q function
        # Q: SxA -> R
        # Return a value for `s` state and `a` action (s,a) pair
        self.q = {}
        # Previous move during the game
        self.move = None
        # Board in the previous iteration
        self.board = (BLANK,) * 9

    # These are the available or empty cells on the grid / board
    def available_moves(self, board: List[str]):
        return [i for i in range(9) if board[i] == BLANK]

    # Q(s,a) -> Q value for (s,a) pair
    # If no Q value exists then create a new one with the default value = 1
    # and otherwise we return the q value present in the dict
    def get_q(self, state: List[str], action: int):
        sa_tuple = (state, action)
        if self.q.get(sa_tuple, None) is None:
            self.q[sa_tuple] = self.default_q
        return self.q[sa_tuple]

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
            # Now calculate the Q value
            self.q[(self.board, self.move)] = prev_q + self.alpha * (
                value + self.gamma * max_q_new - prev_q
            )


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
