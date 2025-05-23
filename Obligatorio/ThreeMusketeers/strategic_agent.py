from agent import Agent
from board import Board


class StrategicAgent(Agent):
    def __init__(self, player, strategy="expectimax", heuristic="default"):
        super().__init__(player)
        self.strategy = strategy
        self.heuristic = heuristic

    def next_action(self, obs):
        if self.strategy == "expectimax":
            _, best_move = self.expectimax(obs, depth=3, maximizing=True)
        elif self.strategy == "minimax":
            _, best_move = self.minimax(
                obs, depth=3, alpha=float("-inf"), beta=float("inf"), maximizing=True
            )
        else:
            raise ValueError("Unknown strategy: " + self.strategy)
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_end(self.player)[0]:
            return self.heuristic_utility(board), None

        best_move = None
        if maximizing:
            max_eval = float("-inf")
            for move in board.get_possible_actions(self.player):
                next_board = board.clone()
                next_board.play(self.player, move)
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in board.get_possible_actions(3 - self.player):
                next_board = board.clone()
                next_board.play(3 - self.player, move)
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def expectimax(self, board, depth, maximizing):
        if depth == 0 or board.is_end(self.player)[0]:
            return self.heuristic_utility(board), None

        if maximizing:
            max_eval = float("-inf")
            best_move = None
            for move in board.get_possible_actions(self.player):
                next_board = board.clone()
                next_board.play(self.player, move)
                eval_score, _ = self.expectimax(next_board, depth - 1, False)
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
            return max_eval, best_move
        else:
            total_eval = 0
            moves = board.get_possible_actions(3 - self.player)
            for move in moves:
                next_board = board.clone()
                next_board.play(3 - self.player, move)
                eval_score, _ = self.expectimax(next_board, depth - 1, True)
                total_eval += eval_score
            avg_eval = total_eval / len(moves) if moves else 0
            return avg_eval, None

    def heuristic_utility(self, board: Board):
        if self.heuristic == "default":
            musketeer_positions = board.find_musketeer_positions()
            num_guards = len(board.find_enemy_positions())
            alignment = self.evaluate_alignment(musketeer_positions)
            return -10000 if alignment else -num_guards
        elif self.heuristic == "surrounding_guards":
            return -self.eval_guards_surrounding_musketeers(board)
        elif self.heuristic == "mobility":
            return self.eval_player_mobility(board)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")

    def eval_player_mobility(self, board: Board):
        player_moves = len(board.get_possible_actions(self.player))
        opponent_moves = len(board.get_possible_actions(3 - self.player))
        return player_moves - opponent_moves

    def evaluate_alignment(self, musketeer_positions):
        rows = [pos[0] for pos in musketeer_positions]
        cols = [pos[1] for pos in musketeer_positions]
        return len(set(rows)) == 1 or len(set(cols)) == 1
