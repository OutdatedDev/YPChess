import chess as ch
import concurrent.futures
from collections import defaultdict
import chess.polyglot

class Engine:
    def __init__(self, board, maxDepth, color):
        self.board = board
        self.color = color
        self.maxDepth = maxDepth
        self.transposition_table = {}
        self.node_count = 0

    def getBestMove(self):
        best_move = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        move_list = list(self.board.legal_moves)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_move = {executor.submit(self.evaluateMove, self.board.copy(stack=False), move, alpha, beta): move for move in move_list}
            for future in concurrent.futures.as_completed(future_to_move):
                move = future_to_move[future]
                try:
                    move_value = future.result()
                    if move_value > best_value:
                        best_value = move_value
                        best_move = move
                except Exception as e:
                    print(f"Error: {e}")
                
        return best_move

    def evaluateMove(self, board, move, alpha, beta):
        board.push(move)
        move_value = self.engine(board, alpha, beta, self.maxDepth - 1)
        board.pop()
        return move_value

    def evalFunct(self, board):
        return self.materialEvaluation(board) + self.mateOpportunity(board) + self.opening_book(board) + self.center_control(board)
    
    def materialEvaluation(self, board):
        score = 0
        for piece_type in ch.PIECE_TYPES:
            score += len(board.pieces(piece_type, self.color)) * self.pieceValue(piece_type)
            score -= len(board.pieces(piece_type, not self.color)) * self.pieceValue(piece_type)
        return score

    def pieceValue(self, piece_type):
        values = {
            ch.PAWN: 1,
            ch.ROOK: 5.25,
            ch.BISHOP: 3.5,
            ch.KNIGHT: 3.5,
            ch.QUEEN: 9,
            ch.KING: 0
        }
        return values.get(piece_type, 0)

    def mateOpportunity(self, board):
        if not list(board.legal_moves):
            if board.turn == self.color:
                return -1e9
            else:
                return 1e9
        return 0
    
    def center_control(self, board):
        center_squares = [ch.D4, ch.D5, ch.E4, ch.E5]
        score = 0
        piece_weights = {
            ch.PAWN: 0.2,
            ch.KNIGHT: 0.25,
            ch.BISHOP: 0.25,
            ch.ROOK: 0.15,
            ch.QUEEN: 0.3,
            ch.KING: 0.025
        }
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                weight = piece_weights.get(piece.piece_type, 0)
                score += weight if piece.color == self.color else -weight
        return score

    def opening_book(self, board) -> float:
        if board.fullmove_number < 10:
            multiplier = 1 / 30 if board.turn == self.color else -1 / 30
            return len(list(board.legal_moves)) * multiplier
        return 0.0

    def engine(self, board, alpha, beta, depth):
        if depth == 0 or board.is_game_over():
            return self.evalFunct(board)

        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]

        if board.turn == self.color:
            value = float("-99999")
            for move in self.orderMoves(board):
                board.push(move)
                value = max(value, self.engine(board, alpha, beta, depth - 1))
                board.pop()
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = float("9999999999")
            for move in self.orderMoves(board):
                board.push(move)
                value = min(value, self.engine(board, alpha, beta, depth - 1))
                board.pop()
                beta = min(beta, value)
                if beta <= alpha:
                    break

        self.transposition_table[board_hash] = value
        return value

    def orderMoves(self, board):
        """Order moves to improve alpha-beta pruning efficiency."""
        move_scores = defaultdict(int)
        for move in board.legal_moves:
            if board.is_capture(move):
                move_scores[move] += 100
            board.push(move)
            if board.is_check():
                move_scores[move] += 50
            board.pop()
        return sorted(board.legal_moves, key=lambda move: move_scores[move], reverse=True)