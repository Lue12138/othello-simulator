"""
An AI player for Othello.
"""

import random
import sys
import time

# You can use the functions in utilities to write your AI
# from utilities import find_lines, get_possible_moves, get_score, play_move
from utilities import get_possible_moves, get_score, play_move

cached_states = {}


# you can use this for debugging, as it will print to sterr and not stdout
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Method to compute utility value of terminal state
def compute_utility(board, color):
    # IMPLEMENT
    score = get_score(board)
    if color == 1:
        return score[0] - score[1]
    else:
        return score[1] - score[0]


# Better heuristic value of board
def compute_heuristic(board, color):  # not implemented, optional
    # IMPLEMENT
    util = compute_utility(board, color)
    n = len(board)
    if n <= 4:
        return util

    variety = 0
    dark = 0
    light = 0
    # the more corner the better, on the other hand the more stars the worse
    corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
    for (i, j) in corners:
        if board[i][j] == 1:
            dark += 500
        elif board[i][j] == 2:
            light += 500
    stars = [(1, 0), (0, 1), (0, -2), (1, -1),
             (-2, 0), (-1, 1), (-1, -2), (-2, -1)]
    for (i, j) in stars:
        if board[i][j] == 1:
            dark -= 250
        elif board[i][j] == 2:
            light -= 250
    # the more edges the better
    for k in range(2, n - 2):
        edge = [(0, k), (-1, k), (k, 0), (k, -1)]
        for (i, j) in edge:
            if board[i][j] == 1:
                dark += 200
            elif board[i][j] == 2:
                light += 200
    if color == 1:
        # the more variety the better
        variety = 100 * (len(get_possible_moves(board, 1)) -
                         len(get_possible_moves(board, 2)))
        return util + variety + (dark - light)
    else:
        # the more variety the better
        variety = 100 * (len(get_possible_moves(board, 2)) -
                         len(get_possible_moves(board, 1)))
        return util + variety + (light - dark)


# Method to reorder possible moves by their utility
def reorder(board, color, possible_moves):
    util_to_move = {}
    ordered_moves = []
    for move in possible_moves:
        new_board = play_move(board, color, move[0], move[1])
        util = compute_utility(new_board, color)
        if util in util_to_move and util_to_move[util] != [move]:
            util_to_move[util].append(move)
        else:
            util_to_move[util] = [move]
    for utils in sorted(util_to_move.keys(), reverse=True):
        ordered_moves += util_to_move[utils]
    return ordered_moves

# --------------- MINIMAX -----------------


def mm_min_node(board, color, limit, caching=0):
    # IMPLEMENT (and replace the line below)
    # first check if we have seen it
    if caching == 1 and (board, color) in cached_states:
        return cached_states[(board, color)]

    possible_moves = get_possible_moves(board, 3 - color)
    # base case
    if len(possible_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    best_move = None
    min_utility = float('inf')
    for move in possible_moves:
        # get the new board after move
        new_board = play_move(board, 3 - color, move[0], move[1])
        # compute utility
        utility = mm_max_node(new_board, color, limit - 1, caching)[1]
        if utility < min_utility:
            best_move = move
            min_utility = utility
    return best_move, min_utility


def mm_max_node(board, color, limit, caching=0):
    # IMPLEMENT (and replace the line below)
    # first check if we have seen it
    if caching == 1 and (board, color) in cached_states:
        return cached_states[(board, color)]

    possible_moves = get_possible_moves(board, color)
    # base case
    if len(possible_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    best_move = None
    max_utility = -float('inf')
    for move in possible_moves:
        # get the new board after move
        new_board = play_move(board, color, move[0], move[1])
        # compute utility
        utility = mm_min_node(new_board, color, limit - 1, caching)[1]
        if utility > max_utility:
            best_move = move
            max_utility = utility
    return best_move, max_utility


def claim_mm(board, color, limit, caching=0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit
    that is equal to the value of the parameter. Search only to nodes at
    a depth-limit equal to the limit.  If nodes at this level are non-terminal
    return a heuristic value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of
    state evaluations. If caching is OFF (i.e. 0), do NOT use state caching to
    reduce the number of state evaluations.
    """
    # IMPLEMENT (and replace the line below)
    # if isinstance(board, list):
    #     board = tuple(tuple(row) for row in board)
    if caching == 1:
        cached_states[(board, color)] = mm_max_node(
            board, color, limit, caching)
    return mm_max_node(board, color, limit, caching)[0]

# ---------------------- ALPHA-BETA PRUNING ----------------------


def ab_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    # IMPLEMENT (and replace the line below)
    # check cached_states
    if caching == 1 and (board, color) in cached_states:
        return cached_states[(board, color)]

    possible_moves = get_possible_moves(board, 3 - color)
    # base case
    if len(possible_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    best_move = None
    min_utility = float('inf')

    # order possible_moves
    if ordering == 1:
        possible_moves = reorder(board, color, possible_moves)
    for move in possible_moves:
        # get the new board after move
        new_board = play_move(board, 3 - color, move[0], move[1])
        # compute utility
        utility = ab_max_node(new_board, color, alpha,
                              beta, limit - 1, caching, ordering)[1]
        if utility < min_utility:
            best_move = move
            min_utility = utility
        # alpha-beta cut
        beta = min(beta, utility)
        if beta <= alpha:
            break
    return best_move, min_utility


def ab_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    # IMPLEMENT (and replace the line below)
    # check cached_states
    if caching == 1 and (board, color) in cached_states:
        return cached_states[(board, color)]

    possible_moves = get_possible_moves(board, color)
    # base case
    if len(possible_moves) == 0 or limit == 0:
        return None, compute_utility(board, color)

    best_move = None
    max_utility = -float('inf')

    if ordering == 1:
        possible_moves = reorder(board, color, possible_moves)
    for move in possible_moves:
        # get the new board after move
        new_board = play_move(board, color, move[0], move[1])
        # compute utility
        utility = ab_min_node(new_board, color, alpha,
                              beta, limit - 1, caching, ordering)[1]
        if utility > max_utility:
            best_move = move
            max_utility = utility
        alpha = max(alpha, max_utility)
        if alpha >= beta:
            break
    return best_move, max_utility


def claim_ab(board, color, limit, caching=0, ordering=0):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit
    that is equal to the value of the parameter. Search only to nodes at a
    depth-limit equal to the limit.  If nodes at this level are non-terminal
    return a heuristic value (see compute_utility) If caching is ON (i.e. 1),
    use state caching to reduce the number of state evaluations. If caching is
    OFF (i.e. 0), do NOT use state caching to reduce the number of state
    evaluations. If ordering is ON (i.e. 1), use node ordering to expedite
    pruning and reduce the number of state evaluations. If ordering is OFF
    (i.e. 0), do NOT use node ordering to expedite pruning and reduce the
    number of state evaluations.
    """
    # IMPLEMENT (and replace the line below)
    if isinstance(board, list):
        board = tuple(tuple(row) for row in board)
    if caching == 1:
        cached_states[(board, color)] = ab_max_node(board, color, -float('inf'), float('inf'), limit, caching, ordering)
    return ab_max_node(board, color, -float('inf'), float('inf'), limit, caching, ordering)[0]

####################################################


def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI")  # First line is the name of this AI
    arguments = input().split(",")

    # Player color: 1 for dark (goes first), 2 for light.
    color = int(arguments[0])
    limit = int(arguments[1])  # Depth limit
    minimax = int(arguments[2])  # Minimax or alpha beta
    caching = int(arguments[3])  # Caching
    ordering = int(arguments[4])  # Node-ordering (for alpha-beta only)

    if (minimax == 1):
        eprint("Running MINIMAX")
    else:
        eprint("Running ALPHA-BETA")

    if (caching == 1):
        eprint("State Caching is ON")
    else:
        eprint("State Caching is OFF")

    if (ordering == 1):
        eprint("Node Ordering is ON")
    else:
        eprint("Node Ordering is OFF")

    if (limit == -1):
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1):
        eprint("Node Ordering should have no impact on Minimax")

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)
        if status == "FINAL":  # Game is over.
            print
        else:
            # Read in the input and turn it into a Python
            board = eval(input())
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)
            # Select the move and send it to the manager
            if (minimax == 1):  # run this if the minimax flag is given
                movei, movej = claim_mm(board, color, limit, caching)
            else:  # else run alphabeta
                movei, movej = claim_ab(board, color, limit, caching, ordering)

            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
