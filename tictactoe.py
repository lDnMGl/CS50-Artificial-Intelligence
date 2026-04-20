import math
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    # En el estado inicial, X siempre empieza.
    # Contamos cuántas jugadas hay de cada uno para saber el turno.
    count_x = sum(row.count(X) for row in board)
    count_o = sum(row.count(O) for row in board)
    return O if count_x > count_o else X

def actions(board):
    # Retorna un conjunto de tuplas (fila, columna) vacías.
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}

def result(board, action):
    # Validamos que la acción sea posible
    if action not in actions(board):
        raise Exception("Movimiento no válido")

    # Creamos una copia profunda (deep copy) para no alterar el tablero original.
    # Esto es requisito indispensable de la especificación.
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    # Revisamos combinaciones ganadoras (filas, columnas y diagonales)
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY: return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY: return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != EMPTY: return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY: return board[0][2]
    return None

def terminal(board):
    # El juego termina si hay un ganador o si ya no hay espacios vacíos.
    return winner(board) is not None or all(EMPTY not in row for row in board)

def utility(board):
    # Asignamos valores numéricos al resultado (X: 1, O: -1, Empate: 0)
    res = winner(board)
    if res == X: return 1
    elif res == O: return -1
    return 0

def minimax(board):
    # Si el juego ya terminó, no hay movimiento que hacer.
    if terminal(board): return None

    current_player = player(board)

    # Si es el turno de X (Maximizar), busca el valor más alto.
    # Si es el turno de O (Minimizar), busca el valor más bajo.
    if current_player == X:
        _, move = max_value(board)
    else:
        _, move = min_value(board)
    return move

# Funciones auxiliares para el algoritmo Minimax
def max_value(board):
    if terminal(board): return utility(board), None
    v = -math.inf
    best_move = None
    for action in actions(board):
        val, _ = min_value(result(board, action))
        if val > v:
            v = val
            best_move = action
    return v, best_move

def min_value(board):
    if terminal(board): return utility(board), None
    v = math.inf
    best_move = None
    for action in actions(board):
        val, _ = max_value(result(board, action))
        if val < v:
            v = val
            best_move = action
    return v, best_move
