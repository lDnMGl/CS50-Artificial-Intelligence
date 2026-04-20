import itertools
import random

class Minesweeper():
    """
    Representación del juego Minesweeper
    """
    def __init__(self, height=8, width=8, mines=8):
        self.height = height
        self.width = width
        self.mines = set()
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        self.mines_found = set()

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        count = 0
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) == cell: continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1
        return count

class Sentence():
    """
    Sentencia lógica: un conjunto de celdas y cuántas de ellas son minas.
    """
    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        if len(self.cells) == self.count and self.count != 0:
            return self.cells
        return set()

    def known_safes(self):
        if self.count == 0:
            return self.cells
        return set()

    def mark_mine(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)

class MinesweeperAI():
    """
    Jugador de Minesweeper basado en IA
    """
    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.moves_made = set()
        self.mines = set()
        self.safes = set()
        self.knowledge = []

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        # 1. Registrar movimiento y marcar como seguro
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # 2. Identificar vecinos cuya situación sea desconocida
        neighbors = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) in self.safes: continue
                if (i, j) in self.mines:
                    count -= 1
                    continue
                if 0 <= i < self.height and 0 <= j < self.width:
                    neighbors.add((i, j))

        # 3. Añadir nueva sentencia al conocimiento
        new_sentence = Sentence(neighbors, count)
        self.knowledge.append(new_sentence)

        # 4. Loop de inferencia
        changed = True
        while changed:
            changed = False

            # Buscar minas y seguros garantizados
            safes = set()
            mines = set()
            for sentence in self.knowledge:
                safes.update(sentence.known_safes())
                mines.update(sentence.known_mines())

            if safes:
                changed = True
                for s in safes:
                    self.mark_safe(s)
            if mines:
                changed = True
                for m in mines:
                    self.mark_mine(m)

            # Limpiar sentencias vacías
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

            # Inferencia por subconjuntos
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1.cells.issubset(s2.cells) and s1 != s2:
                        new_cells = s2.cells - s1.cells
                        new_count = s2.count - s1.count
                        new_s = Sentence(new_cells, new_count)
                        if new_s not in self.knowledge:
                            self.knowledge.append(new_s)
                            changed = True

    def make_safe_move(self):
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell
        return None

    def make_random_move(self):
        possible = []
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.moves_made and (i, j) not in self.mines:
                    possible.append((i, j))
        if not possible:
            return None
        return random.choice(possible)
