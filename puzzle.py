from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Reglas generales: Cada personaje es Knight o Knave, pero no ambos.
# Definimos esto como una estructura base para todos los puzzles.
def get_game_rules():
    return And(
        Or(AKnight, AKnave), Not(And(AKnight, AKnave)),
        Or(BKnight, BKnave), Not(And(BKnight, BKnave)),
        Or(CKnight, CKnave), Not(And(CKnight, CKnave))
    )

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    get_game_rules(),
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    get_game_rules(),
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    get_game_rules(),
    # A dice que son iguales (Knight/Knight o Knave/Knave)
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    # B dice que son diferentes (Knight/Knave o Knave/Knight)
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    get_game_rules(),
    # Lo que sea que dijo A, se cumple la lógica de su tipo
    Or(Biconditional(AKnight, AKnight), Biconditional(AKnight, AKnave)),
    # B afirma que A dijo "Soy un villano"
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),
    # B afirma que C es villano
    Biconditional(BKnight, CKnave),
    # C afirma que A es caballero
    Biconditional(CKnight, AKnight)
)

def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
