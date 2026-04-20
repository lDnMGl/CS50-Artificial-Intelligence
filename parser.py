import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S Conj VP
NP -> N | Det NP | Adj NP | NP PP
VP -> V | V NP | V PP | Adv VP | VP Adv
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Tokeniza, convierte a minúsculas y filtra palabras no alfabéticas.
    """
    tokens = nltk.word_tokenize(sentence.lower())
    # Filtrar: solo palabras que tengan al menos una letra
    return [word for word in tokens if any(char.isalpha() for char in word)]


def np_chunk(tree):
    """
    Retorna subárboles con etiqueta "NP" que no contengan otros "NP".
    """
    chunks = []

    # Buscamos todos los subárboles con etiqueta NP
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        # Verificamos si este NP tiene otros NP dentro (excluyendo a sí mismo)
        has_internal_np = False
        for internal_tree in subtree.subtrees():
            if internal_tree != subtree and internal_tree.label() == "NP":
                has_internal_np = True
                break

        if not has_internal_np:
            chunks.append(subtree)

    return chunks


if __name__ == "__main__":
    main()
