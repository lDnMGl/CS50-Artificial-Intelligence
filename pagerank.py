import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    """
    pages = dict()

    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Devuelve la distribución de probabilidad de qué página visitar a continuación.
    """
    prob_dist = {}
    n_pages = len(corpus)
    links = corpus[page]

    # Si la página no tiene enlaces, probabilidad igual para todas
    if not links:
        for p in corpus:
            prob_dist[p] = 1 / n_pages
        return prob_dist

    # Probabilidad de elegir una página al azar (1 - d)
    random_prob = (1 - damping_factor) / n_pages

    # Probabilidad de seguir un enlace específico (d)
    link_prob = damping_factor / len(links)

    for p in corpus:
        prob_dist[p] = random_prob
        if p in links:
            prob_dist[p] += link_prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Estima el PageRank mediante el modelo del surfista aleatorio con n muestras.
    """
    page_counts = {page: 0 for page in corpus}

    # El primer sample es una página aleatoria del corpus
    current_page = random.choice(list(corpus.keys()))
    page_counts[current_page] += 1

    # Generar el resto de los n-1 samples
    for _ in range(n - 1):
        probs = transition_model(corpus, current_page, damping_factor)
        pages = list(probs.keys())
        weights = list(probs.values())

        # Elegir la siguiente página basada en los pesos (probabilidades)
        current_page = random.choices(pages, weights=weights, k=1)[0]
        page_counts[current_page] += 1

    # Normalizar los conteos para que sumen 1
    return {page: count / n for page, count in page_counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Calcula PageRank mediante la fórmula recursiva hasta la convergencia.
    """
    n = len(corpus)
    pagerank = {page: 1 / n for page in corpus}

    # Manejo de páginas sin enlaces: tratarlas como si enlazaran a todas
    # Esto es crucial para que el algoritmo no "pierda" probabilidad
    fixed_corpus = {page: (links if links else set(corpus.keys())) for page, links in corpus.items()}

    while True:
        new_ranks = {}
        for p in corpus:
            # Parte base de la fórmula
            rank = (1 - damping_factor) / n

            # Sumatoria de PageRanks de páginas i que enlazan a p
            sum_links = 0
            for i, links in fixed_corpus.items():
                if p in links:
                    sum_links += pagerank[i] / len(links)

            rank += damping_factor * sum_links
            new_ranks[p] = rank

        # Condición de salida: ningún valor cambia más de 0.001
        if all(abs(new_ranks[page] - pagerank[page]) < 0.001 for page in corpus):
            # Asegurar normalización final por errores de punto flotante
            total = sum(new_ranks.values())
            return {page: r / total for page, r in new_ranks.items()}

        pagerank = new_ranks.copy()


if __name__ == "__main__":
    main()
