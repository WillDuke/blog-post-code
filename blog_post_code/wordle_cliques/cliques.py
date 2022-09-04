from __future__ import annotations

import itertools as it
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jsonlines
from igraph import Graph, plot
from pipe import Pipe

PARENT_DIR = Path(__file__).parent
WORDS_ARCHIVE_PATH = PARENT_DIR / "words_alpha.zip"
WORDS_FILENAME = "words_alpha.txt"
CLIQUES_PATH = PARENT_DIR / "cliques.jsonl"

EXAMPLE_WORDS = ["burps", "fldxt", "mckay", "vejoz", "whing", "track", "barge", "stink", "wreck"]

def extract_archive_to_word_list(archive_path: Path, filename: str) -> Iterable[str]:
    """
    Provided a path to a zip archive and the name of the file within the archive
    containing an english word list, extract and read the file and return an iterable
    of the words.
    """
    return (
        zipfile.ZipFile(archive_path, 'r')
        .read(filename)
        .decode('utf-8')
        .split("\r\n")
    )

@Pipe
def filter_words_of_length_n(words: List[str], n: int) -> Iterable[str]:
    """Filter out words not of length n."""
    return filter(lambda word: len(word) == n, words)

@Pipe
def filter_words_with_duplicate_letters(words) -> Iterable[str]:
    "Filter out words with more than one of any letter."
    return filter(lambda word: len(word) == len(set(word)), words)

@Pipe
def filter_duplicate_word_sets(words: Iterable[str]) -> Iterable[str]:
    """Filter out words with the same set of letters as an already seen word."""

    seen = set()
    seen_add = seen.add

    for word in words:
        wordset = frozenset(word)
        if wordset not in seen:
            seen_add(wordset)
            yield word

@Pipe
def get_unique_set_words_of_length_n(words: Iterable[str], n: int) -> Iterable[str]:
    """Get the filtered list of words of length n with no repeating digits,
    omitting any words with duplicate letter sets."""
    return (
        words
        | filter_words_of_length_n(n)
        | filter_words_with_duplicate_letters
        | filter_duplicate_word_sets
    )

def create_graph_of_disjoint_words(words: Iterable[str]) -> Graph:
    """
    Create a igraph.Graph where each vertex is a word and two words
    share an edge if they have no letters in common.
    
    igraph.Graph takes a list of edges as tuples of vertex indices.
    The edges are created by iterating through each word and adding
    edges for any remaining words that have disjoint letter sets.
    """
    wordsets = list(map(set, words))
    edges = [
        (i, j)
        for i, left in enumerate(wordsets)
        for j, right in it.islice(enumerate(wordsets), i+1, None)
        if left.isdisjoint(right)
    ]

    return Graph(edges = edges)

def find_all_size_n_cliques(
    words: Iterable[str], size: int
) -> Iterable[Dict[int, Tuple[str, ...]]]:
    """Provided an iterable of strings, return all of the sets of words
    of a given size with no overlapping letters between any pair in the set."""
    _words = list(words)

    graph = create_graph_of_disjoint_words(_words)

    for pos, clique in enumerate(graph.cliques(size, size)):
        yield {pos: tuple(_words[idx] for idx in clique)}

def plot_cliques(words: Iterable[str], filename: str):

    _words = list(words)
    graph = create_graph_of_disjoint_words(_words)

    graph.vs["label"] = words
    graph.vs["color"] = (["red"] * 5) + (["black"] * 4)

    plot(
        graph, 
        filename,
        layout=graph.layout("kk"), 
        bbox=(700, 300), 
        margin=40, 
        vertex_label_dist = 2, 
        vertex_size = 10
    )

def main(
    words_archive_path: Path = WORDS_ARCHIVE_PATH,
    words_filename: str = WORDS_FILENAME,
    cliques_path: Path = CLIQUES_PATH
):
    """
    Load a list of words from a file (filename: word_filename) 
    within a zip archive (words_archive_path). Save a file with
    all of the cliques (of mutually exlusive letters) to cliques_path.
    """

    words = (
        extract_archive_to_word_list(words_archive_path, words_filename)
        | get_unique_set_words_of_length_n(5)
    )

    five_cliques = find_all_size_n_cliques(words, 5)

    with jsonlines.open(cliques_path, 'w') as writer:
        writer.write_all(five_cliques)

if __name__ == '__main__':
    import time
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(end - start)
