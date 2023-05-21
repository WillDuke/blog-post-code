from collections import Counter

import pytest
from blog_post_code.wordle_cliques.cliques import (
    extract_archive_to_word_list,
    get_unique_set_words_of_length_n,
    find_all_size_n_cliques
)

@pytest.fixture
def words():
    return extract_archive_to_word_list()

@pytest.fixture
def subset(words):
    return words | get_unique_set_words_of_length_n(5)

def test_removal_of_anagrams(subset):
    """Test that at most one word with a given set of letters exists in
    the word list after anagrams are removed."""
    assert max(Counter(map(frozenset, subset)).values()) == 1

def test_all_five_cliques_have_no_duplicate_letters(subset):
    assert all(
        len(set(joined := ''.join(clique))) == len(joined)
        for dct in find_all_size_n_cliques(subset, 5)
        for _, clique in dct.items()
    ), "At least one five clique has a duplicate letter."
        

