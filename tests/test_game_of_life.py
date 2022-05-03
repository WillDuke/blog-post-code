import numpy as np
import pytest

from blog_post_code.game_of_life.conway import (
    create_random_grid,
    count_neighbors,
    count_neighbors_np,
    simulate
)

@pytest.fixture(scope='module')
def starting_grid():
    return create_random_grid(seed = 123)

@pytest.fixture(scope='module')
def neighbors(starting_grid):
    return count_neighbors(starting_grid)

@pytest.fixture(scope='module')
def second_grid(starting_grid):
    generations = simulate(starting_grid)
    return next(generations)

def test_np_and_scipy_count_methods(starting_grid):
    """Confirm that the two methods return the same result."""
    assert np.all(count_neighbors(starting_grid) == count_neighbors_np(starting_grid))

def test_array_size(starting_grid, neighbors, second_grid):
    """Confirm that all of the arrays have the expected size."""
    assert all([arr.shape == (10, 10) for arr in (starting_grid, neighbors, second_grid)])

def test_first_rule(starting_grid, neighbors, second_grid):
    """rule 1: any live cell with fewer than two neighbors dies"""
    assert not np.any(second_grid[np.bool_(starting_grid & (neighbors < 2))]), (second_grid, neighbors)

def test_second_rule(starting_grid, neighbors, second_grid):
    """rule 2: any live cell with 2 or 3 neighbors lives"""
    assert np.all(second_grid[np.bool_(starting_grid & ((neighbors == 2) | (neighbors == 3)))])

def test_third_rule(starting_grid, neighbors, second_grid):
    """rule 3: any live cell with more than 3 neighbors dies"""
    assert not np.any(second_grid[np.bool_(starting_grid & (neighbors > 3))])

def test_fourth_rule(starting_grid, neighbors, second_grid):
    """rule 4: any dead cell with 3 neighbors lives"""
    assert np.all(second_grid[np.bool_(~starting_grid & (neighbors == 3))])

def test_all_3_neighbor_cells_alive(second_grid, neighbors):
    """Confirm that there are no cases where a cell has 3 neighbors and the cell is dead in the next cell"""
    assert not np.any((neighbors == 3) & ~second_grid) 

def test_all_non_2_or_3_neighbor_cells_dead(second_grid, neighbors):
    """Confirm that there are no cases where a cell without 2 or 3 neighbors is alive"""
    assert not np.any(second_grid[np.bool_(((neighbors != 3) & (neighbors != 2)))])