import random 

### Crossover with two different methods

def slice_parents(parent1, parent2):
    """
    Crossover method 1:
    ====================
    Slices the seeds of both parents in the middle and combines them. 
    Retruns the combines new seeds.
    """
    length = len(parent1) // 2
    child_seeds = parent1[:length] + parent2[length:]
    return child_seeds


def pick_random(parent1, parent2):
    """
    Crossover method 2:
    ===================
    For each seed in seed_index of both parents. Pick one for the child with a 50/50 prob.
    Returns the new child seeds
    """
    child_seeds = []
    parents = {1:parent1, 2:parent2}
    for idx in range(len(parent1)):
        choice = random.choice([1,2])
        parent = parents[choice]
        if idx < len(parent):
            seed = parent[idx]
            child_seeds.append(seed)
        else:
            pass
    return child_seeds
