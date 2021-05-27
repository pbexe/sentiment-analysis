from random import shuffle
from typing import List, Tuple


def get_data(type_:str = "train", equalise:bool = True, randomise:bool = True) -> Tuple[List[str], List[int]]:
    """Load data from the provided filesystem

    Args:
        type_ (str, optional): Type of data to loads. Defaults to "train".
        equalise (bool, optional): Whether to equalise input classes.
        randomise (bool, optional): Whether to randomise the order of training data.

    Returns:
        Tuple[List[str], List[int]]: Tuple of x and y data.
    """

    # Load the data from the filesystem.
    x = []
    y = []
    with open(f"data/{type_}_text.txt", "r") as fp:
        x = fp.readlines()
    with open(f"data/{type_}_labels.txt", "r") as fp:
        y = [int(i) for i in fp.readlines()]
    
    # If required, shuffle the data.
    if randomise:
        zipped = list(zip(x,y))
        shuffle(zipped)
        x, y = zip(*zipped)
    
    # This is to overcome the unequal amounts of each class in the dataset.
    if equalise:
        freqencies = {item:y.count(item) for item in set(y)}
        min_ = min(freqencies.values())
        zipped = list(zip(x,y))
        total = dict()
        output = []
        for pair in zipped:
            if pair[1] in total:
                total[pair[1]] += 1
                if total[pair[1]] <= min_:
                    output.append(pair)
            else:
                total[pair[1]] = 1
        x, y = zip(*output)
    return x, y
