import numpy as np


FIRST_DATA_ROW = 10


def parse_row(row: str) -> tuple[int, int]:
    row = row.strip()
    x, y = row.split("            ")
    return int(x), int(y)


def load(filename: str) -> np.ndarray:
    with open(filename) as file:
        rows = file.readlines()
    rows = rows[FIRST_DATA_ROW:]
    rows = [parse_row(row)
            for row in rows
            if parse_row(row)[1]]
    x, y = zip(*rows)
    return np.asarray([x, y], dtype=np.int32)
