import numpy as np
import math


class Neuron:
    def __init__(self, input_n: int, label: str):
        self.weights = np.random.random((input_n,))
        self.neighbors: list[Neuron] = []
        self.label = label

    def __str__(self):
        return (
            f"Neuron ({self.label}):"
            + "\n"
            + "Weights:"
            + "\n"
            + str(self.weights)
            + "\n"
            + f"Neighbors: ({', '.join([w.label for w in self.neighbors])})"
        )


def create_grid(n: int, raio: int, input_n: int) -> list[Neuron]:
    neurons = [[Neuron(input_n, str(j + (i * n))) for j in range(n)] for i in range(n)]

    for i in range(n):
        for x in range(-raio, raio + 1):
            if i + x < 0 or i + x >= n:
                continue
            for j in range(n):
                for y in range(-(raio - abs(x)), (raio - abs(x)) + 1):
                    if j + y < 0 or j + y >= n or (x == 0 and y == 0):
                        continue

                    neurons[i][j].neighbors.append(neurons[i + x][j + y])

    return [n for l in neurons for n in l]


def euclidian_dist(x, y) -> float:
    return math.sqrt(sum((x - y) ** 2))


def dist(x) -> float:
    return math.sqrt(sum(x**2))


def unit_normalization(neuron: Neuron):
    neuron.weights /= dist(neuron.weights)


def normalize_grid(grid: list[Neuron]):
    for n in grid:
        unit_normalization(n)


def print_grid(grid: list[Neuron]):
    for n in grid:
        print(f"{str(n)}")
        print("----------------")


def fit(grid: list[Neuron], X, epoch: int = 10, learning_rate: int = 0.1):
    normalize_grid(grid)
    for _ in range(epoch):
        for x_i in X:
            winner = grid[np.argmin([euclidian_dist(n.weights, x_i) for n in grid])]
            winner.weights += (x_i - winner.weights) * learning_rate
            unit_normalization(winner)

            for n in winner.neighbors:
                n.weights += (x_i - n.weights) * learning_rate * 1 / 2
                unit_normalization(n)


def predict_class(grid: list[Neuron], X, class_table: dict[str, str]):
    return class_table[
        grid[np.argmin([euclidian_dist(n.weights, X) for n in grid])].label
    ]


def main():
    class_table = {
        "0": "Classe A",
        "1": "Classe A",
        "2": "Classe B",
        "3": "Classe B",
        "4": "Classe A",
        "5": "Classe A",
        "6": "Classe B",
        "7": "Classe B",
        "8": "Classe A",
        "9": "Classe A",
        "10": "Classe C",
        "11": "Classe C",
        "12": "Classe C",
        "13": "Classe C",
        "14": "Classe C",
        "15": "Classe C",
    }
    grid = create_grid(4, 1, 3)
    example = np.array(
        [
            [0.2471, 0.1778, 0.2905],
            [0.8240, 0.2223, 0.7041],
            [0.4960, 0.7231, 0.5866],
            [0.2923, 0.2041, 0.2234],
            [0.8118, 0.2668, 0.7484],
            [0.4837, 0.8200, 0.4792],
            [0.3248, 0.2629, 0.2375],
            [0.7209, 0.2116, 0.7821],
            [0.5259, 0.6522, 0.5957],
            [0.2075, 0.1669, 0.1745],
            [0.7830, 0.3171, 0.7888],
            [0.5393, 0.7510, 0.5682],
        ]
    )
    fit(grid, example)
    predict = lambda e: predict_class(grid, e, class_table)
    for x in example:
        print(x, predict(x))


if __name__ == "__main__":
    main()
