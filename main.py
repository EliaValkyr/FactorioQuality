import numpy as np
import matplotlib.pyplot as plt

recycling_chance = 0.25

NORMAL_INDEX = 0
UNCOMMON_INDEX = 1
RARE_INDEX = 2
EPIC_INDEX = 3
LEGENDARY_INDEX = 4

def computeQualityCosts(quality_chance: float):
    def createMatrix():
        normal_quality = np.array([1-quality_chance, quality_chance * 0.9, quality_chance * 0.09, quality_chance * 0.009, quality_chance * 0.001])
        def nextRow(arr):
            next_row = np.concatenate(([0], arr[:-1]))
            next_row[-1] += arr[-1]
            return next_row
        uncommon_quality = nextRow(normal_quality)
        rare_quality = nextRow(uncommon_quality)
        epic_quality = nextRow(rare_quality)
        legendary_quality = nextRow(epic_quality)
        quality_matrix = np.array([normal_quality, uncommon_quality, rare_quality, epic_quality, legendary_quality])
        return quality_matrix


    crafting_matrix = createMatrix()
    recycling_matrix = crafting_matrix / 4
    iteration_matrix = np.matmul(crafting_matrix, recycling_matrix)


    legendary_value = 1.0
    quality_costs = np.array([legendary_value]) # Start with the value of legendary, which is 1 by definition

    # Computes the cost of 1 legendary item in terms of the quality given by the index.
    def computeCost(quality_index: int):
        nonlocal quality_costs

        # spend 1 item of current quality, we get this amount back.
        returned_rate_of_same_quality = iteration_matrix[quality_index][quality_index]

        # spend 1 item of current quality, we get these amounts of higher qualities.
        rate_of_higher_qualities = iteration_matrix[quality_index][quality_index + 1:]

        # compute the value obtained from higher qualities, all translated to legendary.
        value_of_higher_qualities = np.dot(rate_of_higher_qualities, 1/quality_costs)

        # finally compute the cost of 1 legendary item in the current quality:
        # the effective spending divided by the amount of legendary items obtained.
        cost = (1-returned_rate_of_same_quality)/value_of_higher_qualities

        # add it to the start because we go from legendary to normal
        quality_costs = np.insert(quality_costs, [0], cost)

    computeCost(EPIC_INDEX)
    computeCost(RARE_INDEX)
    computeCost(UNCOMMON_INDEX)
    computeCost(NORMAL_INDEX)
    # print(quality_costs)
    return quality_costs

xs = np.linspace(0.01, 0.25, 100)
ys = list(map(computeQualityCosts, xs))

plt.plot(xs, ys)
plt.yscale('log')
xticks = np.linspace(0, 0.25, 11)
plt.xticks(xticks, np.round(100*xticks, 1))
plt.show()

# Quality values per module:
# T1: 1     1.3     1.6     1.9     2.5
# T2: 2     2.6     3.2     3.8     5
# T3: 2.5   3.2     4       4.7     6.2
