from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.axes import Axes

RECYCLING_CHANCE = 0.25
RECYCLER_MODULE_SLOTS = 4
NUM_X_SAMPLES = 100

class QualityType:
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    EPIC = 3
    LEGENDARY = 4
    COUNT = 5

def formatQualityLineSubplot(ax: Axes):
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', axis='both')
    # xTicks = np.arange(MIN_X_RANGE, MAX_X_RANGE, 0.005)
    # ax.set_xticks(xTicks, 100 * xTicks)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xlabel("Quality per module")

def getQualityEffectIncrease(item_quality: int):
    match item_quality:
        case QualityType.COMMON: return 1.0
        case QualityType.UNCOMMON: return 1.3
        case QualityType.RARE: return 1.6
        case QualityType.EPIC: return 1.9
        case QualityType.LEGENDARY: return 2.5
        case _: raise Exception(f"unknown quality type {item_quality}")

def getQualityName(item_quality: int):
    match item_quality:
        case QualityType.COMMON: return "Common"
        case QualityType.UNCOMMON: return "Uncommon"
        case QualityType.RARE: return "Rare"
        case QualityType.EPIC: return "Epic"
        case QualityType.LEGENDARY: return "Legendary"
        case _: raise Exception(f"unknown quality type {item_quality}")

def getQualityModuleEffect(quality_module_tier: int):
    match quality_module_tier:
        case 1: return 0.01
        case 2: return 0.02
        case 3: return 0.025
        case _: raise Exception(f"unknown module tier {quality_module_tier}")

def getQualityAmount(quality_module_tier: int, quality_module_quality: int, n_quality_modules: int):
    quality_per_module = np.round(getQualityModuleEffect(quality_module_tier) * getQualityEffectIncrease(quality_module_quality) * n_quality_modules, 3)
    return quality_per_module * n_quality_modules

def getAllQualityModulesValues():
    result = np.array([])
    for tier in range(1, 4):
        for quality in range(QualityType.UNCOMMON, QualityType.COUNT):
            result = np.append(result, getQualityAmount(tier, quality, 1))
    result = np.unique(result)
    return np.sort(result)

def getProductivityModuleEffect(productivity_module_tier: int):
    match productivity_module_tier:
        case 0: return 0.
        case 1: return 0.04
        case 2: return 0.06
        case 3: return 0.1
        case _: raise Exception(f"unknown module tier {productivity_module_tier}")

def getProductivityAmount(productivity_module_tier: int, productivity_module_quality: int, n_productivity_modules: int, extra_productivity: float):
    return 1 + getProductivityModuleEffect(productivity_module_tier) * getQualityEffectIncrease(productivity_module_quality) * n_productivity_modules + extra_productivity

def computeQualityMatrix(quality_chance: float, productivity: float):
    normal_quality = np.array([1-quality_chance, quality_chance * 0.9, quality_chance * 0.09, quality_chance * 0.009, quality_chance * 0.001]) * productivity
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


def computeCraftingOutput(ingredient_amounts: np.ndarray, quality_chance: float, productivity: float, productivity_for_legendaries: float):
    # Legendary ingredients get crafted directly into products using full productivity and no quality modules
    legendary_products = ingredient_amounts[QualityType.LEGENDARY] * productivity_for_legendaries
    ingredient_amounts[QualityType.LEGENDARY] = 0

    quality_matrix = computeQualityMatrix(quality_chance, productivity)
    product_amounts = np.matmul(ingredient_amounts, quality_matrix)
    product_amounts[QualityType.LEGENDARY] += legendary_products
    return product_amounts

def computeRecyclingOutput(product_amounts: np.ndarray, quality_chance: float):
    assert product_amounts[QualityType.LEGENDARY] == 0, "we should not recycle legendary products!"
    quality_matrix = computeQualityMatrix(quality_chance, RECYCLING_CHANCE) #Recyclers have a "productivity" smaller than 1
    ingredient_amounts = np.matmul(product_amounts, quality_matrix)
    return ingredient_amounts

def computeCommonIngredientValue(crafting_quality_chance: float, recycling_quality_chance: float, productivity: float, productivity_for_legendary_crafts: float):
    # We define the value of a legendary product as 1, since that item is our goal, and we compute everything else with respect to this.
    # For simplicity, we assume we are working with a recipe where 1 ingredient makes 1 product (like barrels),
    # but this works for every recipe because it just adds a multiplication factor and everything here is linear.
    legendary_product_value = 1.0
    legendary_ingredient_value: float = productivity_for_legendary_crafts # An ingredient is actually more valuable than a product because of productivity.
    product_values = np.array([legendary_product_value])
    ingredient_values = np.array([legendary_ingredient_value])

    def computeValuesForQuality(quality_index: int):
        nonlocal product_values, ingredient_values
        crafting_matrix = computeQualityMatrix(crafting_quality_chance, productivity)
        recycling_matrix = computeQualityMatrix(recycling_quality_chance, RECYCLING_CHANCE) # Recyclers have a "productivity" smaller than 1

        # If we recycle 1 item of the given quality, we will get some amount of ingredients of the same quality back,
        # plus some more items of higher qualities.
        # We already know the values of higher quality items, so we compute the total value of higher quality stuff we get.
        same_qual_ingredient_amount = recycling_matrix[quality_index][quality_index]
        higher_qual_ingredient_amount = recycling_matrix[quality_index][quality_index + 1:]
        value_of_higher_qual_ingredients = np.dot(higher_qual_ingredient_amount, ingredient_values)

        # We do the same but for crafting 1 product.
        same_qual_product_amount = crafting_matrix[quality_index][quality_index]
        higher_qual_product_amount = crafting_matrix[quality_index][quality_index + 1:]
        value_of_higher_qual_products = np.dot(higher_qual_product_amount, product_values)

        # We now have 2 unknowns: the values of 1 ingredient and 1 product of the current quality (same_qual_ingredient_value and same_qual_product_value).
        # We have 2 linear equations:
        # 1 * !!same_qual_product_value!! = same_qual_ingredient_amount * !!same_qual_ingredient_value!! + value_of_higher_qual_ingredients
        # 1 * !!same_qual_ingredient_value!! = same_qual_product_amount * !!same_qual_product_value!! + value_of_higher_qual_products

        # We rearrange the equations and put them in matrix form: A*x = b
        # 1 * !!same_qual_product_value!! - same_qual_ingredient_amount * !!same_qual_ingredient_value!! = value_of_higher_qual_ingredients
        # - same_qual_product_amount * !!same_qual_product_value!! + 1 * !!same_qual_ingredient_value!! = value_of_higher_qual_products

        # We now solve the system, with:
        # A = [[1, -same_qual_ingredient_amount], [-same_qual_product_amount, 1]]
        # b = [value_of_higher_qual_ingredients], value_of_higher_qual_products]

        equation_matrix = [[1, -same_qual_ingredient_amount], [-same_qual_product_amount, 1]]
        equation_scalars = [value_of_higher_qual_ingredients, value_of_higher_qual_products]
        same_qual_product_value, same_qual_ingredient_value = np.linalg.solve(equation_matrix, equation_scalars)

        # After finding the values of the current quality, we add them to the array so they can be used in the next iteration.
        # Add them at the front, because we want the array to be sorted from lowest to highest quality.
        product_values = np.insert(product_values, [0], same_qual_product_value)
        ingredient_values = np.insert(ingredient_values, [0], same_qual_ingredient_value)

    computeValuesForQuality(QualityType.EPIC)
    computeValuesForQuality(QualityType.RARE)
    computeValuesForQuality(QualityType.UNCOMMON)
    computeValuesForQuality(QualityType.COMMON)

    # The value of 1 common ingredient needs to be divided by the maximum productivity we can get in the recipe,
    # because that's the amount of products we would obtain without grinding for quality.
    common_ingredient_value = ingredient_values[0]
    return common_ingredient_value / productivity_for_legendary_crafts

def computeValue(quality_module_tier: int, quality_module_quality: int, productivity_module_tier: int, productivity_module_quality: int,
                 n_quality_modules: int, module_slots: int, extra_productivity: float):
    # Recyclers can't have productivity, so we put the max quality modules in there.
    crafting_quality = getQualityAmount(quality_module_tier, quality_module_quality, n_quality_modules)
    recycling_quality = getQualityAmount(quality_module_tier, quality_module_quality, module_slots)

    # To craft items with legendary ingredients, we use full productivity, as we have no use for quality modules.
    # Extra productivity comes from cool machines and also repeatable techs for certain recipes.
    crafting_productivity = getProductivityAmount(productivity_module_tier, productivity_module_quality, module_slots - n_quality_modules, extra_productivity)
    legendary_crafting_productivity = getProductivityAmount(productivity_module_tier, productivity_module_quality, module_slots, extra_productivity)

    return computeCommonIngredientValue(crafting_quality, recycling_quality, crafting_productivity, legendary_crafting_productivity)

def computeQualityLineData(xs: np.ndarray, n_quality_modules: int, module_slots: int, productivity_module_tier: int, productivity_module_quality: int, extra_productivity: float):
    # To craft items with legendary ingredients, we use full productivity, as we have no use for quality modules.
    # Extra productivity comes from cool machines and also repeatable techs for certain recipes.
    crafting_productivity = getProductivityAmount(productivity_module_tier, productivity_module_quality, module_slots - n_quality_modules, extra_productivity)
    legendary_crafting_productivity = getProductivityAmount(productivity_module_tier, productivity_module_quality, module_slots, extra_productivity)

    ys = list(map(lambda quality_per_module: 1/computeCommonIngredientValue(
        quality_per_module * n_quality_modules, quality_per_module * RECYCLER_MODULE_SLOTS, crafting_productivity, legendary_crafting_productivity), xs))
    return ys

def createQualityLineSubplot(ax: Axes, module_slots: int, productivity_module_tier: int, productivity_module_quality: int, extra_productivity: float):
    min_quality = getQualityAmount(1, QualityType.COMMON, 1)
    max_quality = getQualityAmount(3, QualityType.LEGENDARY, 1)
    xs = np.linspace(min_quality, max_quality, NUM_X_SAMPLES)

    for n_productivity_modules in range(module_slots + 1):
        n_quality_modules = module_slots - n_productivity_modules
        ys = computeQualityLineData(xs, n_quality_modules, module_slots, productivity_module_tier, productivity_module_quality, extra_productivity)
        ax.plot(xs, ys, label=f"{n_productivity_modules} prod modules")
    
    productivity_per_module = getProductivityAmount(productivity_module_tier, productivity_module_quality, 1, 0)
    title = f"Using {getQualityName(productivity_module_quality)} T{productivity_module_tier} productivity modules (+{np.round((productivity_per_module - 1)*100)}%)"
    ax.set_title(title, y=1.0, pad=30)
    formatQualityLineSubplot(ax)

    def addVerticalLine(quality_module_tier: int, quality_module_quality: int):
        quality = getQualityAmount(quality_module_tier, quality_module_quality, 1)
        path = f"resources/quality-modules/t{quality_module_tier}-{getQualityName(quality_module_quality)}.png"
        img = plt.imread(path)
        im = OffsetImage(img, zoom=0.35)
        im.image.axes = ax

        ab = AnnotationBbox(im, (quality, ax.get_ylim()[1]),  xybox=(0., 12.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=0)
        ax.add_artist(ab)
        
        ax.vlines(quality, 0, 1, transform=ax.get_xaxis_transform(), colors='black', linewidths=0.5)

    addVerticalLine(1, QualityType.COMMON)
    addVerticalLine(2, QualityType.COMMON)
    addVerticalLine(3, QualityType.COMMON)
    addVerticalLine(3, QualityType.UNCOMMON)
    addVerticalLine(3, QualityType.RARE)
    addVerticalLine(3, QualityType.EPIC)
    addVerticalLine(3, QualityType.LEGENDARY)



def createQualityLinePlot(module_slots: int, extra_productivity: float, title: str):
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.set_size_inches(18, 12)
    plt.suptitle(title)

    createQualityLineSubplot(axs[0, 0], module_slots, 1, QualityType.COMMON, extra_productivity)
    createQualityLineSubplot(axs[0, 1], module_slots, 3, QualityType.COMMON, extra_productivity)
    createQualityLineSubplot(axs[1, 0], module_slots, 3, QualityType.RARE, extra_productivity)
    createQualityLineSubplot(axs[1, 1], module_slots, 3, QualityType.LEGENDARY, extra_productivity)

    for ax in axs.flat:
        ax.label_outer()
    plt.text(0.075, 0.5, "# of items to create 1 legendary product", fontsize=11, horizontalalignment='right',
             verticalalignment='center', transform=plt.gcf().transFigure, rotation='vertical')
    plt.text(0.51, 0.5, "# of items to create 1 legendary product", fontsize=11, horizontalalignment='right',
             verticalalignment='center', transform=plt.gcf().transFigure, rotation='vertical')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.25)
    filename = f"out/quality_line_plot_{module_slots}_modules_{np.round(extra_productivity*100, 0)}_prod.png"
    plt.savefig(filename)
    # plt.show()
    print(1)

def main2():
    np.set_printoptions(suppress=True, precision=3)

    createQualityLinePlot(4, 0, f"Quality costs in Assembler Machines")

    createQualityLinePlot(5, 0.5, f"Quality costs in EM Machines")

main2()

# def computeQualityCosts(quality_per_module: float, prod_per_module: float, n_prod_modules: int):
#     def createMatrix(l_n_quality_modules: int, l_n_prod_modules: int):
#         quality_chance = quality_per_module * l_n_quality_modules
#         normal_quality = np.array([1-quality_chance, quality_chance * 0.9, quality_chance * 0.09, quality_chance * 0.009, quality_chance * 0.001])
#         def nextRow(arr):
#             next_row = np.concatenate(([0], arr[:-1]))
#             next_row[-1] += arr[-1]
#             return next_row
#         uncommon_quality = nextRow(normal_quality)
#         rare_quality = nextRow(uncommon_quality)
#         epic_quality = nextRow(rare_quality)
#         legendary_quality = nextRow(epic_quality)
#         quality_matrix = np.array([normal_quality, uncommon_quality, rare_quality, epic_quality, legendary_quality])
#
#         productivity = 1 + prod_per_module * l_n_prod_modules
#         result = quality_matrix * productivity
#         print("====================")
#         print(f"{l_n_quality_modules} x {quality_per_module*100}% Quality | {l_n_prod_modules} x {prod_per_module*100}% Productivity:")
#         print(result)
#         print()
#         return result
#
#     crafting_matrix = createMatrix(MODULE_SLOTS - n_prod_modules, n_prod_modules)
#     recycling_matrix = createMatrix(MODULE_SLOTS, 0) * RECYCLING_CHANCE # recyclers can't have prod modules
#     iteration_matrix = np.matmul(crafting_matrix, recycling_matrix)
#
#     legendary_value = 1.0
#     quality_costs = np.array([legendary_value]) # Start with the value of legendary, which is 1 by definition
#
#     # Computes the cost of 1 legendary item in terms of the quality given by the index.
#     def computeCost(quality_index: int):
#         nonlocal quality_costs
#
#         # spend 1 item of current quality, we get this amount back.
#         returned_rate_of_same_quality = iteration_matrix[quality_index][quality_index]
#
#         # spend 1 item of current quality, we get these amounts of higher qualities.
#         rate_of_higher_qualities = iteration_matrix[quality_index][quality_index + 1:]
#
#         # compute the value obtained from higher qualities, all translated to legendary.
#         cost_of_higher_qualities = np.dot(rate_of_higher_qualities, 1/quality_costs)
#
#         # finally compute the cost of 1 legendary item in the current quality:
#         # the effective spending divided by the amount of legendary items obtained.
#         cost = (1-returned_rate_of_same_quality)/cost_of_higher_qualities
#
#         # add it to the start because we go from legendary to normal
#         quality_costs = np.insert(quality_costs, [0], cost)
#
#     computeCost(EPIC_INDEX)
#     computeCost(RARE_INDEX)
#     computeCost(UNCOMMON_INDEX)
#     computeCost(COMMON_INDEX)
#     return quality_costs
#
# def plotAllQualities(prod_per_module: float, n_prod_modules: int, last_plot_in_column: bool):
#     xs = np.linspace(MIN_X_RANGE, MAX_X_RANGE, NUM_X_SAMPLES)
#     ys = list(map(lambda x: computeQualityCosts(x, prod_per_module, n_prod_modules), xs))
#     # ax.plot(xs, [y[0] for y in ys], label= "normal", color='grey')
#     # ax.plot(xs, [y[1] for y in ys], label= "uncommon", color='green')
#     # ax.plot(xs, [y[2] for y in ys], label= "rare", color='blue')
#     # ax.plot(xs, [y[3] for y in ys], label= "epic", color='purple')
#     # # ax.plot(xs, [y[4] for y in ys], label= "legendary", color='orange')
#     #
#     # formatPlot(ax)
#     # if not last_plot_in_column:
#     #     ax.xaxis.set_ticks_position('none')
#     #     ax.xaxis.set_major_formatter(mtick.NullFormatter())
#     #
#     # ax.text(0.01,  0.02, f"{n_prod_modules} prod modules", horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
#     return [y[0] for y in ys]
#
#
# def createPlotForProductivity(ax: Axes, prod_per_module: float):
#     # fig = plt.figure()
#     # plt.suptitle(f"Quality Costs with prod modules of {prod_per_module * 100}% productivity")
#     # last_subplot = plt.subplot(5, 2, 9)
#     ys = np.array([
#         plotAllQualities(prod_per_module, 0, False),
#         plotAllQualities(prod_per_module, 1, False),
#         plotAllQualities(prod_per_module, 2, False),
#         plotAllQualities(prod_per_module, 3, False),
#         plotAllQualities(prod_per_module, 4, True)
#     ])
#
#     # ax = plt.subplot(1, 2, 2)
#     xs = np.linspace(MIN_X_RANGE, MAX_X_RANGE, NUM_X_SAMPLES)
#
#     for idy, y in enumerate(ys):
#         ax.plot(xs, y, label=f"{idy} prod modules")
#
#     ax.set_title(f"{prod_per_module*100}% per productivity module")
#     ax.vlines([0.01, 0.02, 0.025, 0.032, 0.04, 0.047, 0.062], 0, 1, transform=ax.get_xaxis_transform(), colors='black', linewidths=1)
#     formatPlot(ax)
#
# def main():
#     np.set_printoptions(suppress=True, precision=3)
#
#     fig, axs = plt.subplots(2, 2, sharey=True)
#     plt.suptitle(f"Quality Costs with prod modules of varying productivity")
#
#     createPlotForProductivity(axs[0, 0], 0.04)
#     createPlotForProductivity(axs[0, 1], 0.06)
#     createPlotForProductivity(axs[1, 0], 0.1)
#     createPlotForProductivity(axs[1, 1], 0.25)
#
#     for ax in axs.flat:
#         ax.label_outer()
#     plt.text(0.075, 0.5, "# of items to create 1 legendary item", fontsize=11, horizontalalignment='right',
#              verticalalignment='center', transform=plt.gcf().transFigure, rotation='vertical')
#     plt.text(0.51, 0.5, "# of items to create 1 legendary item", fontsize=11, horizontalalignment='right',
#              verticalalignment='center', transform=plt.gcf().transFigure, rotation='vertical')
#
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#     plt.show()

# main()
# Quality values per module:
# T1: 1     1.3     1.6     1.9     2.5
# T2: 2     2.6     3.2     3.8     5
# T3: 2.5   3.2     4       4.7     6.2
