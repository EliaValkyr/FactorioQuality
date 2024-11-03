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

####################### Utility Functions

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
    quality_per_module = np.round(getQualityModuleEffect(quality_module_tier) * getQualityEffectIncrease(quality_module_quality), 3)
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

def getModuleProductivity(productivity_module_tier: int, productivity_module_quality: int):
    return getProductivityModuleEffect(productivity_module_tier) * getQualityEffectIncrease(productivity_module_quality)

def getProductivityAmount(productivity_module_tier: int, productivity_module_quality: int, n_productivity_modules: int, extra_productivity: float):
    return 1 + getProductivityModuleEffect(productivity_module_tier) * getQualityEffectIncrease(productivity_module_quality) * n_productivity_modules + extra_productivity

####################### Math Functions

# Computes the quality matrix: a matrix that multiplied by some vector of input quality values,
# gives the amount of items of each quality of the output.
def computeQualityMatrix(quality_chance: float, productivity: float):
    normal_quality = np.array([1-quality_chance, quality_chance * 0.9, quality_chance * 0.09, quality_chance * 0.009, quality_chance * 0.001]) * productivity
    def nextRow(arr):
        # Add a zero at the start, and add the last two values
        next_row = np.concatenate(([0], arr[:-1]))
        next_row[-1] += arr[-1]
        return next_row
    uncommon_quality = nextRow(normal_quality)
    rare_quality = nextRow(uncommon_quality)
    epic_quality = nextRow(rare_quality)
    legendary_quality = nextRow(epic_quality)
    quality_matrix = np.array([normal_quality, uncommon_quality, rare_quality, epic_quality, legendary_quality])
    return quality_matrix

# Returns the value of 1 common ingredient, converted to legendary products.
# So a legendary product has a value of 1, and if for example a common ingredient has a value of 0.01 it means that
# it takes 100 times more ingredients to craft a legendary product than to craft a common product.
def computeCommonIngredientValue(crafting_quality_chance: float, recycling_quality_chance: float, productivity: float, productivity_for_legendary_crafts: float):
    # We define the value of a legendary product as 1, since that item is our goal, and we compute everything else with respect to this.
    # For simplicity, we assume we are working with a recipe where 1 ingredient makes 1 product (like barrels),
    # but this works for every recipe because it just adds a multiplication factor and everything here is linear.
    legendary_product_value = 1.0
    legendary_ingredient_value: float = productivity_for_legendary_crafts # An ingredient is actually more valuable than a product because of productivity.
    product_values = np.array([legendary_product_value])
    ingredient_values = np.array([legendary_ingredient_value])

    # Computes the values of an ingredient and a product of the given quality.
    def computeValuesForQuality(quality_index: int):
        nonlocal product_values, ingredient_values
        crafting_matrix = computeQualityMatrix(crafting_quality_chance, productivity)
        recycling_matrix = computeQualityMatrix(recycling_quality_chance, RECYCLING_CHANCE) # Recyclers have a "productivity" smaller than 1

        # If we recycle 1 product of the given quality, we will get some amount of ingredients of the same quality back,
        # plus some more ingredients of higher qualities.
        # We already know the values of higher quality items, so we compute the total value of higher quality ingredients we get.
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
        # x = [same_qual_product_value, same_qual_ingredient_value]
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
    # because that's the amount of common products we would obtain without grinding for quality.
    common_ingredient_value = ingredient_values[0]
    result = common_ingredient_value / productivity_for_legendary_crafts
    
    # print(f"Common ingredient value {result:.4} | Quality = {crafting_quality_chance:.4}|{recycling_quality_chance:.4}, Productivity = {productivity:.4}|{productivity_for_legendary_crafts:.4}")
    return result

####################### Plotting Functions

# Creates a new plot with 2x2 subplots
def createPlot(title: str):
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.set_size_inches(18, 12)
    plt.suptitle(title)

    for ax in axs.flat:
        ax.label_outer()

    plt.text(0.075, 0.5, "# of items to create 1 legendary product", fontsize=11, horizontalalignment='right',
             verticalalignment='center', transform=plt.gcf().transFigure, rotation='vertical')
    plt.text(0.51, 0.5, "# of items to create 1 legendary product", fontsize=11, horizontalalignment='right',
             verticalalignment='center', transform=plt.gcf().transFigure, rotation='vertical')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.25)
    return axs

def formatSubplot(ax: Axes):
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', axis='both')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))

def formatQualityLineSubplot(ax: Axes):
    formatSubplot(ax)
    ax.set_xlabel("Quality per module")

def formatProductivityLineSubplot(ax: Axes):
    formatSubplot(ax)
    ax.set_xlabel("Productivity per module")

# Adds a vertical line at the given x value, with the given icon on top.
def addVerticalLine(ax: Axes, x_value: float, path: str):
    img = plt.imread(path)
    im = OffsetImage(img, zoom=0.35)
    im.image.axes = ax

    ab = AnnotationBbox(im, (x_value, ax.get_ylim()[1]),  xybox=(0., 12.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)
    ax.add_artist(ab)
    
    ax.vlines(x_value, 0, 1, transform=ax.get_xaxis_transform(), colors='black', linewidths=0.5)

# Computes one line of data, where the x axis is the quality of 1 module.
def computeQualityLineData(xs: np.ndarray, n_quality_modules: int, module_slots: int, productivity_module_tier: int, productivity_module_quality: int, extra_productivity: float):
    # To craft items with legendary ingredients, we use full productivity, as we have no use for quality modules.
    # Extra productivity comes from cool machines and also repeatable techs for certain recipes.
    crafting_productivity = getProductivityAmount(productivity_module_tier, productivity_module_quality, module_slots - n_quality_modules, extra_productivity)
    legendary_crafting_productivity = getProductivityAmount(productivity_module_tier, productivity_module_quality, module_slots, extra_productivity)

    # print(f"========= Quality Line: {n_quality_modules}/{module_slots} q. modules, Productivity = {crafting_productivity:.4}|{legendary_crafting_productivity:.4}")
    ys = list(map(lambda quality_per_module: 1/computeCommonIngredientValue(
        quality_per_module * n_quality_modules, quality_per_module * RECYCLER_MODULE_SLOTS, crafting_productivity, legendary_crafting_productivity), xs))
    return ys

# Computes one line of data, where the x axis is the productivity of 1 module.
def computeProductivityLineData(xs: np.ndarray, n_quality_modules: int, module_slots: int, quality_module_tier: int, quality_module_quality: int, extra_productivity: float):
    # Recyclers can't have productivity, so we put the max quality modules in there.
    crafting_quality = getQualityAmount(quality_module_tier, quality_module_quality, n_quality_modules)
    recycling_quality = getQualityAmount(quality_module_tier, quality_module_quality, module_slots)
    n_productivity_modules = module_slots - n_quality_modules
    # print(f"========= Productivity Line: {n_quality_modules}/{module_slots} q. modules, Quality = {crafting_quality:.4}|{recycling_quality:.4}")

    ys = list(map(lambda prod_per_module: 1/computeCommonIngredientValue(crafting_quality, recycling_quality, 
        1 + prod_per_module * n_productivity_modules + extra_productivity, 1 + prod_per_module * module_slots + extra_productivity), xs))
    return ys

# Creates one subplot where the x axis is the quality. It has one line for each amount of quality vs productivity modules.
def createQualityLineSubplot(ax: Axes, module_slots: int, productivity_module_tier: int, productivity_module_quality: int, extra_productivity: float):
    min_quality = getQualityAmount(1, QualityType.COMMON, 1)
    max_quality = getQualityAmount(3, QualityType.LEGENDARY, 1)
    xs = np.linspace(min_quality, max_quality, NUM_X_SAMPLES)

    for n_productivity_modules in range(module_slots + 1):
        n_quality_modules = module_slots - n_productivity_modules
        ys = computeQualityLineData(xs, n_quality_modules, module_slots, productivity_module_tier, productivity_module_quality, extra_productivity)
        ax.plot(xs, ys, label=f"{n_productivity_modules} prod modules")
    
    productivity_per_module = getModuleProductivity(productivity_module_tier, productivity_module_quality)
    title = f"Using {getQualityName(productivity_module_quality)} T{productivity_module_tier} productivity modules (+{np.round(productivity_per_module*100)}%)"
    ax.set_title(title, y=1.0, pad=30)
    formatQualityLineSubplot(ax)

# Creates one subplot where the x axis is the productivity. It has one line for each amount of quality vs productivity modules.
def createProductivityLineSubplot(ax: Axes, module_slots: int, quality_module_tier: int, quality_module_quality: int, extra_productivity: float):
    min_productivity = getModuleProductivity(1, QualityType.COMMON)
    max_productivity = getModuleProductivity(3, QualityType.LEGENDARY)
    xs = np.linspace(min_productivity, max_productivity, NUM_X_SAMPLES)

    for n_quality_modules in range(module_slots + 1):
        ys = computeProductivityLineData(xs, n_quality_modules, module_slots, quality_module_tier, quality_module_quality, extra_productivity)
        ax.plot(xs, ys, label=f"{n_quality_modules} quality modules")
    
    quality_per_module = getQualityAmount(quality_module_tier, quality_module_quality, 1)
    title = f"Using {getQualityName(quality_module_quality)} T{quality_module_tier} quality modules (+{np.round((quality_per_module)*100, 2)}%)"
    ax.set_title(title, y=1.0, pad=30)
    formatProductivityLineSubplot(ax)

# Creates the plot where the x axis is the quality.
def createQualityLinePlot(module_slots: int, extra_productivity: float, title: str):
    axs = createPlot(title)

    createQualityLineSubplot(axs[0, 0], module_slots, 1, QualityType.COMMON, extra_productivity)
    createQualityLineSubplot(axs[0, 1], module_slots, 3, QualityType.COMMON, extra_productivity)
    createQualityLineSubplot(axs[1, 0], module_slots, 3, QualityType.RARE, extra_productivity)
    createQualityLineSubplot(axs[1, 1], module_slots, 3, QualityType.LEGENDARY, extra_productivity)

    for ax in axs.flat:
        def addQualityVerticalLine(module_tier: int, module_quality: int):
            quality = getQualityAmount(module_tier, module_quality, 1)
            path = f"resources/quality-modules/t{module_tier}-{getQualityName(module_quality)}.png"
            addVerticalLine(ax, quality, path)

        addQualityVerticalLine(1, QualityType.COMMON)
        addQualityVerticalLine(2, QualityType.COMMON)
        addQualityVerticalLine(3, QualityType.COMMON)
        addQualityVerticalLine(3, QualityType.UNCOMMON)
        addQualityVerticalLine(3, QualityType.RARE)
        addQualityVerticalLine(3, QualityType.EPIC)
        addQualityVerticalLine(3, QualityType.LEGENDARY)

    filename = f"out/quality_line_plot_{module_slots}_modules_{np.round(extra_productivity*100, 0)}_prod.png"
    plt.savefig(filename)
    # plt.show()

# Creates the plot where the x axis is the productivity.
def createProductivityLinePlot(module_slots: int, extra_productivity: float, title: str):
    axs = createPlot(title)

    createProductivityLineSubplot(axs[0, 0], module_slots, 1, QualityType.COMMON, extra_productivity)
    createProductivityLineSubplot(axs[0, 1], module_slots, 3, QualityType.COMMON, extra_productivity)
    createProductivityLineSubplot(axs[1, 0], module_slots, 3, QualityType.RARE, extra_productivity)
    createProductivityLineSubplot(axs[1, 1], module_slots, 3, QualityType.LEGENDARY, extra_productivity)

    for ax in axs.flat:
        def addProductivityVerticalLine(module_tier: int, module_quality: int):
            productivity = getProductivityAmount(module_tier, module_quality, 1, 0) - 1
            path = f"resources/productivity-modules/t{module_tier}-{getQualityName(module_quality)}.png"
            addVerticalLine(ax, productivity, path)

        addProductivityVerticalLine(1, QualityType.COMMON)
        addProductivityVerticalLine(2, QualityType.COMMON)
        addProductivityVerticalLine(3, QualityType.COMMON)
        addProductivityVerticalLine(3, QualityType.UNCOMMON)
        addProductivityVerticalLine(3, QualityType.RARE)
        addProductivityVerticalLine(3, QualityType.EPIC)
        addProductivityVerticalLine(3, QualityType.LEGENDARY)

    filename = f"out/productivity_line_plot_{module_slots}_modules_{np.round(extra_productivity*100, 0)}_prod.png"
    plt.savefig(filename)

def createPlots(module_slots: int, extra_productivity: float, title: str):
    createQualityLinePlot(module_slots, extra_productivity, title)
    createProductivityLinePlot(module_slots, extra_productivity, title)

def main():
    np.set_printoptions(suppress=True, precision=3)

    createPlots(4, 0, f"Quality costs in Assembler Machines")
    createPlots(5, 0.5, f"Quality costs in Electromagnetic Plants")

main()