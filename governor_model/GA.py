import typing
from PerformancePredictor import predict_performance, build_perf_predictors
from PowerPredictor import predict_power
from time import time
from dataclasses import dataclass
from enum import Enum
from random import randint
from numpy import inf
from copy import deepcopy

# HYPERPARAMETERS WE SHOULD TEST
LATENCY_PENALTY = 2000
FPS_PENALTY = 50000
POPULATION_SIZE = 100


NETWORK_SIZE = 8
SGN = lambda x: (0 < x) - (x < 0)
ABS = lambda x: x if (x >= 0) else -x

class ComponentType(Enum):
    """The type of component."""
    BIG = 1
    GPU = 2
    LITTLE = 3


LittleFrequency = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000]

BigFrequency = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000,
                2100000, 2208000]


@dataclass
class Gene:
    """A gene is a single item in a chromosome. It corresponds to a single layer in the network."""
    componentType: ComponentType
    layers: int
    frequency_level: int | None  # in case of GPU, frequency is None, index in respective frequency list otherwise


@dataclass
class Chromosome:
    """A chromosome is a list of 11 genes."""
    genes: typing.List[Gene]
    fitness: float = 0.0

    def __getitem__(self, item):
        return self.genes[item]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __str__(self):
        # order = [self.genes[i].componentType for i in range(0, 3)]
        l1 = self.genes[0].layers
        l2 = self.genes[1].layers
        l3 = self.genes[2].layers
        bigFreqlv = self.genes[0].frequency_level    #note: order hardcoded
        littleFreqLv = self.genes[2].frequency_level

        return f"L1:{l1}, L2:{l2}, L3:{l3}, bigFreqLv:{bigFreqlv}, littleFreqLv:{littleFreqLv}"


def create_random_chromosome() -> Chromosome:
    """Create a random chromosome, make sure that the order of the components is consistent"""

    # random partition points in C
    ran1 = randint(1, NETWORK_SIZE)
    ran2 = randint(1, NETWORK_SIZE)
    rans = [ran1, ran2]
    order = int(ran1 > ran2)
    pp1 = rans[order]
    pp2 = rans[not order]

    # random frequencies
    little_frequency = randint(0, len(LittleFrequency) - 1)
    big_frequency = randint(0, len(BigFrequency) - 1)

    big_gene = Gene(ComponentType.BIG, pp1, big_frequency)
    gpu_gene = Gene(ComponentType.GPU, pp2 - pp1, None)
    little_gene = Gene(ComponentType.LITTLE, NETWORK_SIZE - pp2, little_frequency)

    return Chromosome(
        [big_gene, gpu_gene, little_gene]
    )

def parse_chromosome(string_rep: str):
    dict_ = dict([tuple(gene.split(":")) for gene in string_rep.split(", ")])
    for k,v in dict_.items():
        dict_[k] = int(v)

    # component_names = dict_["Order"]
    # components = [ComponentType[component] for component in component_names]
    l1 = dict_["L1"]
    l2 = dict_["L2"]
    l3 = dict_["L3"]
    b_freq = dict_["bigFreqLv"]
    l_freq = dict_["littleFreqLv"]
    big_gene = Gene(ComponentType.BIG, l1, b_freq)
    gpu_gene = Gene(ComponentType.GPU, l2, None)
    little_gene = Gene(ComponentType.LITTLE, l3, l_freq)

    return Chromosome(
        [big_gene, gpu_gene, little_gene]
    )


def load_population(filepath: str) -> typing.List[Chromosome]:
    population = []
    with open(filepath, "r") as f:
        f.readline()
        for line in f:
            population.append(parse_chromosome(line.strip()))
    return population


def initialize_population(population_size: int, assessor = None, import_path: str = None) -> typing.List[Chromosome]:
    """Initialize a population of chromosomes."""

    if import_path:
        return load_population(import_path)

    population = []

    for _ in range(population_size):
        individual = create_random_chromosome()
        if assessor:
            individual.fitness = fitness(assessor, individual)
        population.append(individual)

    return population


def layer_err(a: Chromosome) -> int:
    return a.genes[0].layers + a.genes[1].layers + a.genes[2].layers - NETWORK_SIZE


def cure_child_cancer(child: Chromosome, dbgchild=None):
    while layer_err(child):    # in C: while (int err = sgn(layer_err(child1)))
        # print("child during treatment:", child)
        err = SGN(layer_err(child))
        # print("error during treatment:", err)
        correction_layer = randint(0,2)
        layers = child.genes[correction_layer].layers
        if err < 0 and (correction_layer > 0 and layers >= 7 or layers >= 8):
            continue
        if err > 0 and (correction_layer == 0 and layers <= 1 or layers <= 0):
            continue
        # if dbgchild:
        #     print("child 1 before child 2 op:", dbgchild)
        child.genes[correction_layer].layers -= err
        # if dbgchild:
    #         print("child 1  after child 2 op:", dbgchild)
    # print("child  inside hospital:", child)

        # if child.genes[correction_layer].frequency_level != None:
        #     child.genes[correction_layer].frequency_level -= err
        # else: # this freq level code is not ready yet btw, it's not clamped
        #     child.genes[0].frequency_level -= err


def crossover(a: Chromosome, b: Chromosome, random_individual) -> Chromosome:
    """Performs one-point crossover between two chromosomes."""


    cut_point = randint(0,1)+1
    genes1 = a.genes[:cut_point] + b.genes[cut_point:]
    genes2 = b.genes[:cut_point] + a.genes[cut_point:]
    child1 = deepcopy(Chromosome(genes1))
    child2 = deepcopy(Chromosome(genes2))

    # print(str(a))
    # print(str(b))
    # print("ri1:", str(random_individual))

    # precure_child1 = deepcopy(child1)
    # precure_child2 = deepcopy(child2)
    # print("child1 before hospital:", child1)
    cure_child_cancer(child1)
    # print("child1 out of hospital:", child1)
    # postcure_child1 = deepcopy(child1)
    # print()
    # print("child2 before hospital:", child2)
    cure_child_cancer(child2, child1)
    # print("child2 out of hospital:", child2)
    # print()
    # print("ri2:", str(random_individual))

    try:
        assert sum(layer.layers for layer in child1.genes) == 8, "Child1 does not have exactly 8 layers"
        assert (1 <= child1.genes[0].layers <= 8), "Child1 layer 1 is not between size 1 and 8 "
        assert all(0 <= layer.layers <= 7 for layer in child1.genes[1:]), "Child1 layers 2 and 3 are nor of size 0-8"
    except AssertionError:
        # print("precure child 1:", str(precure_child1))
        # print("postcure child 1:", str(postcure_child1))
        print("assertion child 1:", str(child1))
        print(cut_point)
        raise AssertionError
    try:
        assert sum(layer.layers for layer in child2.genes) == 8, "Child2 does not have exactly 8 layers"
        assert (1 <= child2.genes[0].layers <= 8), "Child2 layer 1 is not between size 1 and 8 "
        assert all(0 <= layer.layers <= 7 for layer in child2.genes[1:]), "Child2 layers 2 and 3 are nor of size 0-8"
    except AssertionError:
        print("assertion child 2:", str(child2))
        # print("precure child 2:", str(precure_child2))
        print(cut_point)
        raise AssertionError


    return [child1, child2]


def mutate(individual: Chromosome) -> Chromosome:
    """Performs mutation on a chromosome. Takes a mutation rate 0-100"""

    # partition point mutation
    # if randint(0, 100) < mutation_rate:
    individual = mutate_layer_size(individual)

    # frequency mutation
    # if randint(0, 100) < mutation_rate:
    individual = mutate_frequency(individual)

    return individual


def mutate_layer_size(individual: Chromosome) -> Chromosome:
    """Mutates the partition point of a chromosome."""

    # get first or second partition point
    idx = randint(0, 1)

    # amount of change
    change = randint(-1, 1)
    while randint(0,99) < 20: # !(rand() % 5) (20% chance to go further)
            change += SGN(change)
            if ABS(change) > 2: # limit to ±3
                break

    # clamp
    # beware of boundaries
    if idx:
        change -= min(0, individual[1].layers + change) #L2 min size 0
        change -= max(0, individual[0].layers + individual[1].layers + change - NETWORK_SIZE) # L3 min size 0
    else:
        change -= min(0, individual[0].layers + change - 1) #L1 min size 1
        change += min(0, individual[1].layers - change) #L2 min size 0

    # add change to one
    individual[idx].layers += change

    # subtract from the other, this will always work since the sum is always NETWORK_SIZE
    individual[idx + 1].layers -= change

    cure_child_cancer(individual)    # pretty sure all's good but just some chemo couldn't hurt right?

    return individual


def mutate_frequency(individual: Chromosome) -> Chromosome:
    """Mutates the frequency of a gene in a chromosome."""

    # random gene
    rand_idx = randint(0, 1)*2     # don't select gpu
    gene = individual[rand_idx]

    # # GPU has no frequency to mutate
    # if gene.componentType == ComponentType.GPU:
    #     return individual

    # amount of change
    change = randint(-1, 1)

    # limit is the length of the frequency list
    limit = gene.componentType == ComponentType.BIG and len(BigFrequency)-1 or len(LittleFrequency)-1

    # beware of boundaries
    if gene.frequency_level + change > limit or gene.frequency_level + change < 0:
        change *= -1

    # apply change
    gene.frequency_level += change
    individual[rand_idx] = gene

    assert gene.componentType != ComponentType.GPU
    assert gene.componentType != ComponentType.BIG or gene.frequency_level < 13
    assert gene.componentType != ComponentType.LITTLE or gene.frequency_level < 9

    return individual


def fitness(assessor, chromosome: Chromosome) -> float:
    """Computes the fitness of a chromosome. assessor will be a function pointer"""
    return assessor.assess(chromosome)


def assess_population(population: typing.List[Chromosome], size: int, assessor) -> None:
    """Computes the fitness of an entire population."""
    for i in range(size):
        population[i].fitness = fitness(assessor, population[i])


def partition(arr: typing.List[Chromosome], low: int, high: int) -> int:
    """This function takes last element as pivot, places the pivot element at its correct position in sorted array,
    and places all smaller (smaller than pivot) to left of pivot and all greater elements to right of pivot"""

    i = (low - 1)  # index of smaller element
    pivot = arr[high].fitness  # pivot

    for j in range(low, high):

        # If current element is smaller than the pivot
        if arr[j].fitness > pivot:
            # increment index of smaller element
            i = i + 1

            # swap
            arr[i], arr[j] = arr[j], arr[i]

    # swap
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return i + 1


def quicksort(population: typing.List[Chromosome], low: int, high: int) -> typing.List[Chromosome]:
    """quick sorts a population in descending order of fitness"""

    if low < high:
        # pi is partitioning index, arr[p] is now at right place
        pi = partition(population, low, high)

        # Separately sort elements before partition and after partition
        quicksort(population, low, pi - 1)
        quicksort(population, pi + 1, high)

    return population


def selection(population: typing.List[Chromosome], n: int) -> typing.List[Chromosome]:
    """Selects the fittest n individuals of a population."""

    # sort by fitness
    sorted_population = quicksort(population, 0, len(population) - 1)

    # return the best half
    return sorted_population[:n]


# fisher-yates (google for C implementation).
def shuffle(array, n):
    arr = array[:]
    for i in range(n-1,0,-1):
        j = randint(0,i)
        arr[i],arr[j] = arr[j],arr[i]
    return arr


def bt_selection(population: typing.List[Chromosome], n: int) -> typing.List[typing.Tuple[Chromosome]]:
    """Selects n individuals from a population using binary tournament selection"""
    if (n % 2):
        n -= 1

    pairs = []
    for _ in range(n//2):
        # select first
        c1, c2, c3, c4 = shuffle(population, len(population))[:4]
        parent1 = c1 if (c1.fitness>c2.fitness) else c2
        parent2 = c3 if (c3.fitness>c4.fitness) else c4
        pairs.append((parent1, parent2))

    return pairs


@dataclass
class Assessor:
    """Assesses a chromosome. Placeholder."""
    l_target: float
    f_target: float
    l_penalty_c: float
    f_penalty_c: float

    def __post_init__(self):
        self.max_lat_target = 1/self.f_target
        self.s1, self.s2, self.s3 = build_perf_predictors()

    def penalty_l(self, latency):
        return max(0, self.l_penalty_c * (latency - self.l_target))

    def penalty_f(self, max_lat):
        return max(0, self.f_penalty_c * (max_lat - self.max_lat_target))

    def objective(self, latency, max_lat, power):
        return power + self.penalty_l(latency) + self.penalty_f(max_lat)

    def assess(self, chromosome: Chromosome) -> float:
        """Assesses a chromosome."""
        try:
            total_lat, max_lat = predict_performance(chromosome, self.s1, self.s2, self.s3)
        except RuntimeError:
            print("assess RuntimeError print", str(chromosome))
            raise RuntimeError
        power = predict_power(chromosome)
        # print("total max power:", total_lat, max_lat, power)
        res = self.objective(total_lat, max_lat, power)
        # print("res:", res)
        return -res


def genetic_algorithm(population_size: int, #mutation_rate: int,
        target_latency: int, target_fps: int, time_limit: float,
        staleness_limit: int, save: bool = True, warm: str = None) -> Chromosome:
    """Runs the genetic algorithm."""
    end_time = time() + time_limit

    # initialize assessor
    assessor = Assessor(target_latency, target_fps, LATENCY_PENALTY, FPS_PENALTY)  # function pointer
                                                  # ^ play around with these values!

    # initialize population
    population = initialize_population(population_size, assessor, warm)

    last_update = 0
    best_fitness = -inf

    # run until improvement stops or time limit reached.
    dbg_idx = 0
    while last_update < staleness_limit and time() < end_time:
        last_update += 1
        dbg_idx += 1
        if not dbg_idx % 5:
            print("generation:", dbg_idx)
        # select parents
        parents = bt_selection(population, population_size//2)

        random_individual = population[9]

        # create children
        children = [] # C: chromosome *children[n]: {0}
        for i in range((population_size//2) // 2): # aka >>2
            children += crossover(*parents[i], random_individual)

        # print("c:", len(children))
        # mutate children
        # for j, c in enumerate(children):
            # print(f"child {j}:", str(c))


        for i in range(len(children)):
            children[i] = mutate(children[i]) #, mutation_rate)

        # record child fitness
        assess_population(children, len(children), assessor)
        # print('assessed children of gen', dbg_idx)
        # select survivors
        population = selection(population,population_size-len(children)) + children
        # C: replace list with sorted list and free all non-surviving.

        best = selection(population, population_size)[0]
        if best.fitness > best_fitness:
            last_update = 0
            best_fitness = best.fitness
            # for c in population:
            #     print(str(c))
            print("new most fit individual:", str(best), f"fitness={best.fitness}")
    if last_update >= staleness_limit:
        print("staleness limit reached")
    if time() >= end_time:
        print("time limit reached")

    # return best chromosome
    population = selection(population, population_size)
    record_fitness = 0
    if save == "auto":
        with open("ga_population.txt", "r") as f:
            record_fitness = float(f.readline().strip())
    if save == "force" or save == "auto" and best_fitness > record_fitness:
        with open("ga_population.txt", "w") as f:
            f.write(f"{best_fitness}\n")
            f.write("\n".join([chro.__str__() for chro in population]))
    # for c in population:
    #     print(str(c))
    return population[0]

def chromosome_to_config(chromosome: Chromosome):
    """
    Chromosome -> pp1, pp2, bfreq, lfreq
    """
    pp1 = chromosome[0].layers
    pp2 = pp1 + chromosome[1].layers
    bfreq = BigFrequency[chromosome[0].frequency_level]
    lfreq = LittleFrequency[chromosome[2].frequency_level]
    return pp1, pp2, bfreq, lfreq

# dbg
if __name__ == "__main__":
    pop_size = POPULATION_SIZE
    target_lat = 120
    target_fps = 10
    t_limit = 3*60
    s_limit = 50
    res = genetic_algorithm(pop_size, target_lat, target_fps, t_limit, s_limit)
    print(res)