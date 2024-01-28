import typing
from PerformancePredictor import predict_performance, build_perf_predictors
from PowerPredictor import predict_power
from time import time
from dataclasses import dataclass
from enum import Enum
from random import randint
from numpy import inf

# HYPERPARAMETERS WE SHOULD TEST
LATENCY_PENALTY = 100
FPS_PENALTY = 100
POPULATION_SIZE = 10


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


def cure_child_cancer(child: Chromosome):
    while layer_err(child):    # in C: while (int err = sgn(layer_err(child1)))
        err = SGN(layer_err(child))
        correction_layer = randint(0,2)
        child.genes[correction_layer].layers -= err

        # if child.genes[correction_layer].frequency_level != None:
        #     child.genes[correction_layer].frequency_level -= err
        # else: # this freq level code is not ready yet btw, it's not clamped
        #     child.genes[0].frequency_level -= err


def crossover(a: Chromosome, b: Chromosome) -> Chromosome:
    """Performs one-point crossover between two chromosomes."""

    cut_point = randint(0,1)+1
    genes1 = a.genes[:cut_point] + b.genes[cut_point:]
    genes2 = b.genes[:cut_point] + a.genes[cut_point:]
    child1 = Chromosome(genes1)
    child2 = Chromosome(genes2)

    cure_child_cancer(child1)
    cure_child_cancer(child2)

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
    limit = gene.componentType == ComponentType.IBG and len(BigFrequency) or len(LittleFrequency)

    # beware of boundaries
    if gene.frequency_level + change > limit or gene.frequency_level + change < 0:
        change *= -1

    # apply change
    gene.frequency_level += change
    individual[rand_idx] = gene

    return individual


def fitness(assessor, chromosome: Chromosome) -> float:
    """Computes the fitness of a chromosome. assessor will be a function pointer"""
    return assessor.assess(chromosome)


def assess_population(population: typing.List[Chromosome], size: int, assessor) -> None:
    """Computes the fitness of an entire population."""
    for i in range(size):
        population[i].fitness = fitness(population[i], assessor)


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
        j = randint(0,i+1)
        arr[i],arr[j] = arr[j],arr[i]
    return arr


def bt_selection(population: typing.List[Chromosome], n: int) -> typing.List[typing.Tuple[Chromosome]]:
    """Selects n individuals from a population using binary tournament selection"""
    if (n % 2):
        n -= 1

    pairs = []
    for _ in range(n//2):
        # select first
        c1, c2, c3, c4 = shuffle(population)[:4]
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
        power + self.penalty_l(latency) + self.penalty_f(max_lat)
        pass

    def assess(self, chromosome: Chromosome) -> float:
        """Assesses a chromosome."""
        total_lat, max_lat = predict_performance(chromosome, self.s1, self.s2, self.s3)
        power = predict_power(chromosome)

        return -Assessor.objective(total_lat, max_lat, power)


def genetic_algorithm(population_size: int, #mutation_rate: int,
        target_latency: int, target_fps: int, time_limit: float,
        staleness_limit: int, save: bool = True, warm: str =None) -> Chromosome:
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
    while last_update < staleness_limit and time() < end_time:
        last_update += 1

        # select parents
        parents = bt_selection(population, population_size//2)

        # create children
        children = [] # C: chromosome *children[n]: {0}
        for i in range((population_size//2) // 2): # aka >>2
            children += crossover(*parents[i])

        # mutate children
        for i in range(population_size // 2):
            children[i] = mutate(children[i]) #, mutation_rate)

        # record child fitness
        assess_population(children)

        # select survivors
        population = selection(population,population_size-len(children)) + children
        # C: replace list with sorted list and free all non-surviving.

        best = selection(population, population_size)[0]
        if best.fitness > best_fitness:
            last_update = 0
            best_fitness = best.fitness

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
    return population[0]


# dbg
if __name__ == "__main__":
    population = initialize_population(20)
    pstr1 = str(population)
    with open("ga_population.txt", "w") as f:
        f.write("nonsense\n")
        f.write("\n".join([str(chro) for chro in population]))


    # print(list(map(predict_performance, population)))
    population = []
    population = load_population("ga_population.txt")
    # population = eval(repr(population))
    print(pstr1 == str(population))
