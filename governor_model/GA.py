import typing
from dataclasses import dataclass
from enum import Enum
from random import randint

NETWORK_SIZE = 8


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
    little_frequency = LittleFrequency[randint(0, len(LittleFrequency) - 1)]
    big_frequency = BigFrequency[randint(0, len(BigFrequency) - 1)]

    big_gene = Gene(ComponentType.BIG, pp1, big_frequency)
    gpu_gene = Gene(ComponentType.GPU, pp2 - pp1, None)
    little_gene = Gene(ComponentType.LITTLE, NETWORK_SIZE - pp2, little_frequency)

    return Chromosome(
        [big_gene, gpu_gene, little_gene]
    )


def initialize_population(population_size: int) -> typing.List[Chromosome]:
    """Initialize a population of chromosomes."""
    population = []

    for _ in range(population_size):
        population.append(create_random_chromosome())

    return population


def crossover(a: Chromosome, b: Chromosome) -> Chromosome:
    """Performs crossover between two chromosomes."""

    gene_1 = b.genes[0] if randint(0, 1) == 0 else a.genes[0]
    gene_2 = b.genes[1] if randint(0, 1) == 0 else a.genes[1]
    gene_3 = b.genes[2] if randint(0, 1) == 0 else a.genes[2]

    return Chromosome([gene_1, gene_2, gene_3])


def mutate(individual: Chromosome, mutation_rate) -> Chromosome:
    """Performs mutation on a chromosome. Takes a mutation rate 0-100"""

    # partition point mutation
    if randint(0, 100) < mutation_rate:
        individual = mutate_layer_size(individual)

    # frequency mutation
    if randint(0, 100) < mutation_rate:
        individual = mutate_frequency(individual)

    return individual


def mutate_layer_size(individual: Chromosome) -> Chromosome:
    """Mutates the partition point of a chromosome."""

    # get first or second gene
    idx = randint(0, 1)

    # amount of change
    change = randint(-1, 1)

    # beware of boundaries
    if individual[idx].layers + change > NETWORK_SIZE or individual[idx].layers + change < 0:
        change *= -1

    # add change to one
    individual[idx].layers += change

    # subtract from the other, this will always work since the sum is always NETWORK_SIZE
    individual[idx + 1].layers -= change

    return individual


def mutate_frequency(individual: Chromosome) -> Chromosome:
    """Mutates the frequency of a gene in a chromosome."""

    # random gene
    rand_idx = randint(0, 2)
    gene = individual[rand_idx]

    # GPU has no frequency to mutate
    if gene.componentType == ComponentType.GPU:
        return individual

    # amount of change
    change = randint(-1, 1)

    # limit is the length of the frequency list
    limit = gene.componentType == ComponentType.BIG and len(BigFrequency) or len(LittleFrequency)

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


def selection(population: typing.List[Chromosome], assesor) -> typing.List[Chromosome]:
    """Selects the best chromosomes from a population."""

    # compute fitness
    for individual in population:
        individual.fitness = fitness(assesor, individual)

    # sort by fitness
    sorted_population = quicksort(population, 0, len(population) - 1)

    # return the best half
    return sorted_population[:len(population) // 2]


@dataclass
class Assessor:
    """Assesses a chromosome. Placeholder."""

    def assess(self, chromosome: Chromosome) -> float:
        """Assesses a chromosome."""
        pass


def genetic_algorithm(population_size: int, mutation_rate: int, iterations: int) -> Chromosome:
    """Runs the genetic algorithm."""

    # initialize population
    population = initialize_population(population_size)

    # initialize assessor
    assessor = Assessor()  # function pointer

    # run for n iterations
    for _ in range(iterations):

        # select parents
        parents = selection(population, assessor)

        # create children
        children = []
        for i in range(population_size // 2):
            children.append(crossover(parents[i], parents[population_size - i - 1]))

        # mutate children
        for i in range(population_size):
            children[i] = mutate(children[i], mutation_rate)

        # select survivors
        population = selection(children, assessor)

    # return best chromosome
    return population[0]
