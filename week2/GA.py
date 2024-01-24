import typing
from dataclasses import dataclass
from enum import Enum
from random import randrange, randint, shuffle

NETWORK_SIZE = 11


class ComponentType(Enum):
    """The type of component."""
    LITTLE = 1
    BIG = 2
    GPU = 3


LittleFrequency = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000]

BigFrequency = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000] # last two not used


@dataclass
class Gene:
    """A gene is a single item in a chromosome. It corrsponds to a single layer in the network."""
    componentType: ComponentType
    frequency: int


@dataclass
class Chromosome:
    """A chromosome is a list of 11 genes."""
    genes: typing.List[Gene]


def create_random_chromosome() -> Chromosome:
    """Create a random chromosome, make sure that the order of the components is consistent"""
    genes = []

    # random partition point where p2 > p1
    partitionPoint_1 = randrange(0, NETWORK_SIZE - 1)
    partitionPoint_2 = randrange(partitionPoint_1, NETWORK_SIZE - 1)

    # genes with random frequencies
    geneLittle = Gene(ComponentType.LITTLE, LittleFrequency[randrange(0, len(LittleFrequency) - 1)])
    geneBig = Gene(ComponentType.BIG, BigFrequency[randrange(0, len(BigFrequency) - 1)])
    geneGPU = Gene(ComponentType.GPU, randint(0, 1))

    # random order of components
    components = [geneLittle, geneBig, geneGPU]
    shuffle(components)

    # add genes to chromosome
    for _ in range(0, partitionPoint_1):
        genes.append(components[0])
    for _ in range(partitionPoint_1, partitionPoint_2):
        genes.append(components[1])
    for _ in range(partitionPoint_2, NETWORK_SIZE):
        genes.append(components[2])

    return Chromosome(genes)


def initialize_population(population_size: int) -> typing.List[Chromosome]:
    """Initialize a population of chromosomes."""
    population = []

    for _ in range(population_size):
        population.append(create_random_chromosome())

    return population


def crossover(chromosome1: Chromosome, chromosome2: Chromosome) -> Chromosome:
    """Performs crossover between two chromosomes."""
    crossover_point = randint(0, len(chromosome1.genes) - 1)
    child = Chromosome(chromosome1.genes[:crossover_point] + chromosome2.genes[crossover_point:])
    return child


def mutate(chromosome: Chromosome) -> Chromosome:
    """Performs mutation on a chromosome."""

def fitness(chromosome: Chromosome) -> float:
    """Computes the fitness of a chromosome."""
    return 0.0
