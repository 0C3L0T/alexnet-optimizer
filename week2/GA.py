import typing
from dataclasses import dataclass
from enum import Enum
from random import randrange, randint, shuffle

NETWORK_SIZE = 8


class ComponentType(Enum):
    """The type of component."""
    BIG = 1
    GPU = 2
    LITTLE = 3


LittleFrequency = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000]

BigFrequency = [500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000] # last two not used


# @dataclass
# class Gene:
#     """A gene is a single item in a chromosome. It corresponds to a single layer in the network."""
#     componentType: ComponentType
#     frequency: int


@dataclass
class Chromosome:
    """A chromosome is a list of 11 genes."""
    stage1_part: ComponentType
    stage2_part: ComponentType
    stage3_part: ComponentType

    # if pp1 = pp2 = NETWORK_size, then there is one stage
    # if pp1 = pp2 != NETWORK_size, then there are two stages
    # if pp1 < pp2, then there are three stages
    partitionPoint_1: int
    partitionPoint_2: int

    # if the GPU is in the pipeline, it will be on
    little_frequency: int
    big_frequency: int


def create_random_chromosome() -> Chromosome:
    """Create a random chromosome, make sure that the order of the components is consistent"""

    # random partition point where p2 > p1
    partitionPoint_1 = randint(1, NETWORK_SIZE - 1)
    partitionPoint_2 = randint(partitionPoint_1, NETWORK_SIZE - 1)

    # random frequencies
    little_frequency = LittleFrequency[randint(0, len(LittleFrequency) - 1)]
    big_frequency = BigFrequency[randint(0, len(BigFrequency) - 1)]

    components = list(ComponentType)

    # random order of components
    # shuffle(components)

    return Chromosome(
        components[0],
        components[1],
        components[2],
        partitionPoint_1,
        partitionPoint_2,
        little_frequency,
        big_frequency
    )


def initialize_population(population_size: int) -> typing.List[Chromosome]:
    """Initialize a population of chromosomes."""
    population = []

    for _ in range(population_size):
        population.append(create_random_chromosome())

    return population


def crossover(a: Chromosome, b: Chromosome) -> Chromosome:
    """Performs crossover between two chromosomes."""

    if randint(0, 100) < 50:
        # take partition points from a and frequencies from b
        return Chromosome(
            a.stage1_part,  # stages don't change
            a.stage2_part,
            a.stage3_part,
            a.partitionPoint_1,
            a.partitionPoint_2,
            b.little_frequency,
            b.big_frequency
        )
    else:
        # take partition points from b and frequencies from a
        return Chromosome(
            b.stage1_part,  # stages don't change
            b.stage2_part,
            b.stage3_part,
            b.partitionPoint_1,
            b.partitionPoint_2,
            a.little_frequency,
            a.big_frequency
        )


def mutate(individual: Chromosome, mutation_rate) -> Chromosome:
    """Performs mutation on a chromosome. Takes a mutation rate 0-100"""

    # partition point mutation
    if randint(0, 100) < mutation_rate:
        individual = mutate_partition_point(individual, mutation_rate)

    # frequency mutation
    if randint(0, 100) < mutation_rate:
        individual = mutate_frequency(individual)

    return individual


def mutate_partition_point(individual: Chromosome, mutation_rate: int) -> Chromosome:
    """Mutates the partition point of a chromosome."""
    # random partition point where p2 > p1

    if individual.partitionPoint_1 == individual.partitionPoint_2 == NETWORK_SIZE:
        # there is one stage, keep it
        return individual

    if individual.partitionPoint_1 == individual.partitionPoint_2 != NETWORK_SIZE:
        # there are two stages, move the partition point
        individual.partitionPoint_1 =






def mutate_frequency(individual: Chromosome) -> Chromosome:
    pass

def fitness(chromosome: Chromosome) -> float:
    """Computes the fitness of a chromosome."""
    return 0.0
