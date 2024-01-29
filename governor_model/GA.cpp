//
// Created by ocelot on 1/28/24.
//

#include <stdlib.h>
#include <time.h>

int NETWORK_SIZE = 8;

enum component_type {
    BIG,
    GPU,
    LITTLE,
};

int littleFrequency[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000};
int bigFrequency[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000,
                2100000, 2208000};

typedef struct {
    component_type type;
    int layers;
    int frequency_level;
} gene;


gene* create_gene(component_type type, int layers, int frequency_level) {
    gene* g = (gene*) malloc(sizeof(gene));
    g->type = type;
    g->layers = layers;
    g->frequency_level = frequency_level;
    return g;
}

typedef struct {
    gene* genes[3];
    float fitness;
} chromosome;

typedef chromosome* population;

// TODO: implement fitness function
float fitness(chromosome* c) {
    return 0.0;
}

/***
 * create chromosome struct with random configuration but fixed order
 * @return
 */
chromosome create_random_chromosome() {
    // random partition points
    int ran1 = (rand() % NETWORK_SIZE - 1) + 1;
    int ran2 = (rand() % NETWORK_SIZE - 1) + 1;

    int rans[] = {ran1, ran2};

    int order = (int) (ran1 > ran2);

    int pp1 = rans[order];
    int pp2 = rans[1 - order];

    // random frequencies
    int littleFrequencyLevel = (rand() % 9);
    int bigFrequencyLevel = (rand() % 13);

    // create genes
    gene* big_gene = create_gene(BIG, pp1, bigFrequencyLevel);
    gene* gpu_gene = create_gene(GPU, pp2 - pp1, bigFrequencyLevel);
    gene* little_gene = create_gene(LITTLE, NETWORK_SIZE - pp2, littleFrequencyLevel);

    chromosome c;

    // create chromosome
    c.genes[0] = big_gene;
    c.genes[1] = gpu_gene;
    c.genes[2] = little_gene;
    c.fitness = 0.0;

    return c;
}

void free_chromosome_genes(chromosome c) {
    for (int i = 0; i < 3; i++) {
        free(c.genes[i]);
    }
}

/***
 * fill pre-allocated chromosome array of size with random chromosomes
 * @param population
 * @param size
 */
void initialize_population (population population, int size) {
    for (int i = 0; i < size; i++) {
        population[i] = create_random_chromosome();
    }
}

void free_population(population population, int size) {
    for (int i = 0; i < size; i++) {
        free_chromosome_genes(population[i]);
    }

    free(population);
}

/***
 * @param c chromosome pointer
 * @return the amount of layers above network size in chromosome
 */
int layer_err(chromosome* c) {
    return c->genes[0]->layers + c->genes[1]->layers + c->genes[2]->layers - NETWORK_SIZE;
}

int sign(int x) {
    return (x > 0) - (x < 0);
}

void cure_child_cancer(chromosome* c) {
    while (int err = sign(layer_err(c))) {
        int gene_index = rand() % 3;
        c->genes[gene_index]->layers -= err;
    }
}

/***
 * in-place chromosome crossover
 * @param c1 chromosome pointer
 * @param c2 chromosome pointer
 */
void crossover(chromosome* c1, chromosome* c2) {
    int cut_point = rand() % 2 + 1;

    for (int i = 0; i < cut_point; i++) {
        gene* temp = c1->genes[i];
        c1->genes[i] = c2->genes[i];
        c2->genes[i] = temp;
    }

    cure_child_cancer(c1);
    cure_child_cancer(c2);
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

void mutate_layer_size(chromosome* c) {
    // first or second partition point
    int pp = rand() % 1;

    // change between -1 and 1
    int change = (rand() % 2) - 1;

    // 20% chance to change more
    while (!(random() % 5)) {
        change += sign(change);
        if (abs(change)  > 2) {
            break;
        }
    }

    // clamp to bounds
    if (pp) {
        change -= min(0, c->genes[1]->layers + change); // L2 min size 0
        change -= max(0, c->genes[0]->layers + c->genes[1]->layers + change - NETWORK_SIZE); //L3 min size 0
    } else {
        change -= min(0, c->genes[0]->layers + change - 1); // L1 min size 1
        change -= max(0, c->genes[1]->layers - change); // L2 min size 0
    }

    // apply change
    c->genes[pp]->layers += change;
    c->genes[pp + 1]->layers -= change;

    cure_child_cancer(c);
}

void mutate_frequency(chromosome* c) {
    // random gene (no GPU)
    int index = (rand() % 2) * 2;
    gene* g = c->genes[index];

    // change between -1 and 1
    int change = (rand() % 2) - 1;

    int limit = (g->type == BIG) ? 13 : 9;

    // bound check
    if (g->frequency_level + change < 0 || g->frequency_level + change >= limit) {
        change *= -1;
    }

    // apply change
    g->frequency_level += change;
}

// pointer to chromosome
void mutate(chromosome* c) {
    mutate_layer_size(c);
    mutate_frequency(c);
}

/***
 * Apply fitness function to every individual of population of size
 * @param population chromosome array
 * @param size int
 * @param fitness function pointer that takes chromosome pointer and outputs float
 */
void asses_population(population population, int size, float (*fitness)(chromosome*)) {
    for (int i = 0; i < size; i++) {
        population[i].fitness = fitness(&population[i]);
    }
}

/***
 * pivot for quick sort, sort on fitness in descending order
 * @param population chromosome array
 * @param low
 * @param high
 * @return new pivot
 */
int partition(population population, int low, int high) {
    float pivot = population[high].fitness;
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (population[j].fitness >= pivot) {
            i++;
            chromosome temp = population[i];
            population[i] = population[j];
            population[j] = temp;
        }
    }

    chromosome temp = population[i + 1];
    population[i + 1] = population[high];
    population[high] = temp;

    return (i + 1);
}

/***
 * quick sort chromosome array based on fitness
 * @param population
 * @param low
 * @param high
 */
void quick_sort(population population, int low, int high) {
    if (low < high) {
        int pi = partition(population, low, high);

        quick_sort(population, low, pi - 1);
        quick_sort(population, pi + 1, high);
    }
}

/***
 * fisher-yates shuffle
 * @param population chromosome array
 * @param size
 */
void shuffle(population population, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        chromosome temp = population[i];
        population[i] = population[j];
        population[j] = temp;
    }
}

/***
 * select half of the population of size trough binary tournament and copy into parents array
 * @param population chromosome array
 * @param parents chromosome array
 * @param size
 */
void bt_selection(population population, chromosome* parents, int size) {
    for (int i = 0; i < size/2; i++) {
        shuffle(population, size);
        chromosome c1 = population[0];
        chromosome c2 = population[1];
        chromosome c3 = population[2];
        chromosome c4 = population[3];

        chromosome p1 = (c1.fitness > c2.fitness) ? c1 : c2;
        chromosome p2 = (c3.fitness > c4.fitness) ? c3 : c4;

        // copy values
        parents[i * 2] = p1;
        parents[i * 2 + 1] = p2;
    }
}

chromosome genetic_algorithm(int population_size, // HAS TO BE EVEN
                            int target_latency,
                            int target_fps,
                            int staleness_limit,
                            float (*fitness)(chromosome*)
                            ) {
    chromosome* population = (chromosome*) malloc(sizeof(chromosome) * population_size);
    chromosome* parents = (chromosome*) malloc(sizeof(chromosome) * population_size/2);

    initialize_population(population, population_size);
    asses_population(population, population_size, fitness);

    int last_update = 0;
    float best_fitness = 0.0;

    while (last_update < staleness_limit) {
        // selected individuals are copied to parents array
        bt_selection(population, parents, population_size);

        // sort population
        asses_population(population, population_size, fitness);
        quick_sort(population, 0, population_size - 1);

        // parents in-place crossover to children
        for (int i = 0; i < population_size/2; i += 2) {
            crossover(&parents[i], &parents[i+1]);
        }

        // free bottom half and override with children
        for (int i = 0; i < population_size/2; i++) {
            free_chromosome_genes(population[population_size - i]);
            population[population_size - 1] = parents[i];
        }

        // check staleness
        chromosome best = population[0];
        if (best.fitness > best_fitness) {
            best_fitness = best.fitness;
            last_update = 0;
        } else {
            last_update++;
        }
    }

    // free memory
    free_population(population, population_size);
    free_population(parents, population_size/2);

    return population[0];
}

int main() {
    int population_size = 10;

    chromosome c = genetic_algorithm(population_size, 100, 5, 10, fitness);
}