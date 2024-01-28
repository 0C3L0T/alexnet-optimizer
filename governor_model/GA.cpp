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
    float fitness = 0.0;
} chromosome;

// TODO: implement fitness function
float fitness(chromosome* c) {
    return 0.0;
}

chromosome* create_random_chromosome() {
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

    // create chromosome
    chromosome* c = (chromosome*) malloc(sizeof(chromosome));
    c->genes[0] = big_gene;
    c->genes[1] = gpu_gene;
    c->genes[2] = little_gene;

    return c;
}

void free_chromosome(chromosome* c) {
    for (int i = 0; i < 3; i++) {
        free(c->genes[i]);
    }

    free(c);
}

// pointer to chromosome array
chromosome** initialize_population (int size) {
    chromosome** chromosomes = (chromosome**) malloc(sizeof(chromosome*) * size);

    for (int i = 0; i < size; i++) {
        chromosomes[i] = create_random_chromosome();
    }

    return chromosomes;
}

void free_population(chromosome** population, int size) {
    for (int i = 0; i < size; i++) {
        free_chromosome(population[i]);
    }

    free(population);
}

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

// in-place crossover
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

void mutate(chromosome* c) {
    mutate_layer_size(c);
    mutate_frequency(c);
}

void asses_population(chromosome** population, int size, float (*fitness)(chromosome*)) {
    for (int i = 0; i < size; i++) {
        population[i]->fitness = fitness(population[i]);
    }
}

// sort in descending order
int partition(chromosome** population, int low, int high) {
    float pivot = population[high]->fitness;
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (population[j]->fitness >= pivot) {
            i++;
            chromosome* temp = population[i];
            population[i] = population[j];
            population[j] = temp;
        }
    }

    chromosome* temp = population[i + 1];
    population[i + 1] = population[high];
    population[high] = temp;

    return (i + 1);
}

void quick_sort(chromosome** population, int low, int high) {
    if (low < high) {
        int pi = partition(population, low, high);

        quick_sort(population, low, pi - 1);
        quick_sort(population, pi + 1, high);
    }
}

// fisher-yates shuffle
void shuffle(chromosome** population, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        chromosome* temp = population[i];
        population[i] = population[j];
        population[j] = temp;
    }
}


// return pointers to first n chromosomes from population
chromosome** selection(chromosome** population, int n) {
    chromosome** selected = (chromosome**) malloc(sizeof(chromosome*) * n);

    for (int i = 0; i < n; i++) {
        selected[i] = population[i];
    }

    return selected;
}


// return pointer to size/2 winners
chromosome** bt_selection(chromosome** population, int size) {
    // array of winners
    chromosome** winners = (chromosome**) malloc(sizeof(chromosome*) * (size/2));

    for (int i = 0; i < size/2; i++) {
        shuffle(population, size);
        chromosome* c1 = population[0];
        chromosome* c2 = population[1];
        chromosome* c3 = population[2];
        chromosome* c4 = population[3];

        chromosome* p1 = (c1->fitness > c2->fitness) ? c1 : c2;
        chromosome* p2 = (c3->fitness > c4->fitness) ? c3 : c4;

        // copy winners into array
        winners[i * 2] = p1;
        winners[i * 2 + 1] = p2;
    }


    return winners;
}

chromosome genetic_algorithm(int population_size, // HAS TO BE EVEN
                            int target_latency,
                            int target_fps,
                            int staleness_limit,
                            float (*fitness)(chromosome*)
                            ) {
    chromosome** population = initialize_population(population_size);
    asses_population(population, population_size, fitness);
    quick_sort(population, 0, population_size - 1);

    int last_update = 0;
    float best_fitness = 0.0;

    while (last_update < staleness_limit) {
        chromosome** best_half = selection(population, population_size/2); // array of size/2 best chromosomes pointers
        chromosome** parents = bt_selection(population, population_size); // array of size/2 winners pointers
 
        // override original population with new population
        for (int i = 0; i < population_size/2; i++) {
            *population[i] = *parents[i];
            *population[population_size - 1 - i] = *best_half[i];
        }

        // mutate winners of tournament
        for (int i = 0; i < population_size/2; i+2) {
            crossover(population[i], population[i + 1]);
            mutate(population[i]);
            mutate(population[i + 1]);
        }

        asses_population(population, population_size, fitness);
        quick_sort(population, 0, population_size - 1);

        chromosome* best = population[0];

        if (best->fitness > best_fitness) {
            best_fitness = best->fitness;
            last_update = 0;
        } else {
            last_update++;
        }
    }

    return *population[0];
}

int main() {
    int population_size = 10;

    chromosome c = genetic_algorithm(population_size, 100, 5, 10, fitness);
}