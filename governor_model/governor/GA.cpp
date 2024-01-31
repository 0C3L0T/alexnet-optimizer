//
// Created by ocelot on 1/28/24.
//

#include "GA.h"

#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "GA.h"
#include "algorithm"
#include "fitness.h"
#include <vector>

using namespace std;


const int littleFrequency[] = { 500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000 };
const int bigFrequency[]    = { 500000,  667000,  1000000, 1200000, 1398000, 1512000, 1608000,
                                1704000, 1800000, 1908000, 2016000, 2100000, 2208000 };

gene* create_gene(component_type type, int layers, int frequency_level) {
  gene* g            = (gene*) malloc(sizeof(gene));
  g->type            = type;
  g->layers          = layers;
  g->frequency_level = frequency_level;
  return g;
}

typedef chromosome* population;

/***
 * create chromosome struct with random configuration but fixed order
 * @return
 */
chromosome create_random_chromosome() {
  // random partition points
  int ran1 = (rand() % NETWORK_SIZE - 1) + 1;
  int ran2 = (rand() % NETWORK_SIZE - 1) + 1;

  int rans[] = { ran1, ran2 };

  int order = (int) (ran1 > ran2);

  int pp1 = rans[order];
  int pp2 = rans[1 - order];

  // random frequencies
  int littleFrequencyLevel = (rand() % 9);
  int bigFrequencyLevel    = (rand() % 13);

  // create genes
  gene* big_gene    = create_gene(BIG, pp1, bigFrequencyLevel);
  gene* gpu_gene    = create_gene(GPU, pp2 - pp1, bigFrequencyLevel);
  gene* little_gene = create_gene(LITTLE, NETWORK_SIZE - pp2, littleFrequencyLevel);

  chromosome c;

  // create chromosome
  c.genes[0] = big_gene;
  c.genes[1] = gpu_gene;
  c.genes[2] = little_gene;
  c.fitness  = 0.0;

  return c;
}

int chromosome_operator_equal(chromosome* a, chromosome* b) {
  // Return 1 if the two chromosomes are equivalent, 0 otherwise
  // Compare genes[*].layers, genes[0].frequency_level, genes[2].frequency_level

  // Handle invalid input
  if (a == NULL || b == NULL) return 0;

  for (int i = 0; i < 3; ++i) {
    gene* g1 = a->genes[i];
    gene* g2 = b->genes[i];

    // Handle invalid input
    if (g1 == NULL || g2 == NULL) return 0;
    if (g1->layers != g2->layers) return 0;
  }

  if (a->genes[0]->frequency_level != b->genes[0]->frequency_level) return 0;
  if (a->genes[2]->frequency_level != b->genes[2]->frequency_level) return 0;

  return 1;
}

string componentTypeToString(component_type type) {
    switch (type) {
        case BIG: return "BIG";
        case GPU: return "GPU";
        case LITTLE: return "LITTLE";
        default: return "UNKNOWN";
    }
}

string chromosomeToString(chromosome chromo) {
    ostringstream oss;

    // Convert genes to string
    for (int i = 0; i < 3; ++i) {
        oss << "Gene " << i + 1 << ": ";
        if (chromo.genes[i] != nullptr) {
            oss << "Type: " << componentTypeToString(chromo.genes[i]->type) << ", "
                << "Layers: " << chromo.genes[i]->layers << ", "
                << "Frequency Level: " << chromo.genes[i]->frequency_level;
        } else {
            oss << "NULL";
        }
        oss << "\n";
    }

    // Convert fitness and estimates to string
    oss << "Fitness: " << chromo.fitness << "\n";
    oss << "Estimated Latency: " << chromo.est_lat << "\n";
    oss << "Estimated FPS: " << chromo.est_fps << "\n";
    oss << "Estimated Power: " << chromo.est_pwr << "\n";

    return oss.str();
}

void free_chromosome_genes(chromosome c) {
  for (int i = 0; i < 3; i++) {
    free(c.genes[i]);
  }
}

int is_duplicate(chromosome* population, int size, chromosome* c) {
  for (int i = 0; i < size; i++) {
    if (chromosome_operator_equal(&population[i], c)) {
      return 1;
    }
  }

  return 0;
}

/***
 * fill pre-allocated chromosome array of size with random chromosomes
 * @param population
 * @param size
 */
void initialize_population(population population, int size) {
    cout << "Initializing population.." << endl;
    int idx = 0;
    chromosome c;

    while (idx < size) {
        c = create_random_chromosome();

        if (!is_duplicate(population, idx, &c)) {
            population[idx] = c;
            idx++;
        }
    }
    cout << "Population initialized.." << endl;
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
    cout << "Calculating layer error.." << endl;
    return c->genes[0]->layers + c->genes[1]->layers + c->genes[2]->layers - NETWORK_SIZE;
}

int sign(int x) {
  return (x > 0) - (x < 0);
}

void cure_child_cancer(chromosome* c) {
    cout << "Curing child cancer.." << endl;

    while (int err = sign(layer_err(c))) {
        int gene_index = rand() % 3;
        c->genes[gene_index]->layers -= err;
    }

    cout << "Child cancer cured.." << endl;
}

/***
 * in-place chromosome crossover
 * @param c1 chromosome pointer
 * @param c2 chromosome pointer
 */
void crossover(chromosome* c1, chromosome* c2) {
    cout << "Crossover.." << endl;
    cout << chromosomeToString(*c1) << endl;
    cout << chromosomeToString(*c2) << endl;
    int cut_point = rand() % 2 + 1;

    for (int i = 0; i < cut_point; i++) {
        gene* temp   = c1->genes[i];
        c1->genes[i] = c2->genes[i];
        c2->genes[i] = temp;
    }

    cure_child_cancer(c1);
    cure_child_cancer(c2);
    cout << "Crossover done.." << endl;
}

void mutate_layer_size(chromosome* c) {
  // first or second partition point
  int pp = rand() % 1;

  // change between -1 and 1
  int change = (rand() % 3) - 1;

  // 20% chance to change more
  while (!(random() % 5)) {
    change += sign(change);
    if (abs(change) > 2) {
      break;
    }
  }

  // clamp to bounds
  if (pp) {
    change -= std::min(0, c->genes[1]->layers + change);                                       // L2 min size 0
    change -= std::max(0, c->genes[0]->layers + c->genes[1]->layers + change - NETWORK_SIZE);  // L3 min size 0
  } else {
    change -= std::min(0, c->genes[0]->layers + change - 1);  // L1 min size 1
    change += std::max(0, c->genes[1]->layers - change);      // L2 min size 0
  }

  // apply change
  c->genes[pp]->layers += change;
  c->genes[pp + 1]->layers -= change;

  cure_child_cancer(c);
}

void mutate_frequency(chromosome* c) {
  // random gene (no GPU)
  int   index = (rand() % 2) * 2;
  gene* g     = c->genes[index];

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
    cout << "Mutating.." << endl;
    cout << chromosomeToString(*c) << endl;

    mutate_layer_size(c);
    mutate_frequency(c);

    cout << "Mutated.." << endl;
    cout << chromosomeToString(*c) << endl;
}


/***
 * Apply fitness function to every individual of population of size
 * @param population chromosome array
 * @param size int
 * @param fitness function pointer that takes chromosome pointer and outputs float
 */
void assess_population(population population, int size, float (*fitness)(chromosome*)) {
    cout << "Assessing population.." << endl;
    for (int i = 0; i < size; i++) {
        population[i].fitness = fitness(&population[i]);
    }
    cout << "Population assessed.." << endl;
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
  int   i     = (low - 1);

  for (int j = low; j <= high - 1; j++) {
    if (population[j].fitness >= pivot) {
      i++;
      chromosome temp = population[i];
      population[i]   = population[j];
      population[j]   = temp;
    }
  }

  chromosome temp   = population[i + 1];
  population[i + 1] = population[high];
  population[high]  = temp;

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
    int        j    = rand() % (i + 1);
    chromosome temp = population[i];
    population[i]   = population[j];
    population[j]   = temp;
  }
}

/***
 * select half of the population of size trough binary tournament and copy into parents array
 * @param population chromosome array
 * @param parents chromosome array
 * @param size
 */
void bt_selection(population population, vector<chromosome> *parents, int size) {
    cout << "Selecting parents.." << endl;

    for (int i = 0; i < size / 4; i++) {
        shuffle(population, size);
        chromosome c1 = population[0];
        chromosome c2 = population[1];
        chromosome c3 = population[2];
        chromosome c4 = population[3];

        chromosome p1 = (c1.fitness > c2.fitness) ? c1 : c2;
        chromosome p2 = (c3.fitness > c4.fitness) ? c3 : c4;

        // copy values
        parents->push_back(p1);
        parents->push_back(p2);
    }

    cout << "Parents size after selection: " << parents->size() << endl;

    cout << "Parents selected.." << endl;
}


int cross_dup_check(chromosome* population, int n, chromosome* c) {
    for (int i = 0; i < n; i++) {
        if (chromosome_operator_equal(&population[i], c)) {
            return 1;
        }
    }

    return 0;
}


void make_children(vector<chromosome> *parents, chromosome* population,
                   int population_size) {
    cout << "Making children.." << endl;
    cout << "Parents size: " << parents->size() << endl;

    for (int i = 0; i < parents->size(); i += 2) {
        crossover(&(*parents)[i], &(*parents)[i + 1]);
        mutate(&(*parents)[i]);
        mutate(&(*parents)[i + 1]);

        // remove duplicates
        if (cross_dup_check(population, population_size, &(*parents)[i])) {
            cout << "removing duplicate " << i << endl;
            free_chromosome_genes((*parents)[i]);
            parents->erase(parents->begin() + i);
            i--;
        }
    }

    cout << "Children made.." << endl;
}

// placeholder fitness function
float fitness_function(chromosome* c) {
    return 1.0;
}

chromosome genetic_algorithm(int population_size,  // HAS TO BE EVEN
                             int target_latency,
                             int target_fps,
                             int staleness_limit
                            ) {
  chromosome* population = (chromosome*) malloc(sizeof(chromosome) * population_size);
  vector<chromosome> *parents = new vector<chromosome>();


  initialize_population(population, population_size);
  assess_population(population, population_size, fitness_function);

  int   last_update             = 0;
  float best_fitness            = 0.0;
  int   dbg                     = 0;

  chromosome best_chromosome = population[0];

  while (last_update < staleness_limit) {
    // fill parents with best half
    bt_selection(population, parents, population_size);

    // turn parents vector into children
    make_children(parents, population, population_size);

    // sort population
    assess_population(population, population_size, fitness_function);
    quick_sort(population, 0, population_size - 1);

    cout << "replacing worst population" << endl;
    // remove worst population and replace with children
    for(int i = 0; i < parents->size(); i++) {
        cout << "replacing " << i << endl;
        free_chromosome_genes(population[population_size - 1 - i]);
        population[population_size - 1 - i] = (*parents)[i];
    }

    cout << "testing for staleness" << endl;
    // test for staleness
    if (population[0].fitness > best_fitness) {
      best_fitness = population[0].fitness;
      last_update  = 0;
      best_chromosome = population[0];
    } else {
      last_update++;
    }
  }

  // free memory
  free_population(population, population_size);
  free(parents);

  return best_chromosome;
}
