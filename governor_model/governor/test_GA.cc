#include "GA.h"

using namespace std;

int main() {

  chromosome c = genetic_algorithm(4, 400, 4, 4);

  cout << chromosomeToString(c) << endl;

  return 0;
}