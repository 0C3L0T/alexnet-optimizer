//
// Created by ocelot on 1/29/24.
//
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <array>

#include "GA.h"
#include "fitness.h"

using namespace std;

int LittleFrequencyTable[] = { 500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000 };
int BigFrequencyTable[]    = { 500000,  667000,  1000000, 1200000, 1398000, 1512000, 1608000,
                               1704000, 1800000, 1908000, 2016000, 2100000, 2208000 };

const float NUDGE = 1.3;
const float ANTI_NUDGE = 0.4;
const float ANTI_NUDGE_THRESH = 0.04;
const int REPETITION_LIMIT = 4;

void setup() {
  /* Export OpenCL library path */
  system("export LD_LIBRARY_PATH=/data/local/Working_dir");
  setenv("LD_LIBRARY_PATH", "/data/local/Working_dir", 1);

  /* Setup Performance Governor (CPU) */
  system("echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor");
  system("echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor");

  /* set fan speed to 100% */
  system("echo 1 > /sys/class/fan/enable");
  system("echo 0 > /sys/class/fan/mode");
  system("echo 4 > /sys/class/fan/level");
}

void scale_little(int freq) {
  string Command = "echo " + to_string(freq) + " > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq";
  system(Command.c_str());
}

void scale_big(int freq) {
  string Command = "echo " + to_string(freq) + " > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq";
  system(Command.c_str());
}

void run_graph(int pp1, int pp2, int big_freq, int little_freq) {
  scale_little(little_freq);
  scale_big(big_freq);

  int N_Frames = 20;

  char Run_Command[150];
  sprintf(Run_Command,
          "./graph_alexnet_all_pipe_sync --threads=4 --threads2=2 --target=NEON --n=%d --partition_point=%d "
          "--partition_point2=%d --order=B-G-L &> output.txt",
          N_Frames,
          pp1,
          pp2);
  system(Run_Command);
}

#define PARSEAMOUNT 2

array<string, PARSEAMOUNT> lineCharecteristics =
{
  "Frame rate is:",
  "Frame latency is:",
};

array<string, PARSEAMOUNT> outputLines =
{
  "fps:",
  "latency:",
};

string getOutput(int index, string text) {
  float floatOutput;
  if (stringstream(text) >> floatOutput) {
    return to_string(floatOutput);
  }
  return "";
}

// return 1 if target performance is met, 0 otherwise and write current to pointers
int check_performance(int target_fps, int target_latency, int* current_fps, int* current_latency) {
  ifstream myfile("/data/local/Working_dir/output.txt");

  int fps = 0;
  int latency = 0;

  /* Read Output.txt File and Extract Data */
  for (std::string line; getline(myfile, line);) {
    string temp;
    for (int i = 0; i < PARSEAMOUNT; i++) {
      if (line.find(lineCharecteristics[i]) == 0) {
        std::istringstream ss(line);
        while (!ss.eof()) {
          /* Extracting word by word from stream */
          ss >> temp;
          /* Checking the given word is float or not */
          string output = getOutput(i, temp);
          if (output != "") {
            if (i == 0) {
              fps = stoi(output);
            } else {
              latency = stoi(output);
            }
            break;
          }
        }
      }
    }
  }

  if (fps >= target_fps && latency <= target_latency) {
    return 1;
  } else {
    *current_fps = fps;
    *current_latency = latency;
    return 0;
  }
}

// optionally implement time
void govern(int target_latency, int target_fps, int population_size, int staleness_limit) {
  int adjusted_fps     = target_fps;
  int adjusted_latency = target_latency;

  int  stale_count     = 0;

  int pp1;
  int pp2;
  int big_freq;
  int little_freq;

  chromosome last_attempt;

  while (stale_count < staleness_limit) {
    // pass requirements to GA
    chromosome solution = genetic_algorithm(
          population_size,
          adjusted_latency,
          adjusted_fps,
          staleness_limit
        );

    // if solution is same as last solution, update staleness counter
    if (solution.fitness == last_attempt.fitness) {
      stale_count++;
    } else {
      last_attempt = solution;
      stale_count  = 0;
    }

    // get info from solution
    pp1         = solution.genes[0]->layers;
    pp2         = pp1 + solution.genes[1]->layers;
    big_freq    = BigFrequencyTable[solution.genes[0]->frequency_level];
    little_freq = LittleFrequencyTable[solution.genes[2]->frequency_level];

    printf("trying configuration: %d %d %d %d\n", pp1, pp2, big_freq, little_freq);
    run_graph(pp1, pp2, big_freq, little_freq);

    int current_fps;
    int current_latency;

    if (check_performance(target_fps, target_latency, &current_fps, &current_latency)) {
      cout << "Solution found." << endl;
      return;
    }

    string nudged = "";
    if (current_fps < target_fps) {
      adjusted_fps += (target_fps = current_fps) * NUDGE;
      nudged = "fps";
    } else if (current_latency > target_latency) {
      adjusted_latency -= (current_latency - target_latency) * NUDGE;
        nudged = "latency";
    }

    if (current_fps > target_fps * (1 + ANTI_NUDGE_THRESH)) {
      adjusted_fps -= (current_fps - target_fps*(1 + ANTI_NUDGE_THRESH)) * ANTI_NUDGE;
    } else if (current_latency < target_latency * (1 - ANTI_NUDGE_THRESH)) {
      adjusted_latency += (target_latency * (1 - ANTI_NUDGE_THRESH) - current_latency) * ANTI_NUDGE;
    }
    
    cout << "Configuration failed to reach " << nudged << " target\n";
  }

  cout << "Staleness limit reached\n";
}

int main(int argc, char** argv) {
  // Display help message
  cout << "Usage: " << argv[0] << " <Population_Size> <Target_Latency> <Target_FPS> <Staleness_Limit>\n";

  if (argc != 5) {
    std::cerr << "Error: Insufficient arguments. Please provide values for all parameters.\n";
    return EXIT_FAILURE;
  }

  // Parse population size
  int population_size = std::stoi(argv[1]);

  // Parse target latency
  int target_latency = std::stoi(argv[2]);

  // Parse target FPS
  int target_fps = std::stoi(argv[3]);

  // Parse staleness limit
  int staleness_limit = std::stoi(argv[4]);

  setup();

  govern(target_latency, target_fps, population_size, staleness_limit);
}
