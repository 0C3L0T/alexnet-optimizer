//
// Created by ocelot on 1/29/24.
//
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "GA.h"
#include "fitness.h"

using namespace std;

int LittleFrequencyTable[] = { 500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000 };
int BigFrequencyTable[]    = { 500000,  667000,  1000000, 1200000, 1398000, 1512000, 1608000,
                               1704000, 1800000, 1908000, 2016000, 2100000, 2208000 };

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
  string Command = "";
  Command        = "echo " + to_string(freq) + " > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq";
  system(Command.c_str());
}

void scale_big(int freq) {
  string Command = "";
  Command        = "echo " + to_string(freq) + " > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq";
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

int ParseResults() {
  float    FPS;
  float    Latency;
  ifstream myfile("output.txt");
  cout << endl;
  /* Read Output.txt File and Extract Data */
  for (std::string line; getline(myfile, line);) {
    string temp;
    /* Extract Frame Rate */
    if (line.find("Frame rate is:") == 0) {
      // cout<<"line is: "<<line<<std::endl;
      std::istringstream ss(line);
      while (!ss.eof()) {
        /* extracting word by word from stream */
        ss >> temp;
        /* Checking the given word is float or not */
        if (stringstream(temp) >> FPS) {
          printf("Throughput is: %f FPS\n", FPS);
          break;
        }
        temp = "";
      }
    }
    /* Extract Frame Latency */
    if (line.find("Frame latency is:") == 0) {
      // cout<<"line is: "<<line<<std::endl;
      std::istringstream ss(line);
      while (!ss.eof()) {
        /* extracting word by word from stream */
        ss >> temp;
        /* Checking the given word is float or not */
        if (stringstream(temp) >> Latency) {
          printf("Latency is: %f ms\n", Latency);
          break;
        }
        temp = "";
      }
    }
    /* Extract Stage One Inference Time */
    if (line.find("stage1_inference_time:") == 0) {
      // cout<<"line is: "<<line<<std::endl;
      std::istringstream ss(line);
      while (!ss.eof()) {
        /* extracting word by word from stream */
        ss >> temp;
        /* Checking the given word is float or not */
        if (stringstream(temp) >> StageOneInferenceTime) {
          // printf("StageOneInferenceTime is: %f ms\n", StageOneInferenceTime);
          break;
        }
        temp = "";
      }
    }
    /* Extract Stage Two Inference Time */
    if (line.find("stage2_inference_time:") == 0) {
      // cout<<"line is: "<<line<<std::endl;
      std::istringstream ss(line);
      while (!ss.eof()) {
        /* extracting word by word from stream */
        ss >> temp;
        /* Checking the given word is float or not */
        if (stringstream(temp) >> StageTwoInferenceTime) {
          // printf("StageTwoInferenceTime is: %f ms\n", StageTwoInferenceTime);
          break;
        }
        temp = "";
      }
    }
    /* Extract Stage Three Inference Time */
    if (line.find("stage3_inference_time:") == 0) {
      // cout<<"line is: "<<line<<std::endl;
      std::istringstream ss(line);
      while (!ss.eof()) {
        /* extracting word by word from stream */
        ss >> temp;
        /* Checking the given word is float or not */
        if (stringstream(temp) >> StageThreeInferenceTime) {
          // printf("StageThreeInferenceTime is: %f ms\n", StageThreeInferenceTime);
          break;
        }
        temp = "";
      }
    }
  }
  /* Check Throughput and Latency Constraints */
  if (Latency <= Target_Latency) {
    LatencyCondition = 1;  // Latency requirement was met.
  }
  if (FPS >= Target_FPS) {
    FPSCondition = 1;  // FPS requirement was met.
  }
}

// optionally implement time
void govern(float target_latency, float target_fps) {
  float adjusted_fps     = target_fps;
  float adjusted_latency = target_latency;

  bool win             = false;
  int  stale_count     = 0;
  int  population_size = 100;
  int  staleness_limit = 40;

  int pp1;

  while (!win) {
    // pass requirements to GA
    chromosome solution =
        genetic_algorithm(population_size, target_latency, target_fps, staleness_limit, fitness_function);

    // if solution is same as last solution, update staleness counter

    // if solution is different, reset staleness counter

    // get info from solution
    pp1             = solution.genes[0]->layers;
    int pp2         = pp1 + solution.genes[1]->layers;
    int big_freq    = BigFrequencyTable[solution.genes[0]->frequency_level];
    int little_freq = LittleFrequencyTable[solution.genes[2]->frequency_level];

    printf("trying configuration: %d %d %d %d\n", pp1, pp2, big_freq, little_freq);
    run_graph(pp1, pp2, big_freq, little_freq);
  }
}

int main(int argc, char* argv[]) {
  // Display help message
  std::cout << "Usage: " << argv[0] << " <Population_Size> <Target_Latency> <Target_FPS> <Staleness_Limit>\n";

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

  float (*fitness_function)(chromosome*);

  setup();

  govern(target_latency, target_fps);

  // print configuration
  char buffer[256];
  chromosomeToString(&solution, buffer, 256);
  printf("%s", buffer);

  // optionally: parse feedback (look at exampleGovernor)
}
