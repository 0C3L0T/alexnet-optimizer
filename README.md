# Alexnet optimizer
This repository contains the code used to predict the optimal power governor and pipline configuration for inference of the Alexnet model on an edge device.
Under the hood, this optimization engine uses a genetic algorithm which utilizes two distinct neural networks as a fitness score in rating possible configurations


## To run

First of all, apologies for the messy directory structure.

Since our governor is written in Python, the system needs to be connected to the board through adb. Make sure the adb connection is root (command: ``adb root``).

To run the python version, if you don't have the parse_perf executable on the board, run ``./build_push_parser.sh`` in the ``governor_model`` directory.

To run the governor:
```bash
    python3 governor.py <latency_target> <FPS_target>
```

To compile and push the C version (with broken performance predictor) to the board:
```
../build *.cpp && ../../push governor
```

Running it without arguments prints usage.
