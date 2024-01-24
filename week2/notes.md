Week 2: governor design
---
We need to figure out a design for the governor.
It helps that the governor will be specific for AlexNet,
so we can make some assumptions based on our tests.

## Governor requirements
- must be able to change the hardware configuration
- must be able to measure the performance of the hardware configuration
- fit the FPS and latency to predefined values using multiple iterations

## Option 1: Proportional–integral–derivative
PID uses an iterative feedback loop to control a system.
In our case, implementing PID means we have to find a way to optimize multiple parameters at once:
- frequency of little cluster
- frequency of big cluster
- order of Little, Big, GPU
- partition points (graph specific)

based on the feedback we get from the system:
- FPS
- latency
- power usage

### Pros
- pid is a well known algorithm

### Cons
- PID works well for systems with a single input and output, but we have multiple inputs and outputs.
- we only have three iterations to converge to the optimal configuration, which might not be enough.
---

## Option 2: Neural network
We can use a neural network to learn the optimal configuration.
We can make a bunch of measurements and use those to train the network.

## Input
- FPS
- latency

## Output
- frequency of little cluster
- frequency of big cluster
- order of Little, Big, GPU
- partition point 1
- partition point 2

## reward function
- power usage?

### Pros
- we already have a way to automate the testing

### Cons


---

## Option 5: Simulated Annealing

### Pros

### Cons

---

## Planning
1. figure out which option we're going with
2. figure out how to implement it
3. write the governor (in kernel space)
4. write the report
