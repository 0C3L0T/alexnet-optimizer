# Dingen die we kunnen testen

## Knobs
1. frequency of little cluster
2. frequency of big cluster
3. order of Little, Big, GPU
4. partition points (graph specific)



## Power measurement
- we hebben een python script dat constant de power usage uitleest
- we hebben een python script dat continu tests uitvoert

we moeten een manier bedenken om de measurements af te stemmen op de tests.

**Run de tests via python**

./governor ./graph_alexnet_all_pipe_sync --threads=4 --threads2=2 --n=60 --total_cores=6 --partition_point=3 --partition_point2=5 --order=G-L-B

run CNN on Big, Little at all power frequencies, also gpu
all possible hardware orders x all layers
GPU has to work with BIG CPU

## Shrinking of the search space

Are there knobs that we can turn that result in good performance independent of other factors?



We can try and put the hardware with the most cache space earlier in the order, since the earlier stages contain more data.

In this case, we can see that the big cluster has a bigger cache, we can try and put it first in the order.


## Neural Network
What measurements do we need to train a NN?

### in:
- freq. little
- freq. big
- order
- partition 1
- partition 2

### out:
- latency
- FPS
- Power