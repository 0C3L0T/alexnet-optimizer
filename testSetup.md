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

## Shrinking of the search space

Are there knobs that we can turn that result in good performance independent of other factors?

### quad core Cortex-A73 'big' cluster
- L1 cache: 96-128 KiB per core
- L2 cache: 1-8 MiB

### dual core Cortex-A53 'little' cluster
- L1 cache: 8-64 KiB
- L2 cache: 128 KiB - 2 MiB

### dual core Mali-G52 MP4 GPU
- L2 cache: 128 KB

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