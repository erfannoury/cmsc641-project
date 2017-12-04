# CMSC 641 Project
UMBC CMSC 641 Project - PageRank, HITS, and their Monte Carlo variants

Contributors: Rose Kunnappallil, Erfan Noury

## Usage
```
usage: main.py [-h] -a {pagerank,hits,mcpagerank,mchits} [--random-data]
               [--teleporting-prob TELEPORTING_PROB] [--sparsity SPARSITY]
               [-n NUM_NODES] [--num-iter NUM_ITER] [--iterative] [--verbose]

Calculate importance of webpages using PageRank, HITS, and their Monte Carlo
variants.

optional arguments:
  -h, --help            show this help message and exit
  -a {pagerank,hits,mcpagerank,mchits}, --algorithm {pagerank,hits,mcpagerank,mchits}
                        Select which algorithm to use
  --random-data         Use random data to test the algorithms
  --teleporting-prob TELEPORTING_PROB
                        Teleporting probability for the PageRank algorithm
  --sparsity SPARSITY   Sparsity of the random graph (probability of an edge
                        being present between two nodes
  -n NUM_NODES, --num-nodes NUM_NODES
                        Graph size (number of nodes)
  --num-iter NUM_ITER   Number of iterations to run the algorithm
  --iterative           Whether to use the Power Iteration
  --verbose             Increase verbosity to show running time
```

### Examples
To run PageRank algorithm on a random graph with 2000 nodes and report the running time, use
```
$ python main.py -a pagerank --random-data -n 2000 --verbose
```

To run the Iterative HITS algorithm on a random graph with 15000 nodes, with sparsity of 0.1, use
```
$ python main.py -a hits --random-data -n 15000 --sparsity 0.1 --num-iter 20 --iterative
```