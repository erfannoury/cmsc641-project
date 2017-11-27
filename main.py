import sys
from datetime import datetime
import argparse
import numpy as np

from pagerank import PageRank
from utils import random_graph

ALGORITHMS = ['pagerank', 'hits', 'mcpagerank', 'mchits']


def main(args):
    parser = argparse.ArgumentParser(
        description='Calculate importance of webpages using PageRank, HITS, '
        'and their Monte Carlo variants.')

    parser.add_argument('-a', '--algorithm', required=True, choices=ALGORITHMS,
                        help='Select which algorithm to use')
    parser.add_argument('--random-data', action='store_true',
                        help='Use random data to test the algorithms')
    parser.add_argument('--teleporting-prob', type=float, default=0.15,
                        help='Teleporting probability for the PageRank'
                        ' algorithm')
    parser.add_argument('--sparsity', type=float, default=0.01,
                        help='Sparsity of the random graph (probability of'
                        ' an edge being present between two nodes')
    parser.add_argument('-n', '--num-nodes', type=int, default=1000,
                        help='Graph size (number of nodes)')

    args = parser.parse_args(args)

    if args.algorithm == 'pagerank':
        if not args.teleporting_prob:
            raise ValueError('When PageRank is selected, teleporting '
                             'probability should be provided.')
        pr = PageRank(args.teleporting_prob)

        if args.random_data:
            adj_mat = random_graph(args.num_nodes, args.sparsity)
        else:
            raise NotImplementedError('Creating adjacency matrix from '
                                      'external sources not yet implemented.')

        print('Running the PageRank algorithm')
        now = datetime.now()
        page_rank = pr.calculate(adj_mat)
        print('Calculating the PageRank took {}.'.format(datetime.now() - now))
    else:
        raise NotImplementedError('Selected algorithm not yet implemented.')


if __name__ == '__main__':
    main(sys.argv[1:])
