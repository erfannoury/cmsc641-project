import sys
from datetime import datetime
import argparse
import numpy as np

from pagerank import PageRank, MCPageRank
from hits import HITS, MCHITS
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
    parser.add_argument('--num-iter', type=int, default=10,
                        help='Number of iterations to run the algorithm')
    parser.add_argument('--iterative', action='store_true',
                        help='Whether to use the Power Iteration')
    parser.add_argument('--verbose', action='store_true',
                        help='Increase verbosity to show running time')

    args = parser.parse_args(args)

    if args.algorithm == 'pagerank':
        pr = PageRank(args.teleporting_prob)

        if args.random_data:
            adj_mat = random_graph(args.num_nodes, args.sparsity)
        else:
            raise NotImplementedError('Creating adjacency matrix from '
                                      'external sources not yet implemented.')

        print('Running the PageRank algorithm')
        if args.verbose:
            if args.iterative:
                page_rank, delta = pr.iterative_calculate(
                    adj_mat, args.num_iter, True)
            else:
                page_rank, delta = pr.calculate(adj_mat, True)
            print('Calculating the PageRank took {}.'.format(delta))
        else:
            if args.iterative:
                page_rank = pr.iterative_calculate(adj_mat, args.num_iter)
            else:
                page_rank = pr.calculate(adj_mat)

    elif args.algorithm == 'mcpagerank':
        mcpr = MCPageRank(args.teleporting_prob)

        if args.random_data:
            adj_mat = random_graph(args.num_nodes, args.sparsity)
        else:
            raise NotImplementedError('Creating adjacency matrix from '
                                      'external sources not yet implemented.')

        print('Running the MCPageRank algorithm')
        if args.verbose:
            page_rank, delta = mcpr.calculate(adj_mat, args.num_iter, True)
            print('Calculating the PageRank took {}.'.format(delta))
        else:
            page_rank = mcpr.calculate(adj_mat, args.num_iter)

    elif args.algorithm == 'hits':
        hits = HITS()

        if args.random_data:
            adj_mat = random_graph(args.num_nodes, args.sparsity)
        else:
            raise NotImplementedError('Creating adjacency matrix from '
                                      'external sources not yet implemented.')

        print('Running the HITS algorithm')
        if args.verbose:
            if args.iterative:
                au_scores, delta = hits.iterative_calculate(
                    adj_mat, args.num_iter, True)
            else:
                au_scores, delta = hits.calculate(adj_mat, True)
            print('Calculating the Authority scores took {}.'.format(delta))
        else:
            if args.iterative:
                au_scores = hits.iterative_calculate(adj_mat, args.num_iter)
            else:
                au_scores = hits.calculate(adj_mat)

    elif args.algorithm == 'mchits':
        mchits = MCHITS(1 - args.teleporting_prob)

        if args.random_data:
            adj_mat = random_graph(args.num_nodes, args.sparsity)
        else:
            raise NotImplementedError('Creating adjacency matrix from '
                                      'external sources not yet implemented.')

        print('Running the MCHITS algorithm')
        if args.verbose:
            au_scores, delta = mchits.calculate(adj_mat, args.num_iter, True)
            print('Calculating the Authority scores took {}.'.format(delta))
        else:
            au_scores = mchits.calculate(adj_mat, args.num_iter)

    else:
        raise NotImplementedError('Selected algorithm not yet implemented.')


if __name__ == '__main__':
    main(sys.argv[1:])
