import sys
from datetime import datetime
import argparse
import numpy as np

from pagerank import PageRank

ALGORITHMS = ['pagerank', 'hits', 'mcpagerank', 'mchits']

def main(args):
    parser = argparse.ArgumentParser(
        description='Calculate importance of webpages using PageRank, HITS, '
        'and their Monte Carlo variants.')

    parser.add_argument('-a', '--algorithm', required=True, choices=ALGORITHMS,
                        help='Select which algorithm to use')
    parser.add_argument('--random-data', action='store_true',
                        help='Use random data to test the algorithms')
    parser.add_argument('-df', '--damping-factor', type=float,
                        help='Damping factor for the PageRank algorithm')

    args = parser.parse_args(args)

    if args.algorithm == 'pagerank':
        if not args.damping_factor:
            raise ValueError('When PageRank is selected, damping factor '
                             'should be provided.')
        pr = PageRank(args.damping_factor)

        if args.random_data:
            adj_mat = np.random.binomial(1, 0.01, size=(1000, 1000))
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
