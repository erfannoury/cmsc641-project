from datetime import datetime
from functools import partial
import numpy as np
import utils
from pagerank import PageRank, MCPageRank
from hits import HITS, MCHITS
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='whitegrid')

if __name__ == '__main__':
    TELEPORTING_PROB = 0.15
    MAX_ITERATION = 5
    NODE_COUNT = 1024
    SPARSITY = 0.01

    mcpr = MCPageRank(TELEPORTING_PROB)
    pr = PageRank(TELEPORTING_PROB)
    hits = HITS()
    mchits = MCHITS(1 - TELEPORTING_PROB)

    web = utils.random_graph(NODE_COUNT, SPARSITY)

    pr_vals = pr.calculate(web)
    pr_indices = np.argsort(pr_vals)
    hits_vals = hits.calculate(web)
    hits_indices = np.argsort(hits_vals)

    alg_results = {}
    alg_results['PageRank'] = []
    alg_results['HITS'] = []

    for i in range(1, MAX_ITERATION + 1):
        print(i)
        mcpr_vals = mcpr.calculate(web, i)
        alg_results['PageRank'].append((
            i,
            np.mean(np.sqrt((pr_vals[pr_indices[-10:]] - mcpr_vals[pr_indices[-10:]]) ** 2))
        ))

        mchits_vals = mchits.calculate(web, i)
        alg_results['HITS'].append((
            i,
            np.mean(np.sqrt((hits_vals[hits_indices[-10:]] - mchits_vals[hits_indices[-10:]]) ** 2))
        ))

    fig = plt.figure(figsize=(18, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(
        list(map(lambda x: x[0], alg_results['PageRank'])),
        list(map(lambda x: x[1], alg_results['PageRank'])),
        alpha=0.75,
    )
    ax.scatter(
        list(map(lambda x: x[0], alg_results['PageRank'])),
        list(map(lambda x: x[1], alg_results['PageRank'])),
        alpha=0.8,
        label='PageRank'
    )

    ax.plot(
        list(map(lambda x: x[0], alg_results['HITS'])),
        list(map(lambda x: x[1], alg_results['HITS'])),
        alpha=0.75,
    )
    ax.scatter(
        list(map(lambda x: x[0], alg_results['HITS'])),
        list(map(lambda x: x[1], alg_results['HITS'])),
        alpha=0.8,
        label='HITS',
    )
    ax.set_title('Top-10 error comparison')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Number of iterations')
    ax.set_xticks(list(range(1, MAX_ITERATION + 1)))
    ax.set_xlim([0.5, MAX_ITERATION + 0.5])
    fig.legend(loc='lower center', ncol=2)
    plt.savefig('figures/error_numiter.png', bbox_inches='tight')
