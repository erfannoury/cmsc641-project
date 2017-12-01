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
    NUM_ITER = 4
    EXP_COUNT = 1
    SPARSITY = 0.01
    MAX_SIZE = 10
    MIN_SIZE = 6

    mcpr = MCPageRank(TELEPORTING_PROB)
    pr = PageRank(TELEPORTING_PROB)
    hits = HITS()
    mchits = MCHITS(1 - TELEPORTING_PROB)

    alg_dict = {
        'PageRank': partial(pr.calculate, verbose=True),
        'PI PageRank': partial(pr.iterative_calculate, verbose=True,
                               iter_count=NUM_ITER * 10),
        'MCPageRank': partial(mcpr.calculate, verbose=True, num_iter=NUM_ITER),
        'HITS': partial(hits.calculate, verbose=True),
        'PI HITS': partial(hits.iterative_calculate, num_iter=NUM_ITER * 10,
                           verbose=True),
        'MCHITS': partial(mchits.calculate, verbose=True, num_iter=NUM_ITER)}

    alg_results = {}
    for a, f in alg_dict.items():
        print('Benchmarking', a)
        if a not in alg_results:
            alg_results[a] = []
        for _ in range(EXP_COUNT):
            for i in range(MIN_SIZE, MAX_SIZE + 1):
                print('\tfor size', 2 ** i)
                web = utils.random_graph(2 ** i, SPARSITY)
                _, delta = f(web)
                alg_results[a].append((i, delta.total_seconds()))

    fig = plt.figure(figsize=(18, 10), dpi=150)
    ax = fig.add_subplot(111)
    for a, r in alg_results.items():
        ax.plot(
            list(map(lambda x: x[0], r)),
            list(map(lambda x: np.log10(1 + x[1]), r)),
            alpha=0.75
        )
        ax.scatter(
            list(map(lambda x: x[0], r)),
            list(map(lambda x: np.log10(1 + x[1]), r)),
            label=a,
            alpha=0.8,
            edgecolors='black'
        )
    ax.set_title('Run-time comparison')
    ax.set_ylabel('$\\log_{10}(1 + t)$ s - Time')
    ax.set_xlabel('$\\log_2(n)$ - Number of nodes')
    ax.set_xticks(list(range(MIN_SIZE, MAX_SIZE + 1)))
    ax.set_xlim([MIN_SIZE - 0.5, 0.5 + MAX_SIZE])
    fig.legend(loc='lower center', ncol=6)
    plt.savefig('figures/run-time.png', bbox_inches='tight')
