import time
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt

import os

if __name__ == '__main__':
    # Part II: 向量相似度
    def cossim(u, v):
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        cosine = u @ v / norm
        ang = np.arccos(cosine)
        return 1 - ang / np.pi

    # 生成数据
    # get some random data points
    N = 1000
    D = 50
    A = np.random.randn(N, D)
    # artificially make some similar to others
    A[5] = A[99] + np.random.randn(D) * 0.05
    A[20] = A[85] + np.random.randn(D) * 0.15
    A[13] = A[19] + np.random.randn(D) * 0.25
    A[56] = A[71] + np.random.randn(D) * 0.5
    A[45] = A[49] + np.random.randn(D) * 0.66

    # 暴力计算基线
    true_pairs_dict = {}

    thresh = 0.8

    start = time.time()
    for (i, u), (j, v) in itertools.combinations([(i, x) for i, x in enumerate(A)], 2):
        val = cossim(u, v)
        if val > thresh:
            true_pairs_dict[(i, j)] = val
    t_brute = time.time() - start

    # save just the keys without the values. Easier to compare later to LSH
    true_pairs = set(true_pairs_dict.keys())

    print(f"Brute force calculation time: {t_brute:.3f}")
    print(f"Discovered pairs:")
    for k, v in true_pairs_dict.items():
        print(f"Pair: {k},\tSimilarity: {v:.2f}.")

    # 局部敏感哈希方法
    # 1. 寻找给定阈值的 r、b 参数
    b, r = 50, 18

    n = b * r
    print(f"Transition probability: {(1 / b) ** (1 / r):.2f}")

    # 2. 计算 LSH 对
    start = time.time()

    # Compute signature matrix
    R = A @ np.random.randn(D, n)
    S = np.where(R > 0, 1, 0)

    # Break into bands
    S = np.split(S, b, axis=1)

    # column vector to convert binary vector to integer e.g. (1,0,1)->5
    binary_column = 2 ** np.arange(r).reshape(-1, 1)

    # convert each band into a single integer,
    # i.e. convert band matrices to band columns
    S = np.hstack([M @ binary_column for M in S])

    # Every value in the matrix represents a hash bucket assignment
    # For every bucket in row i, add index i to that bucket
    d = collections.defaultdict(set)
    with np.nditer(S, flags=['multi_index']) as it:
        for x in it:
            d[int(x)].add(it.multi_index[0])

    # For every bucket, find all pairs. These are the LSH pairs.
    candidate_pairs = set()
    for k, v in d.items():
        if len(v) > 1:
            for pair in itertools.combinations(v, 2):
                candidate_pairs.add(tuple(sorted(pair)))

    # Finally, perform the actually similarity computation
    # to weed out false positive
    lsh_pairs = set()
    for (i, j) in candidate_pairs:
        if cossim(A[i], A[j]) > thresh:
            lsh_pairs.add((i, j))

    t_lsh = time.time() - start

    print(f"LSH calculation time: {t_lsh:.3f}")

    # 比较这两种方法
    print(f"t_brute: {t_brute:.3f}\t t_lsh: {t_lsh:.3f}. Speed-up: {t_brute / t_lsh:.0f}x")
    print("True pairs: ", true_pairs)
    print("LSH pairs: ", lsh_pairs)
    print(f"Candidate pairs: {len(candidate_pairs)}.\n\
    False negatives: {len(true_pairs - lsh_pairs)}")