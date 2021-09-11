import time
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt

import os

if __name__ == '__main__':
    # Part I: 文档相似度
    # MinHashing 不包含 Locality Sensitive Hashing
    # 1. Shingling
    HOME = os.getcwd()
    TARGET = os.path.join(HOME, 'sampledocs/')

    documents = []
    for article in os.listdir(TARGET):
        if article == 'stopwords':
            continue
        path = os.path.join(TARGET, article)
        with open(path, 'r') as file:
            documents.append(file.read())

    stopwords = []
    with open(os.path.join(TARGET, 'stopwords'), 'r') as file:
        for line in file:
            stopwords.append(line.strip())

    for i, doc in enumerate(documents):
        doc = doc.strip().replace('\n', ' ').lower()
        for word in stopwords:
            doc = doc.replace(' ' + word + ' ', ' ')
        documents[i] = doc

    print(f"Average char-length: \
    {np.mean(np.array([len(x) for x in documents]))}")
    print(f"Min char-length: {min(len(x) for x in documents)}")
    print(f"Max char-length: {max(len(x) for x in documents)}")

    # create K-shingles by sliding window approach
    def get_shingles(str1, k=5):
        d1 = set()
        for i in range(len(str1) - k):
            d1.add(str1[i:i + k])
        print(f"Found {len(d1)} unique shingles, out of {len(str1)} possible.")
        return d1

    doc_shingles = [get_shingles(s, 5) for s in documents]

    # 2. 定义 Jaccard 相似度（交集除以并集）
    def jaccard_sim(d1, d2):
        return len(d1.intersection(d2)) / len(d1.union(d2))

    # itertools.combinations finds all (,n) n-pairs
    # then we use a map op on the tuples with jaccard_sim
    pairs = itertools.combinations(documents, 2)
    pair_labels = []
    pair_sims = []
    for x1, x2 in itertools.combinations(zip(range(len(doc_shingles)), doc_shingles), 2):
        pair_labels.append((x1[0], x2[0]))
        pair_sims.append(jaccard_sim(x1[1], x2[1]))

    print(f"**~~~~~~ True similarity scores ~~~~~~**")
    print("Pair\tScore")
    print("-" * 14)
    for pair, score in zip(pair_labels, pair_sims):
        print(f"{pair}\t{score:.3f}")

    # Take union of all sets. Convert to an array and assign
    # each element an integer based on position in array
    fullset = set.union(*doc_shingles)
    shingle_dict = dict(zip(list(fullset), range(len(fullset))))
    print(f"There are {len(shingle_dict)} shingles")

    # 3. 定义 MinHash 类，能够创建一个 signature matrix
    # Create a hash function
    # define as a callable class, so that we only
    # intialize random functions once
    class HashManager:
        def __init__(self, shingle_dict):
            self.shingle_dict = shingle_dict
            self.N = len(shingle_dict)
            self.params = None

        def _initParams(self, n_sig):
            self.params = np.random.randint(self.N, size=[n_sig, 2])
            pass

        def _permuteRow(self, row):
            permute_row = (self.params @ np.array([1, row])) % self.N
            return permute_row

        def __call__(self, docs, n_sig, init=True):
            # Initialize if we change signature matrix length
            # or if we request to re-initialize
            if self.params is None or len(self.params) != n_sig or init:
                self._initParams(n_sig)

            # initialize signature matrix
            sig = np.full((n_sig, len(docs)), np.inf)

            # each doc in docs is assumed to be an iterable object
            for j, doc in enumerate(docs):
                for shingle in doc:
                    orig_row = shingle_dict[shingle]
                    curr_col = self._permuteRow(orig_row)
                    sig[:, j] = np.minimum(sig[:, j], curr_col)
            return sig.astype(int)


    # run some tests:
    try:
        print("Initialization test: ", end="")
        hm = HashManager(shingle_dict)
        print("passed")

        print("Set parameters to right size: ", end="")
        hm._initParams(n_sig=4)
        assert (hm.params.shape == (4, 2))
        print("passed")

        print("Permuting a row integer returns array: ", end="")
        curr_col = hm._permuteRow(3)
        assert (curr_col.shape == (4,))
        print("passed")

        print("Compute minhashed signature matrix: ", end="")
        hm(doc_shingles, 4)
        print("passed")
    except Exception as e:
        print("failure")
        print(e.args)

    hm = HashManager(shingle_dict)

    # 4. 使用MinHashing来计算相似度分数，看看它的表现如何
    def true_sim_scores(doc_shingles):
        pair_labels = []
        pair_sims = []
        idxs = range(len(doc_shingles))
        for x1, x2 in itertools.combinations(zip(idxs, doc_shingles), 2):
            pair_labels.append((x1[0], x2[0]))
            pair_sims.append(jaccard_sim(x1[1], x2[1]))
        return dict(zip(pair_labels, pair_sims))


    def sig_sim_scores(sig_mat):
        #     cols = [sig_mat[:,i] for i in range(sig_mat.shape[1])]
        cols = sig_mat.T
        idxs = range(sig_mat.shape[1])

        pair_labels = []
        pair_sims = []
        for (i, col1), (j, col2) in itertools.combinations(zip(idxs, cols), 2):
            pair_labels.append((i, j))
            pair_sims.append(np.mean(col1 == col2))

        return dict(zip(pair_labels, pair_sims))


    def print_score_comparison(true_dict, approx_dict):
        print(f"**~~~~~~ Similarity score comparison ~~~~~~**")
        print("Pair\t\tApprox\t\tTrue\t\t%Error")
        for pair, true_value in true_dict.items():
            approx_value = approx_dict[pair]
            err = 100 * abs(true_value - approx_value) / true_value
            print(f"{pair}\t\t{approx_value:.3f}\t\t{true_value:.3f}\t\t{err:.2f}")


    def candidate_pairs(score_dict, threshold):
        return set(pair for pair, scr in score_dict.items() if scr >= threshold)


    def acc_matrix(true_dict, approx_dict, threshold):
        true_pairs = candidate_pairs(true_dict, threshold)
        approx_pairs = candidate_pairs(approx_dict, threshold)
        false_negatives = len(true_pairs - approx_pairs)
        false_positives = len(approx_pairs - true_pairs)
        print(f"False negatives: {false_negatives}")
        print(f"Potential false positives: {false_positives}")


    sig_mat = hm(doc_shingles, 10)
    true_score_dict = true_sim_scores(doc_shingles)
    approx_score_dict = sig_sim_scores(sig_mat)
    print_score_comparison(true_score_dict, approx_score_dict)

    print("True pairs:", candidate_pairs(true_score_dict, 0.25))
    print("Candidate pairs:", candidate_pairs(approx_score_dict, 0.25))
    acc_matrix(true_score_dict, approx_score_dict, 0.4)

    # 暴力 band 候选对函数，用于以后检查哈希方法
    def banded_candidate_pair(col1, col2, b, r):
        """Returns a boolean if the two columns are a candidate pair
        inputs must obey n=len(col1)=len(col2)=b*r"""
        n = len(col1)
        assert (n == b * r)
        assert (n == len(col2))
        truth_array = (col1 == col2)
        return any(all(band) for band in np.array_split(truth_array, b))


    def banded_candidate_pairs(sig_mat, b, r):
        d = sig_mat.shape[1]
        idxs = range(d)
        cols = [sig_mat[:, i] for i in range(d)]
        pairs = set()
        for (i, col1), (j, col2) in itertools.combinations(zip(idxs, cols), 2):
            if banded_candidate_pair(col1, col2, b, r):
                pairs.add((i, j))
        return pairs


    # set p = 0.3 arbitrarily
    p = 0.3
    n = 120
    b = 30
    r = 4

    # see how many candidate pairs we got right!
    sig_mat = hm(doc_shingles, n)
    true_score_dict = true_sim_scores(doc_shingles)
    approx_score_dict = sig_sim_scores(sig_mat)
    print("True pairs:", candidate_pairs(true_score_dict, p))
    print("LSH pairs:", banded_candidate_pairs(sig_mat, b, r))
    print("Vanilla MinHash pairs:", candidate_pairs(approx_score_dict, p))

    print_score_comparison(true_score_dict, approx_score_dict)

    # 使用一个 band 和列 ID 的哈希表，进行快速的候选对搜索
    def fast_candidate_pairs(sig_mat, b, r):
        n, d = sig_mat.shape
        assert (n == b * r)
        hash_buckets = collections.defaultdict(set)
        bands = np.array_split(sig_mat, b, axis=0)
        for i, band in enumerate(bands):
            for j in range(d):
                # The last value must be made a string, to prevent accidental
                # key collisions of r+1 integers when we really only want
                # keys of r integers plus a band index
                band_id = tuple(list(band[:, j]) + [str(i)])
                hash_buckets[band_id].add(j)
        candidate_pairs = set()
        for bucket in hash_buckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidate_pairs.add(pair)
        return candidate_pairs


    # to make sure it works,
    # compare with the brute force method on a few trials

    # set p = 0.3 arbitrarily
    p = 0.3
    n = 120
    b = 30
    r = 4

    # see how many candidate pairs we got right!
    sig_mat = hm(doc_shingles, n)
    true_score_dict = true_sim_scores(doc_shingles)
    approx_score_dict = sig_sim_scores(sig_mat)
    print('True pairs:\t', candidate_pairs(true_score_dict, p))
    print("True LSH pairs:\t", banded_candidate_pairs(sig_mat, b, r))
    print("Fast LSH pairs:\t", fast_candidate_pairs(sig_mat, b, r))
    print("MinHash pairs:\t", candidate_pairs(approx_score_dict, p))
