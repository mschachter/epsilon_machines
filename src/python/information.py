import copy
import itertools
import numpy as np

import matplotlib.pyplot as plt


def word_str(w):
    return ''.join(['%d' % x for x in w])


def empirical_word_distributions(process_string, L, expected_words):
    """ Compute empirical word distributions for a given string """

    wcnt = dict()
    for w in expected_words:
        wstr = word_str(w)
        wcnt[wstr] = 0
    nwords = 0
    for k in range(len(process_string)):
        if k >= L-1:
            w = process_string[k-L+1:k+1]
            wstr = word_str(w)
            wcnt[wstr] += 1
            nwords += 1
    for w,wc in wcnt.iteritems():
        wcnt[w] /= float(nwords)
    return wcnt


def analytical_word_distributions(process, L):
    """ Compute analytical word distributions for a given string """

    adist = dict()
    expected_words = get_all_binary_words(L)
    for w in expected_words:
        wstr = word_str(w)
        wp = word_probability(w, process.T0, process.T1, process.stationary_distribution)
        adist[wstr] = wp
    return adist


def remove_duplicates(wlist):
    """ removes duplicates from list of permutations... """

    wdict = dict()
    for w in wlist:
        wstr = word_str(w)
        wdict[wstr] = w
    return wdict.values()


def word_probability(word, T0, T1, p):
    """ Computes the analytic probability of a given binary word. """
    Tdict = {0:T0, 1:T1}
    T = np.eye(2)
    for w in word:
        Tw = Tdict[w]
        T = np.dot(T, Tw)
    return np.dot(T.transpose(), p).sum()


def get_all_binary_words(L):
    """ Generate all possible binary words of length L """

    wlist = list()
    wbase = [0]*L
    wlist.append(copy.copy(wbase))
    for k in range(L):
        wbase[k] = 1
        wp = remove_duplicates(itertools.permutations(wbase))
        wlist.extend(wp)
    return remove_duplicates(wlist)


def word_entropy(process, Lvals):
    """ Compute the word entropy for each length L in Lvals """

    Hvals = list()
    for L in Lvals:
        awdist = analytical_word_distributions(process, L)
        p = np.array(awdist.values())
        H = -np.sum(np.log2(p[p > 0.0])*p[p > 0.0])
        Hvals.append(H)
    return np.array(Hvals)


def plot_process_info(process, Lvals):

    Hvals = word_entropy(process, Lvals)

    Hrate = list(np.diff(Hvals))
    Hrate.insert(0, np.log2(2))

    Hpred = np.diff(Hrate)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(Lvals, Hvals, 'ko-')
    plt.xlabel('L')
    plt.ylabel('Word Entropy $H(L)$ (bits)')
    plt.axis('tight')

    plt.subplot(2, 2, 2)
    plt.plot(Lvals, Hrate, 'bx-')
    plt.xlabel('L')
    plt.ylabel('Entropy Gain $h_\mu (L)$ (bits/symbol) ')
    plt.axis('tight')

    plt.subplot(2, 2, 3)
    plt.plot(Lvals[1:], Hpred, 'ro-')
    plt.xlabel('L')
    plt.ylabel('Predictability Gain $\Delta^2 H(L)$')
    plt.axis('tight')
