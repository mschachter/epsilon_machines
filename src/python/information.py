import copy
import itertools
import numpy as np


def word_str(w):
    return ''.join(['%d' % x for x in w])


def emp_word_dists(process_string, L, expected_words):
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


def remove_duplicates(wlist):
    """ removes duplicates from list of permutations... """

    wdict = dict()
    for w in wlist:
        wstr = word_str(w)
        wdict[wstr] = w
    return wdict.values()


def word_probability(word, T0, T1, p, nstates=2):
    """ Computes the analytic probability of a given binary word. """
    Tdict = {0:T0, 1:T1}
    T = np.eye(nstates)
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
