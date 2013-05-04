from epsilon_machine import BinaryParseTree
from process import *
from information import *


def test_golden_mean():

    gm = GoldenMean()

    gstr = gm.simulate(500000)

    for L in range(2, 5):
        expected_words = get_all_binary_words(L)
        ewdist = empirical_word_distributions(gstr, L, expected_words)
        awdist = analytical_word_distributions(gm, L)

        dprobs = dict()
        for ws,wp in ewdist.iteritems():
            dprobs[ws] = list()
            dprobs[ws].append(wp)
        for ws,wp in awdist.iteritems():
            dprobs[ws].append(wp)

        print 'Distribution of length %d (empirical, analytical)' % L
        for ws,(ewp,awp) in dprobs.iteritems():
            print '%s: %0.4f, %0.4f' % (ws, ewp, awp)

        Lvals = range(1, 8)
        plot_process_info(gm, Lvals)

    bpt = BinaryParseTree(5, 2, 2)
    bpt.parse(gstr)
    bpt.show()

def test_biasedcoin(p=0.5):

    bc = BiasedCoin(p=p)

    bstr = bc.simulate(100000)

    bpt = BinaryParseTree(4, 2, 4)
    bpt.parse(bstr)
    bpt.show()

def get_gm_tree():

    gm = GoldenMean()
    gstr = gm.simulate(500000)
    bpt = BinaryParseTree(5, 2, 2)
    bpt.parse(gstr)
    return bpt

def get_evenprocess_tree(dthresh=1e-1, D=5, M=2, L=3):

    ep = EvenProcess()
    estr = ep.simulate(1000000)
    bpt = BinaryParseTree(D, M, L, dthresh=dthresh)
    bpt.parse(estr)
    return bpt


def get_rrxor_tree():

    rr = RRXOR()
    rstr = rr.simulate(1000000*3)
    bpt = BinaryParseTree(3, 2, 1)
    bpt.parse(rstr)
    return bpt
