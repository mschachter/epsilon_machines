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

def get_gee_tree(dthresh=1e-3):

    wdist = {
        '00100':1.0/8.0,
        '00110':1.0/16.0,
        '00111':1.0/16.0,
        '01001':1.0/8.0,
        '01100':1.0/16.0,
        '01110':1.0/32.0,
        '01111':1.0/32.0,
        '10010':1.0/8.0,
        '10011':1.0/8.0,
        '11001':1.0/8.0,
        '11100':1.0/16.0,
        '11110':1.0/32.0,
        '11111':1.0/32.0
    }

    bpt = BinaryParseTree(5, 2, 2, dthresh=dthresh)
    bpt.parse_from_word_distribution(wdist)
    return bpt


def get_oh_tree(dthresh=1e-3):

    wdist = {
        '00000':1.0/40.0,
        '00001':1.0/40.0,
        '00010':1.0/40.0,
        '00011':1.0/40.0,
        '00100':1.0/40.0,
        '00101':1.0/40.0,
        '00111':1.0/20.0,
        '01000':1.0/40.0,
        '01001':1.0/40.0,
        '01010':1.0/40.0,
        '01011':1.0/40.0,
        '01110':1.0/20.0,
        '01111':1.0/20.0,
        '10000':1.0/40.0,
        '10001':1.0/40.0,
        '10010':1.0/40.0,
        '10011':1.0/40.0,
        '10100':1.0/40.0,
        '10101':1.0/40.0,
        '10111':1.0/20.0,
        '11000':1.0/40.0,
        '11001':1.0/40.0,
        '11010':1.0/40.0,
        '11011':1.0/40.0,
        '11100':1.0/20.0,
        '11101':1.0/20.0,
        '11110':1.0/20.0,
        '11111':3.0/20.0
    }

    bpt = BinaryParseTree(5, 2, 2, dthresh=dthresh)
    bpt.parse_from_word_distribution(wdist)
    return bpt


