from process import *
from information import *


def test_golden_mean():

    gm = GoldenMean()

    gstr = gm.simulate(100000)

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
