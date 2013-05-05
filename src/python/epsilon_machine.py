import copy
import string

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


class BinaryParseTree(object):

    def __init__(self, tree_depth, past_length, future_length, dthresh=1e-3):
        self.D = tree_depth
        self.L = future_length
        self.K = past_length
        self.dthresh = dthresh

        self.construct_tree()

    def construct_tree(self):
        self.g = nx.DiGraph()
        self.g.add_node(0, depth=0)

        self.add_child_nodes(self.g, 0, 0, self.D)

    def add_child_nodes(self, g, node, current_depth, max_depth):

        if current_depth >= max_depth:
            return

        zero_node = len(g.nodes())
        g.add_node(zero_node, count=0, depth=current_depth+1)
        g.add_edge(node, zero_node, label=0)
        self.add_child_nodes(g, zero_node, current_depth+1, max_depth)

        one_node = len(g.nodes())
        g.add_node(one_node, count=0, depth=current_depth+1)
        g.add_edge(node, one_node, label=1)
        self.add_child_nodes(g, one_node, current_depth+1, max_depth)

    def parse(self, process_string):

        if process_string is not None:
            #build up word counts
            M = len(process_string)
            nwords = M - self.D + 1
            for t in range(self.D, M+1):
                word = process_string[t-self.D:t]
                self.update_count(self.g, word, 0)
            self.g.node[0]['count'] = nwords

            #normalize word counts to probabilities
            for n in self.g.nodes():
                self.g.node[n]['probability'] = float(self.g.node[n]['count']) / nwords

        #compute the transition probabilites (edges)
        self.compute_transition_probability(self.g, 0)

        #prune graph
        self.g.remove_nodes_from([n for n in self.g.nodes() if n != 0 and self.g.node[n]['probability'] == 0])

        #find morphs
        self.morphs = self.find_morphs()

        #build causal state graph
        self.causal_state_graph = self.build_causal_state_graph()

    def parse_from_word_distribution(self, word_dist):

        psum = np.sum(word_dist.values())
        if psum - 1.0 > 1e-6:
            print 'Word distribution does not sum to 1! sum=%f' % psum

        #initialize word probabilities to zero
        for n in self.g.nodes():
            self.g.node[n]['probability'] = 0.0

        #set leaf probabilities
        for word,pword in word_dist.iteritems():
            wlist = [int(w) for w in word]
            node = self.iterate_to_word_node(self.g, 0, wlist)
            self.g.node[node]['probability'] = pword

        #compute node probabilities from bottom up
        for d in range(self.D-1, -1, -1):
            for node in self.get_nodes_at_depth(self.g, d):
                pchildren = [self.g.node[n]['probability'] for n in self.g.successors(node)]
                self.g.node[node]['probability'] = np.sum(pchildren)

        #parse tree as usual
        self.parse(None)

    def get_nodes_at_depth(self, g, depth):
        return [n for n in g.nodes() if g.node[n]['depth'] == depth]

    def iterate_to_word_node(self, g, node, word):

        children = g.successors(node)
        #print 'node=%d, word=%s, len(children)=%d' % (node, ''.join(['%d' % w for w in word]), len(children))
        if len(children) == 0:
            return None

        w = word[0]
        next_nodes = [n for n in children if g[node][n]['label'] == w]
        if len(next_nodes) != 1:
            print 'Something is wrong! Zero or multiple child nodes for letter %d from node %d' % (w, node)
            return None

        next_node = next_nodes[0]
        #print 'next_node=%d' % next_node
        if len(word) == 1:
            return next_node
        else:
            return self.iterate_to_word_node(g, next_node, word[1:])


    def compute_transition_probability(self, g, node):
        for child in g.successors(node):
            if 'probability' in g.node[node] and 'probability' in g.node[child]:
                pparent = g.node[node]['probability']
                pchild = g.node[child]['probability']
                ptransition = 0.0
                if pparent > 0.0:
                    ptransition = pchild / pparent
                g[node][child]['probability'] = ptransition

            else:
                d = float(g.node[node]['count'])
                if d > 0.0:
                    g[node][child]['probability'] = float(g.node[child]['count']) / d
                else:
                    g[node][child]['probability'] = 0.0
            self.compute_transition_probability(g, child)

    def update_count(self, g, word, node):
        #print '[update_count] node=%d, word=' % node,word

        for child in g.successors(node):
            if g[node][child]['label'] == word[0]:
                g.node[child]['count'] += 1
                if len(word) > 1:
                    self.update_count(g, word[1:], child)

    def find_morphs(self):
        morphs = dict()
        subtrees = self.get_all_subtrees(self.g, self.L, 0)

        for sg in subtrees:
            causal_state = self.find_match(morphs, sg)
            if causal_state is None:
                causal_state = string.uppercase[len(morphs)]
                morphs[causal_state] = copy.copy(sg)

            #label the top of every subtree with it's causal state
            tt = [n for n in sg.nodes() if len(sg.predecessors(n)) == 0]
            if len(tt) == 1:
                self.g.node[tt[0]]['causal_state'] = causal_state

        return morphs

    def build_causal_state_graph(self):

        sg = nx.DiGraph()
        for causal_state,morph in self.morphs.iteritems():
            sg.add_node(causal_state, transition_count=0)

        for n1,n2 in self.g.edges():
            if 'causal_state' in self.g.node[n1] and 'causal_state' in self.g.node[n2]:
                s1 = self.g.node[n1]['causal_state']
                s2 = self.g.node[n2]['causal_state']
                sg.node[s1]['transition_count'] += 1

                p = self.g[n1][n2]['probability']
                l = self.g[n1][n2]['label']
                if s2 not in sg[s1]:
                    sg.add_edge(s1, s2, probabilities=list(), labels=list())
                sg[s1][s2]['probabilities'].append(p)
                sg[s1][s2]['labels'].append(l)

        for s1,s2 in sg.edges():
            #sg[s1][s2]['probability'] = np.mean(sg[s1][s2]['probabilities'])
            cnt = len(sg[s1][s2]['probabilities'])
            lbls = np.unique(sg[s1][s2]['probabilities'])
            if len(lbls) > 1:
                print 'Multiple labels found between causal states %s and %s' % (s1, s2)
            sg[s1][s2]['probability'] = cnt / float(sg.node[s1]['transition_count'])
            sg[s1][s2]['label'] = lbls[0]

        #find transition matrix
        N = len(sg.nodes())
        T = np.zeros([N, N])
        T0 = np.zeros([N, N])
        T1 = np.zeros([N, N])
        edges = sg.edges()
        sorted_states = sorted(sg.nodes())
        for k,s1 in enumerate(sorted_states):
            for j,s2 in enumerate(sorted_states):
                if (s1,s2) in edges:
                    T[k, j] = sg[s1][s2]['probability']
                    l = sg[s1][s2]['label']
                    if int(l) == 0:
                        T0[k, j] = sg[s1][s2]['probability']
                    else:
                        T1[k, j] = sg[s1][s2]['probability']


        self.causal_transition_states = sorted_states
        self.causal_transition_probability = T
        self.emission_transition_probability = [T0, T1]

        return sg

    def find_match(self, unique_subtrees, subtree):
        """ Check the dictionary of unique subtrees to see if there's a match with the
            given subtree. If so, return the key in the dictionary corresponding to that.
        """
        for causal_state,ug in unique_subtrees.iteritems():
            d = self.subtree_distance(ug, subtree)
            if d < self.dthresh:
                return causal_state

    def subtree_distance(self, g1, g2):
        """ Compute a distance measure between two given subtrees on their transition probabilities. """

        w1 = self.get_all_word_probabilities(g1)
        w2 = self.get_all_word_probabilities(g2)
        if len(w1) != len(w2):
            return np.inf

        #compute symmetric KL distance between word distributions
        common_words = np.intersect1d(w1.keys(), w2.keys())
        d = 0.0
        for w in common_words:
            pw1 = w1[w]
            pw2 = w2[w]

            for p1,p2 in zip(pw1, pw2):
                d += p1*(np.log2(p1) - np.log2(p2))
                d += p2*(np.log2(p2) - np.log2(p1))
        d /= 2.0
        return d

    def get_all_word_probabilities(self, g):
        words = dict()

        lowest_nodes = [n for n in g.nodes() if len(g.successors(n)) == 0]
        for n in lowest_nodes:
            w,pvals = self.get_word_probabilities(g, n)
            wstr = ''.join(['%d' % x for x in w])
            words[wstr] = pvals
        return words

    def get_word_probabilities(self, g, n):

        word = list()
        pvals = list()
        pnodes = g.predecessors(n)
        if len(pnodes) == 1:
            parent_node = pnodes[0]
            p = g[parent_node][n]['probability']
            lbl = g[parent_node][n]['label']

            word.append(lbl)
            pvals.append(p)

            pword,ppvals = self.get_word_probabilities(g, parent_node)
            word.extend(pword)
            pvals.extend(ppvals)

        return word,pvals

    def get_all_subtrees(self, g, L, node):
        """ Get all subtrees of a given length starting from node"""

        subtrees = list()

        subtree_nodes = [node]
        current_children = g.successors(node)
        subtree_depth = 0
        for k in range(L):
            if len(current_children) == 0:
                break

            subtree_nodes.extend(current_children)

            new_children = list()
            for child in current_children:
                child_subtrees = self.get_all_subtrees(g, L, child)
                if len(child_subtrees) > 0:
                    subtrees.extend(child_subtrees)
                child_children = g.successors(child)
                if len(child_children) > 0:
                    new_children.extend(child_children)
            current_children = new_children
            subtree_depth += 1

        if subtree_depth == L:
            subtree = g.subgraph(subtree_nodes)
            subtrees.append(subtree)

        return subtrees

    def plot_subtrees(self, subtrees):
        dlist = list()
        for g in subtrees:
            dlist.append({'g':g})
        multi_plot(dlist, self.plot_subtree_func, nrows=3, ncols=3)

    def plot_subtree_func(self, pdata, ax):
        g = pdata['g']
        self.plot_tree(g)

    def show(self, show_probability=False):
        plt.figure()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.plot_tree(self.g, show_probability=show_probability)
        plt.figure()
        self.plot_causal_state(self.causal_state_graph)

        for causal_state,morph in self.morphs.iteritems():
            plt.figure()
            self.plot_tree(morph)
            plt.suptitle('Morph %s' % causal_state)

    def plot_tree(self, g, show_count=False, show_probability=False, show_node_num=False):
        elabels = dict()
        for parent,child in g.edges():
            elabels[(parent,child)] = '%d | %0.2f' % (g[parent][child]['label'], g[parent][child]['probability'])

        nlabels = dict()
        for n in g.nodes():

            if 'causal_state' in g.node[n] and not show_probability:
                nlabels[n] = g.node[n]['causal_state']
            else:
                nlabels[n] = ''
                if show_count:
                    nlabels[n] = g.node[n]['count']
                if show_probability:
                    nlabels[n] = '%0.2f' % g.node[n]['probability']
                if show_node_num:
                    nlabels[n] = '%s' % str(n)


        pos = nx.graphviz_layout(g, prog='dot')
        nx.draw(g, pos, with_labels=True, arrows=True, labels=nlabels, node_size=750)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=elabels)

    def plot_causal_state(self, g):
        elabels = dict()
        for parent,child in g.edges():
            elabels[(parent,child)] = '%0.2f' % (g[parent][child]['probability'])

        nlabels = dict()
        for n in g.nodes():
            nlabels[n] = n

        pos = nx.graphviz_layout(g, prog='dot')
        nx.draw(g, pos, with_labels=True, arrows=True, labels=nlabels, node_size=750)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=elabels)


def multi_plot(data_list, plot_func, title=None, nrows=4, ncols=5):

    nsp = 0
    fig = None
    plots_per_page = nrows*ncols
    for pdata in data_list:
        if nsp % plots_per_page == 0:
            fig = plt.figure()
            fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20)
            if title is not None:
                plt.suptitle(title)

        nsp += 1
        sp = nsp % plots_per_page
        ax = fig.add_subplot(nrows, ncols, sp)
        plot_func(pdata, ax)
