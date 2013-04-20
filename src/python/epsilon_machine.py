import networkx as nx

import matplotlib.pyplot as plt


class BinaryParseTree(object):

    def __init__(self, tree_depth, past_length, future_length):
        self.D = tree_depth
        self.L = future_length
        self.K = past_length

        self.construct_tree()

    def construct_tree(self):
        self.g = nx.DiGraph()
        self.g.add_node(0)

        self.add_child_nodes(self.g, 0, 0, self.D)

    def add_child_nodes(self, g, node, current_depth, max_depth):

        if current_depth >= max_depth:
            return

        zero_node = len(g.nodes())
        g.add_node(zero_node, count=0)
        g.add_edge(node, zero_node, label=0)
        self.add_child_nodes(g, zero_node, current_depth+1, max_depth)

        one_node = len(g.nodes())
        g.add_node(one_node, count=0)
        g.add_edge(node, one_node, label=1)
        self.add_child_nodes(g, one_node, current_depth+1, max_depth)

    def parse(self, process_string):

        #build up word counts
        M = len(process_string)
        nwords = M - self.D + 1
        for t in range(self.D, M+1):
            word = process_string[t-self.D:t]
            self.update_count(self.g, word, 0)

        #normalize word counts to probabilities
        for parent,child in self.g.edges():
            self.g.node[child]['p'] = float(self.g.node[child]['count']) / nwords

        #prune graph
        self.g.remove_nodes_from([n for n in self.g.nodes() if n != 0 and self.g.node[n]['count'] == 0])

    def update_count(self, g, word, node):
        #print '[update_count] node=%d, word=' % node,word

        for child in g.successors(node):
            if g[node][child]['label'] == word[0]:
                g.node[child]['count'] += 1
                if len(word) > 1:
                    self.update_count(g, word[1:], child)

    def show(self, show_count=False):
        elabels = dict()
        for parent,child in self.g.edges():
            elabels[(parent,child)] = '%d' % self.g[parent][child]['label']

        nlabels = dict()
        for n in self.g.nodes():
            if n == 0:
                nlabels[n] = '0'
            else:
                if show_count:
                    nlabels[n] = self.g.node[n]['count']
                else:
                    nlabels[n] = '%0.2f' % self.g.node[n]['p']

        plt.figure()
        pos = nx.graphviz_layout(self.g, prog='dot')
        nx.draw(self.g, pos, with_labels=True, arrows=True, labels=nlabels, node_size=750)
        nx.draw_networkx_edge_labels(self.g, pos, edge_labels=elabels)
