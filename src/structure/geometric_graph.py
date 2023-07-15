import random
from typing import List
from collections import defaultdict, OrderedDict
from functools import cmp_to_key

from copy import deepcopy
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data

from src.language.foq import ConjunctiveFormula
from src.structure.knowledge_graph import KnowledgeGraph
from fol import AppFOQEstimator


class QueryGraph(nx.MultiGraph):
    def __init__(self, input_formula: ConjunctiveFormula, device):
        super().__init__()
        index2name, index2edge = OrderedDict(), OrderedDict()
        e_num, f_num = len(input_formula.existential_variable_dict), len(input_formula.free_variable_dict)
        c_num = len(input_formula.term_dict) - e_num - f_num
        for i in range(c_num):
            index2name[i] = 's' + str(i + 1)
        for j in range(e_num):
            index2name[c_num + j] = 'e' + str(j + 1 + c_num)
        for k in range(f_num):
            index2name[c_num + e_num + k] = 'f' + str(k + 1 + c_num + e_num)
        self.add_nodes_from(input_formula.term_dict.keys())
        for pred_name in input_formula.predicate_dict:
            pred = input_formula.predicate_dict[pred_name]
            if pred.negated is False:
                self.add_edge(pred.head.name, pred.tail.name, name=pred_name, type='positive')
            else:
                self.add_edge(pred.head.name, pred.tail.name, name=pred_name, type='negative')
        self.c_num, self.e_num, self.f_num = c_num, e_num, f_num
        self.name2node = {index2name[i]: i for i in index2name}
        self.node_dict = input_formula.term_dict
        self.edge_dict = input_formula.predicate_dict
        self.node_grounded_entity_id_dict = input_formula.term_grounded_entity_id_dict
        self.edge_grounded_entity_id_dict = input_formula.pred_grounded_relation_id_dict
        self.lstr = input_formula.lstr
        ordering_list = self.get_ordering()
        self.ordering = {name: i for i, name in enumerate(ordering_list)}
        self.ordering_list = ordering_list
        self.device = device

    def get_ordering(self):
        """
        Get the ordering for computing the node embedding.
        """
        ordering = []
        constant_node_list = ['s' + str(i + 1) for i in range(self.c_num)]
        for i in range(self.c_num):
            ordering.append('s' + str(i + 1))
        deepcopy_g = deepcopy(self)
        deepcopy_g.remove_nodes_from(constant_node_list)
        initialized_existential, initialized_free = set(), set()
        for node in constant_node_list:
            adj_node_list = list(self.adj[node])
            for adj_node in adj_node_list:
                if 'e' in adj_node:
                    initialized_existential.add(adj_node)
                else:
                    initialized_free.add(adj_node)
        whole_distance_dict = defaultdict(int)
        for free_variable_index in range(self.f_num):
            f_name = f'f{free_variable_index + 1}'
            distance_dict = nx.single_source_shortest_path_length(deepcopy_g, f_name)
            for e_node in distance_dict:
                whole_distance_dict[e_node] += distance_dict[e_node]

        def nx_ordering(g: nx.Graph, to_chose_existential, to_choose_free, existential_distance_dict):
            """
            A bit like bfs, but always postpone the free variable.
            """
            if len(g.nodes) == 1:
                return list(g.nodes)
            for node in g.nodes:
                if len(g.adj[node]) == 1:
                    if 'e' in node:
                        copy_g = deepcopy(g)
                        copy_g.remove_node(node)
                        if node in to_chose_existential:
                            next_node = list(g.adj[node])[0]
                            if 'e' in next_node:
                                to_chose_existential.add(next_node)
                            else:
                                to_choose_free.add(next_node)
                            to_chose_existential.remove(node)
                        left_ordering = nx_ordering(copy_g, to_chose_existential, to_choose_free,
                                                    existential_distance_dict)
                        final_ordering = [node] + left_ordering
                        return final_ordering
            #  We have to choose from the explored nodes here.
            if len(to_chose_existential) > 0:
                choose_node_list = list(to_chose_existential)

                def compare_node(x, y):
                    if existential_distance_dict[x] != existential_distance_dict[y]:
                        return existential_distance_dict[y] - existential_distance_dict[x]  # Choose the farthest one.
                    else:
                        return x < y  # Final method, use string name to compare.
                choose_node_list.sort(key=cmp_to_key(compare_node))
            else:  # Because we know the whole graph is connected, to choose free has nodes.
                choose_node_list = list(to_choose_free)
                choose_node_list.sort()
            to_choose_node = choose_node_list[0]
            copy_g = deepcopy(g)
            copy_g.remove_node(to_choose_node)
            to_chose_existential.discard(to_choose_node)
            to_choose_free.discard(to_choose_node)
            for next_choose_node in g.adj[to_choose_node]:
                if 'e' in next_choose_node:
                    to_chose_existential.add(next_choose_node)
                else:
                    to_choose_free.add(next_choose_node)
            left_ordering = nx_ordering(copy_g, to_chose_existential, to_choose_free, existential_distance_dict)
            final_ordering = [to_choose_node] + left_ordering
            return final_ordering

        ordering = nx_ordering(deepcopy_g, initialized_existential, initialized_free, whole_distance_dict)
        constant_ordering = [f's{i +1}' for i in range(self.c_num)]
        return constant_ordering + ordering

    def get_node_embedding(self, model: AppFOQEstimator, node_name: str, stored_embedding_dict: dict):
        if 's' in node_name:
            ent_emb = model.get_entity_embedding(
                torch.tensor(self.node_grounded_entity_id_dict[node_name]).to(self.device))
            return ent_emb
        adj_edge_list = self.edges(node_name, data=True)
        previous_edge_list = []
        for adj_edge in adj_edge_list:
            adj_node = adj_edge[1]
            if self.ordering[adj_node] < self.ordering[node_name]:
                previous_edge_list.append(adj_edge)
        previous_embedding_list = []
        for adj_edge in previous_edge_list:
            adj_node = adj_edge[1]
            attr_dict = adj_edge[2]
            pred_name = attr_dict['name']
            adj_emb = stored_embedding_dict[adj_node]
            if adj_emb is not None:
                proj_emb = model.get_projection_embedding(
                    torch.tensor(self.edge_grounded_entity_id_dict[pred_name]).to(self.device), adj_emb)
                if attr_dict['type'] == 'positive':
                    previous_embedding_list.append(proj_emb)
                else:
                    previous_embedding_list.append(model.get_negation_embedding(proj_emb))
        if len(previous_embedding_list) == 0:
            return None
        elif len(previous_embedding_list) == 1:
            final_emb = previous_embedding_list[0]
        else:
            final_emb = model.get_conjunction_embedding(previous_embedding_list)
        return final_emb

    def get_whole_graph_embedding(self, model: AppFOQEstimator):
        embedding_dict = {}
        for node in self.ordering_list:
            node_emb = self.get_node_embedding(model, node, embedding_dict)
            embedding_dict[node] = node_emb
        if self.f_num == 1:
            return [embedding_dict['f1']]
        else:
            return [embedding_dict['f1'], embedding_dict['f2']]
