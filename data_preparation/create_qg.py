"""
To create query graph.
"""



import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict, Counter
from itertools import combinations, product, combinations_with_replacement

from copy import deepcopy
import math
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.language.grammar import parse_lstr_to_disjunctive_formula


parser = argparse.ArgumentParser()
parser.add_argument("--max_e", type=int, default=2)
parser.add_argument("--max_f", type=int, default=2)
parser.add_argument("--max_c", type=int, default=3)
parser.add_argument("--max_edge", type=int, default=4)
parser.add_argument("--max_surpass", type=int, default=1)
parser.add_argument("--max_pair", type=int, default=2)
parser.add_argument("--max_distance", type=int, default=3)
parser.add_argument("--max_negation", type=int, default=1)
parser.add_argument("--output_dir", type=str, default='data')
parser.add_argument("--max_total_edge", type=int, default=6)
parser.add_argument("--max_total_node", type=int, default=6)

partition_save = {}

# TODO: Consider EFOX, make sure there is not one variable that only connects to another variable.


def partitions(n, partition_num=1):
    if partition_num == 0:
        return [[]]
    if (n, partition_num) in partition_save:
        return partition_save[(n, partition_num)]
    all_set = set()
    if partition_num == 1:
        return {(n,)}
    for i in range(math.ceil(n/partition_num), n+1):
        sub_partition = partitions(n-i, partition_num-1)
        for p in sub_partition:
            new_list = list(p)
            new_list.append(i)
            new_list.sort(reverse=True)
            all_set.add(tuple(new_list))
    partition_save[(n, partition_num)] = all_set
    return all_set


def create_qg_whole(max_e_node, max_f_node, max_c_node, max_edge,
                    max_surpass_edge, pair_limit, max_f_distance, max_negation_num, max_total_node, max_total_edge):
    """
    The e,f,c is for existential, free and constant node number。
    """
    all_lstr_dict = defaultdict(set)
    for e_num, f_num, c_num, addition_edge in product(
            range(max_e_node + 1), range(1, max_f_node + 1), range(1, max_c_node + 1), range(max_surpass_edge + 1)):
        if addition_edge + e_num + f_num - 1 > max_edge:
            continue
        if addition_edge + e_num + f_num + c_num - 1 > max_total_edge:
            continue
        if e_num + f_num + c_num > max_total_node:
            continue
        qg_list = create_qg(e_num, f_num, c_num, addition_edge, pair_limit, max_f_distance, max_negation_num)
        for qg in qg_list:
            real_addition_edge = len(qg.edges) - e_num - f_num + 1
            lstr = qg2lstr(qg)
            all_lstr_dict[(e_num, f_num, c_num, real_addition_edge)].add((lstr, qg))
    return all_lstr_dict


def create_qg(e_num, f_num, c_num, additional_edge_num, pair_limit, max_f_distance, max_negation_num):
    """
    Three steps:
    1. create ef simple graph ef
    2. create ef multi graph
    3. create efc graph
    4. Check the distance condition.
    5. Check the existential leaf condition.
    6. create the final graph with negation
    """
    all_graph_list = []
    for multi_edge_num in range(additional_edge_num + 1):
        simple_edge_num = additional_edge_num - multi_edge_num + e_num + f_num - 1
        if simple_edge_num > (e_num + f_num) * (e_num + f_num - 1) / 2:
            continue
        if multi_edge_num > pair_limit * simple_edge_num:
            continue
        simple_connected_list = enumerate_connected_graph(e_num, f_num, simple_edge_num)
        for i, simple_connect_g in enumerate(simple_connected_list):
            multi_qg_list = add_multi_edge(simple_connect_g, multi_edge_num, pair_limit)
            for j, multi_qg_ef in enumerate(multi_qg_list):
                epfo_qg_list = add_constants(multi_qg_ef, c_num)
                for k, epfo_qg in enumerate(epfo_qg_list):
                    if distance_condition(epfo_qg, max_f_distance) and existential_leaf_condition(epfo_qg):
                        all_graph_list.append(epfo_qg)
                        negative_qg_list = replace_negation(epfo_qg, max_negation_num)
                        all_graph_list.extend(negative_qg_list)
    return all_graph_list


def replace_negation(graph: nx.MultiGraph, max_negation_num):
    """
    Replace some edges with negation.
    Two further assumptions to make:
    1. Non-constant node must have one positive edge.
    2. There is a positive constant node.
    """
    all_graph_list = []
    non_c_num = 0
    nm = iso.categorical_node_match("type", "existential")
    em = iso.categorical_node_match("type", "positive")
    for i in range(len(graph.nodes())):
        if graph.nodes[i]['type'] == 'constant':
            break
        else:
            non_c_num = i + 1
    multi_edge_list = [e for e in graph.edges]
    free_node_list = [u for u in graph.nodes() if graph.nodes[u]['type'] == "free"]
    max_negation_num = min(max_negation_num, len(multi_edge_list))
    for negative_edge_list in combinations(multi_edge_list, max_negation_num):
        copy_g = deepcopy(graph)
        for negative_edge in negative_edge_list:
            copy_g.remove_edge(*negative_edge)
        for constant in range(non_c_num, len(graph.nodes())):
            distance_dict = nx.single_source_shortest_path_length(copy_g, source=constant)
            free_node_in = [u for u in free_node_list if u in distance_dict]
            if free_node_in:  # We want there is a positive path from a constant to a free.
                break
        else:
            continue
        for non_c_node in range(non_c_num):
            if copy_g.degree[non_c_node] == 0:
                break
        else:  # We have to make sure the negative requirement, non-constant node must have one positive edge.
            copy_g.add_edges_from(negative_edge_list, type='negative')
            for previous_graph in all_graph_list:
                if nx.is_isomorphic(previous_graph, copy_g, node_match=nm, edge_match=em):
                    break
            else:
                all_graph_list.append(copy_g)
    return all_graph_list


def distance_condition(graph: nx.MultiGraph, max_f_distance) -> bool:
    """
    Make sure every node is within max_f_distance hop in the f
    """
    within_dict = {node: False for node in graph.nodes()}
    free_node_list = []
    for i in range(len(graph.nodes())):
        if graph.nodes[i]['type'] == 'free':
            free_node_list.append(i)
    for free_node in free_node_list:
        distance_dict = nx.single_source_shortest_path_length(graph, source=free_node)
        for node in graph.nodes:
            if distance_dict[node] <= max_f_distance:
                within_dict[node] = True
    if False in within_dict.values():
        return False
    else:
        return True


def existential_leaf_condition(graph: nx.MultiGraph) -> bool:
    """
    We omit those query graphs that have existential leaves that are more than depth of 1.
    For example, we omit the following query graph:
    (c1) --[f1]-- (e1) --[e2] since e2 is an existential leaf of depth 2.
    Mathematically speaking, every path from e to an constant must come across a free variable first +
    is only connects to existential nodes.
    """
    copy_graph = deepcopy(graph)
    for node in graph.nodes():
        if graph.nodes[node]['type'] == 'free':
            copy_graph.remove_node(node)
    connected_components = nx.connected_components(copy_graph)
    to_check_list = []
    for component in connected_components:
        for node in component:
            if graph.nodes[node]['type'] == 'constant':
                break
        else:  # In this case, a cluster of existential nodes is ''leaves''.
            for node in component:
                to_check_list.append(node)
    for node in to_check_list:
        for adj_node in graph.adj[node]:
            if graph.nodes[adj_node]['type'] != 'existential':
                break
        else:
            return False
    return True


def add_constants(graph: nx.MultiGraph, c_num):
    """
    Adding constant nodes at last to create EPFO query graphs.
    """
    graph_node_list = deepcopy(graph.nodes())
    non_c_num = len(graph_node_list)
    copy_template = deepcopy(graph)
    copy_template.add_nodes_from(list(range(non_c_num, non_c_num + c_num)), type="constant")
    all_graph_list = []
    nm = iso.categorical_node_match("type", "existential")
    for constant_endpoints in combinations_with_replacement(graph_node_list, c_num):
        copy_g = deepcopy(copy_template)
        for i, endpoint in enumerate(constant_endpoints):
            copy_g.add_edge(non_c_num + i, endpoint)
        for previous_graph in all_graph_list:
            if nx.is_isomorphic(previous_graph, copy_g, node_match=nm):
                break
        else:
            all_graph_list.append(copy_g)
    return all_graph_list
    

def create_qg_ef(e_num, f_num, additional_edge_num, pair_limit):
    """
    Firstly create a simple connected graph, then add multi-edge
    """
    all_graph_list = []
    for multi_edge_num in range(additional_edge_num + 1):
        simple_edge_num = additional_edge_num - multi_edge_num + e_num + f_num - 1
        simple_connected_list = enumerate_connected_graph(e_num, f_num, simple_edge_num)
        for simple_connect_g in simple_connected_list:
            multi_qg_list = add_multi_edge(simple_connect_g, multi_edge_num, pair_limit)
            all_graph_list.extend(multi_qg_list)
    return all_graph_list


def add_multi_edge(graph: nx.MultiGraph, addition_edge: int, pair_limit: int):
    """
    addition_edge is the total number of surpass edge added to a simple connected graph.
    pair_limit controls maximum number of edge allowed in a pair of node.
    """
    if addition_edge == 0:
        return [graph]
    all_graph_list = []
    simple_edge_list = [e for e in graph.edges()]
    nm = iso.categorical_node_match("type", "existential")
    if pair_limit == 2:  # combination without replacement
        multiple_edge_choices = list(combinations(simple_edge_list, addition_edge))
    else:
        multiple_edge_choices = list(combinations_with_replacement(simple_edge_list, addition_edge))
        multiple_edge_choices = [a for a in multiple_edge_choices if max(Counter(a).values()) <= pair_limit]
    for multiple_edge in multiple_edge_choices:
        copy_g = deepcopy(graph)
        copy_g.add_edges_from(multiple_edge, type='positive')
        for previous_graph in all_graph_list:
            if nx.is_isomorphic(previous_graph, copy_g, node_match=nm):
                break
        else:
            all_graph_list.append(copy_g)
    return all_graph_list


def enumerate_connected_graph(e_num, f_num, connect_edge):
    """
    Enumerate the connected simple graph, multi graph is not added.
    Manually returns a list of list, each inner list means that the node degree of out edge. e.g. partition(4):[3,1,0]
    We assume the edge in a "decreasing" manner to avoid duplication, we also note this can be done by inverse relation.
    """
    node_num = e_num + f_num
    all_graph = []
    if f_num == 1:
        all_set = partitions(connect_edge, e_num)
        for partition in all_set:
            if not partition or max(partition) < node_num:  # The maximum should not exceed the node num
                graph_list = partition_2_matrix(partition, [0])
                all_graph.extend(graph_list)
    else:
        all_set = partitions(connect_edge, e_num)
        all_set_2 = partitions(connect_edge - 1, e_num)
        for partition in all_set:
            if not partition or max(partition) < node_num:
                graph_list = partition_2_matrix(partition, [0, 0])
                all_graph.extend(graph_list)
        for partition in all_set_2:
            if not partition or max(partition) < node_num:
                graph_list = partition_2_matrix(partition, [1, 0])
                all_graph.extend(graph_list)
    return all_graph


def partition_2_matrix(e_partition, f_partition):
    e_num, f_num = len(e_partition), len(f_partition)
    choice_dict = {}
    all_graph_list = []
    nm = iso.categorical_node_match("type", "existential")
    for i in range(e_num):
        possible_choice = list(combinations(list(range(i + 1, e_num + f_num + 1)), e_partition[i]))
        choice_dict[i] = possible_choice
    for graph_edge_list in product(*choice_dict.values()):
        g_instance = nx.MultiGraph()
        g_instance.add_nodes_from(list(range(e_num)), type="existential")
        g_instance.add_nodes_from(list(range(e_num, e_num + f_num)), type="free")
        for i, graph_edge_from_i in enumerate(graph_edge_list):
            for end_point in graph_edge_from_i:
                g_instance.add_edge(i, end_point)
        if f_partition[0] == 1:
            g_instance.add_edge(e_num, e_num + 1)
        if nx.is_connected(g_instance):
            for previous_graph in all_graph_list:
                if nx.is_isomorphic(g_instance, previous_graph, node_match=nm):
                    break
            else:
                all_graph_list.append(g_instance)
        else:
            pass
        return all_graph_list


def qg2lstr(query_graph: nx.MultiGraph):
    """
    Return the lstr as the logic formula.
    """
    node2value = {}
    e_num, f_num, c_num, edge_num = 0, 0, 0, 0
    n = len(query_graph.nodes())
    node2name, edge2name = {}, {}
    atomic_list = []
    for node in query_graph.nodes():
        if query_graph.nodes[node]['type'] == 'existential':
            e_num += 1
            node2name[node] = f'e{e_num}'
            node2value[node] = (n + 1) * e_num
        elif query_graph.nodes[node]['type'] == 'free':
            f_num += 1
            node2name[node] = f'f{f_num}'
            node2value[node] = (n + 1) ** 2 * f_num
        else:
            c_num += 1
            node2name[node] = f's{c_num}'
            node2value[node] = c_num
    all_edges = []
    for edge in query_graph.edges:
        all_edges.append(edge)
    all_edges.sort(key=lambda x: min(node2value[x[0]], node2value[x[1]]), reverse=False)
    for edge in all_edges:
        start_point, end_point, pair_index = edge
        if node2value[end_point] < node2value[start_point]:
            start_point, end_point = end_point, start_point
        edge_num += 1
        edge2name[edge] = f'r{edge_num}'
        if 'type' in query_graph.edges[edge] and query_graph.edges[edge]['type'] == 'negative':
            atomic = f'(!{edge2name[edge]}({node2name[start_point]},{node2name[end_point]}))'
        else:
            atomic = f'({edge2name[edge]}({node2name[start_point]},{node2name[end_point]}))'
        atomic_list.append(atomic)
    final_lstr = '&'.join(atomic_list)
    DNF_instance = parse_lstr_to_disjunctive_formula(final_lstr)
    DNF_lstr = DNF_instance.lstr
    recursive_lstr = parse_lstr_to_disjunctive_formula(DNF_lstr).lstr
    assert recursive_lstr == DNF_lstr
    assert len(DNF_instance.formula_list) == 1
    return DNF_lstr


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    new_lstr_dict = create_qg_whole(args.max_e, args.max_f, args.max_c, args.max_edge, args.max_surpass, args.max_pair,
                                    args.max_distance, args.max_negation, args.max_total_node, args.max_total_edge)
    data_dict = defaultdict(list)
    total_count = 0
    for keys in new_lstr_dict:
        e_num, f_num, c_num, edge_num = keys
        for lstr, qg in new_lstr_dict[keys]:
            data_dict['formula_id'].append(f"type{total_count:04d}")
            total_count += 1
            data_dict['formula'].append(lstr)
            data_dict['f_num'].append(f_num)
            data_dict['e_num'].append(e_num)
            data_dict['c_num'].append(c_num)
            data_dict['edge_num'].append(edge_num)
            being_cyclic = (len(set(qg.edges())) != e_num + f_num + c_num - 1)
            being_multiple = len(qg.edges()) != len(set(qg.edges()))
            data_dict['cyclic'].append(being_cyclic)
            data_dict['multi'].append(being_multiple)
    pd.DataFrame(data_dict).to_csv(
        osp.join(args.output_dir, f"DNF_EFO{args.max_f}_{args.max_e}{args.max_c}_{args.max_edge}{args.max_surpass}{args.max_pair}{args.max_distance}{args.max_negation}{args.max_total_node}{args.max_total_edge}_filtered.csv"))






