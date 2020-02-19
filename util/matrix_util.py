# -*- coding: UTF-8 -*-

"""
第一步，根据用户物品二分图graph得到M矩阵
第二步，根据M矩阵得到单位矩阵E 减去 α*M矩阵的转置
第三步，对第二步得到的稀疏矩阵求逆
"""
import sys

from scipy.sparse import coo_matrix
import numpy as np
from util import read


def graph_to_matrix(graph):
    """
    第一步，根据用户物品二分图graph得到M矩阵
    :param graph: user item graph
    :return:
        a coo_matrix, sparse matrix M
        a list, total user item nodes
        a dict, map all the nodes to row index
    """
    # 使用一个数组来保存所有顶点（user节点 + item节点）
    vertex = list(graph.keys())
    total_len = len(vertex)
    # 记录每个顶点在vertex中的位置，从而知道每一行对应的是哪个顶点
    # node : address_index in vertex
    address_dict = {item[1]: item[0] for item in enumerate(vertex)}
    # 根据coo_matrix这种稀疏矩阵的形式，它需要3个数组分别来存储行索引、列索引 以及 对应的数值
    rows = []
    cols = []
    values = []
    for outer_node in graph:
        weight = round(1 / len(graph[outer_node]), 3)
        row_index = address_dict[outer_node]
        for inner_node in graph[outer_node]:
            col_index = address_dict[inner_node]
            rows.append(row_index)
            cols.append(col_index)
            values.append(weight)
    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)
    M = coo_matrix((values, (rows, cols)), shape=(total_len, total_len))
    return M, vertex, address_dict


def maxtrix_all_nodes(m_mat, vertex, alpha):
    """
    第二步，根据M矩阵得到单位矩阵E 减去 α*M矩阵的转置
    get E - alpha*m_mat.T
    :param m_mat:
    :param vertex: total user item nodes
    :param alpha: the prob for random walk
    :return: a sparse matrix
    """
    total_len = len(vertex)
    rows = []
    cols = []
    values = []
    # 初始化单位矩阵E，这也是一个稀疏矩阵，只有对角线上的元素为1
    for i in range(total_len):
        rows.append(i)
        cols.append(i)
        values.append(1)
    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)
    E = coo_matrix((values, (rows, cols)), shape=(total_len, total_len))
    # 使用csr格式，这种格式可以使得运算变得快一些
    return E.tocsr() - alpha * m_mat.tocsr().transpose()


if __name__ == '__main__':
    graph = read.get_graph_from_data("../data/log.csv")
    m, vertex, address_dict = graph_to_matrix(graph)
    # print(address_dict)
    # print(m.todense())
    print(maxtrix_all_nodes(m, vertex, alpha=0.8).todense())
