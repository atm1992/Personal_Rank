# -*- coding: UTF-8 -*-
import operator
from util import read, matrix_util
from scipy.sparse.linalg import gmres
import numpy as np


def personal_rank(graph, root, alpha, iter_num, recom_num=10):
    """
    personal rank算法的基础版本实现，该版本的时间复杂度很高
    :param graph: user item graph
    :param root: the fix user for which to recommendation
    :param alpha: the prob to go to random walk
    :param iter_num: iteration num
    :param recom_num: recommend item num
    :return: a dict, key itemId, value PR
    """
    rank = {node: 0 for node in graph}
    rank[root] = 1
    recom_result = {}
    for i in range(iter_num):
        tmp_rank = {node: 0 for node in graph}
        for outer_node, outer_dict in graph.items():
            for inner_node, value in outer_dict.items():
                tmp_rank[inner_node] += round(alpha * rank[outer_node] / len(outer_dict), 4)
                if inner_node == root:
                    tmp_rank[inner_node] += round(1 - alpha, 4)
        # 若这一次得到的rank与上一次得到的rank一致，说明PR值已经收敛了，可以停止迭代
        if tmp_rank == rank:
            print("迭代到了第" + str(i + 1) + "次收敛。次数从1开始")
            break
        rank = tmp_rank
    # 对rank按PR值排序，过滤掉user节点 以及 与root user节点已经有过行为的item节点
    for node, pr_value in sorted(rank.items(), key=operator.itemgetter(1), reverse=True):
        # 如果该节点不是item节点，则忽略
        if len(node.split('_')) < 2:
            continue
        # 如果是root用户节点行为过的item节点，也将忽略
        if node in graph[root]:
            continue
        recom_result[node] = pr_value
        recom_num -= 1
        if recom_num == 0:
            break
    return recom_result


def personal_rank_matrix(graph, root, alpha, recom_num=10):
    """

    :param graph: user item graph
    :param root: fix user to recom
    :param alpha: the prob to random walk
    :param recom_num: the length of recom items
    :return:
        a dict, key itemId, value PR
    Ax = E, 矩阵x就是矩阵A的逆矩阵
    """
    m, vertex, address_dict = matrix_util.graph_to_matrix(graph)
    if root not in address_dict:
        return {}
    # 保存item顶点以及PR值
    score_dict = {}
    recom_dict = {}
    # 对matrix_all稀疏矩阵求逆，便可得到所有顶点的推荐结果
    matrix_all = matrix_util.maxtrix_all_nodes(m, vertex, alpha)
    # 得到r0矩阵，负责选取某一节点为固定节点
    root_index = address_dict[root]
    # 一个M+N行、1列的二维数组（即 矩阵）
    initial_list = [[0] for i in range(len(vertex))]
    initial_list[root_index] = [1]
    r0 = np.array(initial_list)
    # tol 表示允许的误差。gmres得到的结果是一个tuple
    # 该tuple的第一个值为一个数组，保存着其余所有顶点对root节点的PR值
    res = gmres(matrix_all, r0, tol=1e-8)[0]
    for i in range(len(res)):
        node = vertex[i]
        # 如果该节点不是item节点，则忽略
        if len(node.split('_')) < 2:
            continue
        # 如果是root用户节点行为过的item节点，也将忽略
        if node in graph[root]:
            continue
        score_dict[node] = round(res[i], 3)
    for item_node, pr_value in sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True):
        recom_dict[item_node] = pr_value
    return recom_dict


def get_one_user_recom():
    user = "1"
    alpha = 0.7
    graph = read.get_graph_from_data("../data/ratings.csv")
    iter_num = 100
    recom_result = personal_rank(graph, user, alpha, iter_num, 100)
    # item_info = read.get_item_info("../data/movies.csv")
    # for itemId in graph[user]:
    #     pure_itemId = itemId.split("_")[1]
    #     print(item_info[pure_itemId])
    # print("-" * 100)
    # for itemId in recom_result:
    #     pure_itemId = itemId.split("_")[1]
    #     print(item_info[pure_itemId])
    #     print(recom_result[itemId])
    return recom_result


def get_one_user_recom_matrix():
    user = "1"
    alpha = 0.7
    graph = read.get_graph_from_data("../data/ratings.csv")
    recom_result = personal_rank_matrix(graph, user, alpha, 100)
    return recom_result


if __name__ == '__main__':
    # 在工业界中通常使用Spark来并行计算，同时计算多个用户的推荐结果
    recom_result_base = get_one_user_recom()
    recom_result_matrix = get_one_user_recom_matrix()
    num = 0
    for ele in recom_result_base:
        if ele in recom_result_matrix:
            num += 1
    print(num)
