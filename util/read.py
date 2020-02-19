# -*- coding: UTF-8 -*-
import csv
import os


def get_graph_from_data(input_file):
    """

    :param input_file: userId,movieId,rating file
    :return: a dict {userA:{item_b:1,item_c:1},item_b:{userA:1}} 表示哪些user行为过哪些item，以及哪些item被哪些user行为过
    """
    if not os.path.exists(input_file):
        return {}
    graph = {}
    score_thr = 4.0
    with open(input_file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for item in data:
            if len(item) < 3:
                continue
            userId, movieId, rating = item[0], "item_" + item[1], float(item[2])
            if rating < score_thr:
                continue
            if userId not in graph:
                graph[userId] = {}
            graph[userId][movieId] = 1
            if movieId not in graph:
                graph[movieId] = {}
            graph[movieId][userId] = 1
    return graph


def get_item_info(input_file):
    """获取每个电影的标题以及分类"""
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    with open(input_file, newline='') as f:
        data = csv.reader(f)
        header = next(data)
        for item in data:
            if len(item) < 3:
                continue
            elif len(item) == 3:
                movieId, title, genres = item[0], item[1], item[2]
            elif len(item) > 3:
                movieId = item[0]
                genres = item[-1]
                # 有些电影名称中可能会出现逗号，
                title = ",".join(item[1:-1])
            item_info[movieId] = [title, genres]
    return item_info

if __name__ == '__main__':
    graph = get_graph_from_data("../data/ratings.csv")
    # 指定userId为"1"
    print(graph["1"])
