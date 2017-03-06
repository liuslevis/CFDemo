#!/usr/sbin/python
#-*- encoding:utf-8 -*-

import operator

with open('user_prefs.txt') as f:
    prefs_str = ''.join(f.readlines())

# {'andy': {'霍乱时期的爱情': 1},...}
def read_prefs(prefs_str):
    prefs = {}
    for line in prefs_str.split('\n'):
        parts = line.rstrip().split()
        if len(parts) == 2:
            userId, itemId = parts
            prefs.setdefault(userId, {})
            prefs[userId].update({itemId:1})
    return prefs

prefs = read_prefs(prefs_str)

def jaccard_distance(prefs, user1, user2):
    s1 = set(prefs[user1].keys())
    s2 = set(prefs[user2].keys())
    return 1.0 * len(s1.intersection(s2)) / len(s1.union(s2))

# 找出与 user1 兴趣最相近的用户 user1 -> [(1.0, user3), (0.8, user4), ...]
def top_matches(prefs, user1, similarity, n = 5):
    scores = [(similarity(prefs, user1, user2), user2) for user2 in prefs if user1 != user2]
    scores.sort()
    scores.reverse()
    return scores[0:n]

# prefs -> "用户 xx 根据 xx 推荐 xx 书籍 xxx"
def calculate_user_cf(prefs, similarity, n = 10):
    ret = {}
    for user in prefs.keys():
        scores = top_matches(prefs, user, similarity, n)
        ret[user] = scores
    return ret

def print_recomendation(prefs, similiar_users, min_score = 0.1):
    # 对每个用户
    for target_user in similiar_users:
        itemId_cnt = {}
        # 找出兴趣相近的用户
        for score, similiar_user in similiar_users[target_user]:
            if score > min_score:
                   # 统计兴趣相近用户最喜欢的书，排序
                for itemId in set(prefs[similiar_user]) - set(prefs[target_user]):
                    itemId_cnt.setdefault(itemId, 0)
                    itemId_cnt[itemId] += 1
        recommends_itemId_cnt = sorted(itemId_cnt.items(), key=operator.itemgetter(1), reverse=True)
        print('\n用户:%s\n\t喜欢:%s\n\t相似用户:%s\n\t推荐:%s' % (target_user, list(prefs[target_user].keys()), list(filter(lambda score_user:score_user[0] > min_score, similiar_users[target_user])), recommends_itemId_cnt))

print('\n基于用户的协同过滤推荐')
similiar_users = calculate_user_cf(prefs, jaccard_distance, n = 10)
print_recomendation(prefs, similiar_users)

# 矩阵行列互换 prefs[userId][itemId] -> prefs[itemId][userId]
def transpose_prefs(prefs):
    ret = {}
    for userId in prefs:
        for itemId in prefs[userId]:
            ret.setdefault(itemId, {})
            ret[itemId][userId] = prefs[userId][itemId]
    return ret

def calculate_item_cf(prefs, similarity, n = 10):
    ret = {}
    itemPrefs = transpose_prefs(prefs)
    # 对于每个物品，找出用户评价向量距离相近的物品
    for item in itemPrefs:
        scores = top_matches(itemPrefs, item, similarity, n)
        ret[item] = scores
    return ret

def print_similiar_items(similiar_items, min_score = 0.1):
    for target_itemId in similiar_items:
        print('\n根据 %s 推荐:' % target_itemId)
        score_items = similiar_items[target_itemId]
        for score_item in score_items:
            score, itemId = score_item
            if score > min_score: print('\t%s %f' % (itemId, score))

print('\n基于书籍的协同过滤推荐')
similiar_items = calculate_item_cf(prefs, jaccard_distance, n = 10)
print_similiar_items(similiar_items)
