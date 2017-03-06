#!/usr/sbin/python
#-*- encoding:utf-8 -*-
#
# http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
#

import operator
import numpy as np

with open('user_prefs.txt') as f:
    prefs_str = ''.join(f.readlines())

# {'andy': {'霍乱时期的爱情': 1},...}
def read_prefs(prefs_str):
    itemset = set()
    userset = set()
    prefs = {}
    for line in prefs_str.split('\n'):
        parts = line.rstrip().split()
        if len(parts) == 2:
            userId, itemId = parts
            itemset.add(itemId)
            userset.add(userId)
            prefs.setdefault(userId, {})
            prefs[userId].update({itemId:1})

    mat = np.empty((len(userset), len(itemset)))
    for user in prefs:
        for item in prefs[user]:
            i = list(userset).index(user)
            j = list(itemset).index(item)
            mat[i][j] = prefs[user][item]
    return prefs, mat, list(itemset), list(userset)

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T


prefs, mat, items, users = read_prefs(prefs_str)

print(mat)
print(items)
print(users)

R = mat
    # [
    #  [5,3,0,1],
    #  [4,0,0,1],
    #  [1,1,0,5],
    #  [1,0,0,4],
    #  [0,1,5,4],
    # ]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02)
nR = np.dot(nP, nQ.T)
# print(repr(R))
# print(repr(nP))
# print(repr(nQ))
# print(repr(nR))

print('\n======= 用户矩阵 ========')
for i in range(0, len(nP)):
    print(users[i], nP[i])

print('\n======= 物品矩阵 ========')
for i in range(0, len(nQ)):
    print(items[i], nQ[i])

print('\n======= Matrix Factorization 推荐结果 ========')
for i in range(0, len(nP)):
    for j in range(0, len(nQ)):
        score = np.sum(nP[i] * nQ[j])
        if score > 1.1 and R[i][j] == 0:
            print(users[i], items[j], score)