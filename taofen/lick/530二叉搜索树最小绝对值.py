# Author: tangweize
# Date: 2025/3/12 20:02
# Description: 
# Data Studio Task:


def cal_min_abs_value(root):

    prev = float('-inf')
    if not root:
        return float('inf')


    from collections import deque
    stack = []

    res = float('inf')

    while q or root:
        if root:
            stack.append(root)
            root = root.left

        top = stack.pop()
        res = min(res, top.val - prev)
        prev = top.val

        if top.right:
            root = top.right



    return res