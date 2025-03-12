# Author: tangweize
# Date: 2025/3/12 19:55
# Description: 
# Data Studio Task:
# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
#
# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”


def find_common_father(root, p, q):
    if not root :
        return None

    if root == p or root == q:
        return root

    l1 = find_common_father(root.left, p, q)
    l2 = find_common_father(root.right, p, q)

    if not l1 and l2:
        return l2

    if l1 and not l2:
        return l1

    if l1 and l2:
        return root

    return None

