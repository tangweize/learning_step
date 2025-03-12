# Author: tangweize
# Date: 2025/3/12 19:48
# Description: 
# Data Studio Task:
# 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
#
# 完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，
# 并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层（从第 0 层开始），则该层包含 1~ 2h 个节点。

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import deque
class Solution:
    def countNodes(self, root) -> int:
        q = deque()

        q.append(root)

        res  = 0
        while q:
            top = q.popleft()
            res += 1

            if top.left:
                q.append(top.left)
            if top.right:
                q.append(top.right)

        return res


