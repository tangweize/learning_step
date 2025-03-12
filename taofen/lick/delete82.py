# Author: tangweize
# Date: 2025/3/12 10:35
# Description: 
# Data Studio Task:



# [82] 删除排序链表中的重复元素 II
#

# @lc code=start
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next  [1,2,3,3,4,4,5]



class Solution:
    def deleteDuplicates(self, head):
        dummy = ListNode(-1)
        prev = dummy

        l = head
        p = head

        while p:
            while p and p.val == l.val:
                p = p.next

            if l.next == p:
                prev.next = l
                prev = prev.next
                prev.next = None

            l = p

        return dummy.next
