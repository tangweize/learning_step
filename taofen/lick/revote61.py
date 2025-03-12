# Author: tangweize
# Date: 2025/3/12 10:33
# Description: 
# Data Studio Task:

# 这能用 n-k的效率？？？ 这k不遍历一遍怎么搞？
class Solution:
    def rotateRight(self, head, k: int) -> Optional[ListNode]:

        dummy = ListNode(-1)
        dummy.next = head

        n = 0
        p = head

        while p:
            n += 1
            p = p.next

        #  n== 0会报错
        k = k % n
        # k == 0 不需要后续操作。
        if k == 0:
            return head

        tail = dummy
        while k:
            tail = tail.next
            k -= 1

        prev = dummy

        while tail.next:
            tail = tail.next
            prev = prev.next

        new_head = prev.next
        prev.next = None

        tail.next = head

        return new_head
