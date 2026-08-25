---
title: "LeetCode Hot 100：Python 解题思路与 API 速查手册"
date: 2026-08-02 13:00:02
updated: 2026-08-02 13:00:02
description: "以题型识别、核心状态、模板代码和边界陷阱为主线整理 LeetCode Hot 100。"
categories:
  - 面试
  - 面试准备
tags:
  - LeetCode
  - Python
  - 算法
---

> 目标：不背 100 份独立答案，而是掌握可迁移的题型模板。
>
> 复习顺序：**识别信号 → 核心状态 → 模板代码 → 边界陷阱**。

参考：[LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/)

---

## 1. 题目特征与算法选择

| 题目特征 | 优先想到 | 典型题 |
|---|---|---|
| 查找配对、去重、计数 | 哈希表、集合 | 两数之和、字母异位词分组、最长连续序列 |
| 有序数组、两端移动 | 双指针 | 三数之和、盛最多水的容器 |
| 最长或最短连续区间 | 滑动窗口 | 无重复字符最长子串、最小覆盖子串 |
| 连续子数组和等于 K | 前缀和 + 哈希 | 和为 K 的子数组 |
| 原地调整数组 | 快慢指针、下标置换 | 移动零、缺失的第一个正数 |
| 区间重叠 | 排序 + 扫描 | 合并区间 |
| 链表倒数、环、相交 | 快慢指针 | 环形链表、相交链表 |
| 树的路径、深度、结构 | DFS 递归 | 最大深度、直径、最近公共祖先 |
| 树的逐层处理 | BFS + `deque` | 层序遍历、右视图 |
| 图的连通性、最短步数 | DFS/BFS | 岛屿数量、腐烂的橘子 |
| 枚举所有选择 | 回溯 | 全排列、子集、组合总和 |
| 有序或答案具有单调性 | 二分查找 | 搜索旋转数组、二分答案 |
| 下一个更大或更小元素 | 单调栈 | 每日温度、柱状图最大矩形 |
| 动态维护最大或最小 K 个 | 堆 | 数组第 K 大、前 K 个高频元素 |
| 局部选择可导出全局最优 | 贪心 | 跳跃游戏、划分字母区间 |
| 最优值、方案数、可达性 | 动态规划 | 打家劫舍、零钱兑换、最长递增子序列 |
| 两个字符串或两个维度 | 二维 DP | 最长公共子序列、编辑距离 |

---

## 2. 推荐导入清单

```python
from collections import Counter, defaultdict, deque
from functools import cache
from bisect import bisect_left, bisect_right
from itertools import accumulate, combinations, permutations
import heapq
import math
```

注意：是 `collections`，不是 `collection`。

---

## 3. 哈希表

### 3.1 两数之和：先查再存

```python
pos = {}

for i, x in enumerate(nums):
    other = target - x
    if other in pos:
        return [pos[other], i]
    pos[x] = i
```

先查再存，可以避免同一个元素被使用两次。

### 3.2 按特征分组

```python
from collections import defaultdict

groups = defaultdict(list)

for s in strs:
    key = ''.join(sorted(s))
    groups[key].append(s)

return list(groups.values())
```

也可以使用字符计数元组作为 key：

```python
count = [0] * 26
for ch in s:
    count[ord(ch) - ord('a')] += 1

key = tuple(count)  # list 不可哈希，必须转为 tuple
```

### 3.3 最长连续序列

只从连续序列起点扩展：

```python
seen = set(nums)
answer = 0

for x in seen:
    if x - 1 not in seen:
        y = x
        while y in seen:
            y += 1
        answer = max(answer, y - x)
```

---

## 4. 双指针

### 4.1 相向双指针

```python
left, right = 0, len(nums) - 1

while left < right:
    if 满足条件:
        ...
    elif 当前结果偏小:
        left += 1
    else:
        right -= 1
```

典型用途：有序数组求和、回文判断、两端选择。

### 4.2 三数之和

```python
nums.sort()
answer = []

for i in range(len(nums) - 2):
    if i > 0 and nums[i] == nums[i - 1]:
        continue
    if nums[i] > 0:
        break

    left, right = i + 1, len(nums) - 1

    while left < right:
        total = nums[i] + nums[left] + nums[right]

        if total < 0:
            left += 1
        elif total > 0:
            right -= 1
        else:
            answer.append([nums[i], nums[left], nums[right]])
            left += 1
            right -= 1

            while left < right and nums[left] == nums[left - 1]:
                left += 1
            while left < right and nums[right] == nums[right + 1]:
                right -= 1
```

### 4.3 快慢指针原地修改数组

```python
slow = 0

for fast in range(len(nums)):
    if nums[fast] != 0:
        nums[slow], nums[fast] = nums[fast], nums[slow]
        slow += 1
```

---

## 5. 滑动窗口

适用于连续子串或子数组，以及最长、最短、至多、至少等问题。

### 5.1 通用模板

```python
left = 0

for right, x in enumerate(nums):
    # 将 x 加入窗口

    while 窗口不合法:
        # 移除 nums[left]
        left += 1

    # 更新答案
```

### 5.2 无重复字符的最长子串

```python
last = {}
left = 0
answer = 0

for right, ch in enumerate(s):
    if ch in last:
        left = max(left, last[ch] + 1)

    last[ch] = right
    answer = max(answer, right - left + 1)
```

必须使用 `max`，否则左边界可能倒退。

### 5.3 最小覆盖子串

维护四个核心量：

- `need`：目标字符频数；
- `window`：窗口字符频数；
- `valid`：有多少种字符已达到目标频数；
- `left/right`：窗口边界。

达到 `valid == len(need)` 后，不断收缩左边界并更新最短答案。

---

## 6. 前缀和

若 `prefix[i]` 表示 `nums[:i]` 的和，则区间 `[left, right]` 的和为：

```python
prefix[right + 1] - prefix[left]
```

### 和为 K 的子数组

```python
from collections import defaultdict

count = defaultdict(int)
count[0] = 1

prefix = 0
answer = 0

for x in nums:
    prefix += x
    answer += count[prefix - k]
    count[prefix] += 1
```

`count[0] = 1` 表示空前缀，用于统计从数组开头开始的合法子数组。

---

## 7. 数组、区间与矩阵

### 7.1 最大子数组和：Kadane

```python
current = answer = nums[0]

for x in nums[1:]:
    current = max(x, current + x)
    answer = max(answer, current)
```

### 7.2 合并区间

```python
intervals.sort(key=lambda x: x[0])
merged = []

for left, right in intervals:
    if not merged or left > merged[-1][1]:
        merged.append([left, right])
    else:
        merged[-1][1] = max(merged[-1][1], right)
```

### 7.3 矩阵方向数组

```python
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

for dr, dc in directions:
    nr, nc = row + dr, col + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        ...
```

### 7.4 顺时针旋转矩阵 90°

先沿主对角线转置，再反转每一行：

```python
n = len(matrix)

for i in range(n):
    for j in range(i + 1, n):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

for row in matrix:
    row.reverse()
```

### 7.5 螺旋矩阵

维护四条边界：

```python
top, bottom = 0, rows - 1
left, right = 0, cols - 1
```

每走完一条边就向内收缩。遍历下边和左边前，要重新检查边界是否仍然有效。

---

## 8. 链表

### 8.1 虚拟头节点

```python
dummy = ListNode(0, head)
prev = dummy

# 操作链表

return dummy.next
```

虚拟头节点可统一处理删除头节点、合并链表等边界情况。

### 8.2 反转链表

```python
prev = None
cur = head

while cur:
    nxt = cur.next
    cur.next = prev
    prev = cur
    cur = nxt

return prev
```

记忆：保存后面 → 反转指针 → 两指针前进。

### 8.3 找环与环入口

```python
slow = fast = head

while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

    if slow is fast:
        break
else:
    return None

p = head
while p is not slow:
    p = p.next
    slow = slow.next

return p
```

节点身份比较用 `is`，因为要判断是否为同一个节点，而不是节点值是否相等。

### 8.4 删除倒数第 N 个节点

```python
dummy = ListNode(0, head)
fast = slow = dummy

for _ in range(n):
    fast = fast.next

while fast.next:
    fast = fast.next
    slow = slow.next

slow.next = slow.next.next
return dummy.next
```

### 8.5 链表归并排序

1. 快慢指针寻找中点；
2. 切断链表；
3. 递归排序左右两半；
4. 合并两个有序链表。

时间复杂度 `O(n log n)`。

---

## 9. 二叉树 DFS

### 9.1 通用递归模板

```python
def dfs(node):
    if not node:
        return 边界值

    left = dfs(node.left)
    right = dfs(node.right)

    return 根据 left、right、node.val 计算结果
```

遍历位置：

- 前序：进入节点时处理，适合构造路径、复制结构；
- 中序：左右子树之间处理，BST 中得到有序序列；
- 后序：左右子树处理完再计算，适合深度、直径和子树信息。

### 9.2 二叉树直径

```python
answer = 0

def depth(node):
    nonlocal answer
    if not node:
        return 0

    left = depth(node.left)
    right = depth(node.right)

    answer = max(answer, left + right)
    return max(left, right) + 1
```

`depth` 返回节点深度，同时用 `left + right` 更新经过当前节点的最长路径。

### 9.3 最近公共祖先

```python
def lowestCommonAncestor(root, p, q):
    if not root or root is p or root is q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root
    return left or right
```

---

## 10. 二叉树 BFS

```python
from collections import deque

queue = deque([root])
result = []

while queue:
    level = []

    for _ in range(len(queue)):
        node = queue.popleft()
        level.append(node.val)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    result.append(level)
```

进入每层时，`range(len(queue))` 固定了这一层的节点数。

不要使用列表的 `pop(0)` 模拟队列，因为头部删除是 `O(n)`。

---

## 11. 图：DFS、BFS、拓扑排序

### 11.1 岛屿问题：DFS 淹没

```python
def dfs(r, c):
    if not (0 <= r < rows and 0 <= c < cols):
        return
    if grid[r][c] != '1':
        return

    grid[r][c] = '0'

    for dr, dc in directions:
        dfs(r + dr, c + dc)
```

进入节点后立即标记，避免重复搜索。

### 11.2 多源 BFS

把所有起点同时加入队列：

```python
queue = deque(all_sources)
steps = 0

while queue:
    for _ in range(len(queue)):
        node = queue.popleft()
        # 扩展相邻节点
    steps += 1
```

适用于多个火源同时扩散、多个起点到其他位置的最短距离。

### 11.3 拓扑排序

```python
from collections import deque

queue = deque(i for i in range(n) if indegree[i] == 0)
visited = 0

while queue:
    node = queue.popleft()
    visited += 1

    for nxt in graph[node]:
        indegree[nxt] -= 1
        if indegree[nxt] == 0:
            queue.append(nxt)

return visited == n
```

处理的节点数少于总数，说明图中存在环。

---

## 12. 回溯

### 12.1 通用模板

```python
def backtrack(状态):
    if 满足结束条件:
        answer.append(path.copy())
        return

    for choice in choices:
        if choice 不合法:
            continue

        path.append(choice)
        backtrack(新状态)
        path.pop()
```

记忆：选择 → 递归 → 撤销选择。

### 12.2 子集

```python
answer = []
path = []

def dfs(start):
    answer.append(path.copy())

    for i in range(start, len(nums)):
        path.append(nums[i])
        dfs(i + 1)
        path.pop()
```

### 12.3 全排列

```python
used = [False] * len(nums)

def dfs():
    if len(path) == len(nums):
        answer.append(path.copy())
        return

    for i, x in enumerate(nums):
        if used[i]:
            continue

        used[i] = True
        path.append(x)
        dfs()
        path.pop()
        used[i] = False
```

### 12.4 同层去重

先排序，然后跳过同一层的相同元素：

```python
if i > start and nums[i] == nums[i - 1]:
    continue
```

“同层去重”和“同一路径不能重复选择”是两种不同限制。

---

## 13. 二分查找

### 13.1 闭区间模板

```python
left, right = 0, len(nums) - 1

while left <= right:
    mid = left + (right - left) // 2

    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        left = mid + 1
    else:
        right = mid - 1

return -1
```

### 13.2 左右边界

```python
from bisect import bisect_left, bisect_right

i = bisect_left(nums, x)   # 第一个 >= x 的位置
j = bisect_right(nums, x)  # 第一个 > x 的位置
```

`x` 的出现次数：

```python
bisect_right(nums, x) - bisect_left(nums, x)
```

### 13.3 二分答案

```python
left, right = 最小可能答案, 最大可能答案

while left < right:
    mid = (left + right) // 2

    if check(mid):
        right = mid
    else:
        left = mid + 1

return left
```

使用前提：`check(x)` 的真假变化具有单调性。

---

## 14. 栈与单调栈

### 14.1 括号匹配

```python
pairs = {')': '(', ']': '[', '}': '{'}
stack = []

for ch in s:
    if ch not in pairs:
        stack.append(ch)
    elif not stack or stack.pop() != pairs[ch]:
        return False

return not stack
```

### 14.2 每日温度：单调栈

```python
stack = []
answer = [0] * len(temperatures)

for i, temperature in enumerate(temperatures):
    while stack and temperatures[stack[-1]] < temperature:
        j = stack.pop()
        answer[j] = i - j

    stack.append(i)
```

栈中存下标，因为答案通常需要距离或位置。

### 14.3 柱状图哨兵

```python
heights = [0] + heights + [0]
```

首尾哨兵可以统一弹栈逻辑，避免循环结束后额外清空栈。

---

## 15. 堆 `heapq`

Python 的 `heapq` 默认是小根堆，`heap[0]` 是最小元素。

```python
import heapq

heap = []
heapq.heappush(heap, x)
smallest = heapq.heappop(heap)
peek = heap[0]
```

### 15.1 原地建堆

```python
heapq.heapify(nums)
```

`heapify()` 原地修改列表，返回值是 `None`。不要写：

```python
heap = heapq.heapify(nums)  # 错误
```

### 15.2 大根堆

兼容性最好的方式是存负数：

```python
heapq.heappush(heap, -x)
largest = -heapq.heappop(heap)
```

### 15.3 第 K 大

维护大小为 K 的小根堆：

```python
heap = []

for x in nums:
    heapq.heappush(heap, x)
    if len(heap) > k:
        heapq.heappop(heap)

return heap[0]
```

### 15.4 元组优先队列

```python
heapq.heappush(heap, (distance, node))
distance, node = heapq.heappop(heap)
```

元组先比较第一项，第一项相同时继续比较后续项。对象不可比较时，加入唯一序号：

```python
from itertools import count

counter = count()
heapq.heappush(heap, (priority, next(counter), obj))
```

### 15.5 易混 API

```python
heapq.heappushpop(heap, x)  # 先加入，再弹出最小值
heapq.heapreplace(heap, x)  # 先弹出原堆顶，再加入；堆必须非空
heapq.nlargest(k, nums)
heapq.nsmallest(k, nums)
```

---

## 16. 贪心

### 16.1 跳跃游戏

```python
farthest = 0

for i, jump in enumerate(nums):
    if i > farthest:
        return False
    farthest = max(farthest, i + jump)

return True
```

### 16.2 贪心证明思路

- 当前选择不会使未来更差；
- 只保留覆盖最远、结束最早或成本最低的状态；
- 可以把某个最优解的一步替换为当前选择，而不降低答案。

如果无法说明局部最优为什么能推出全局最优，题目可能需要动态规划。

---

## 17. 动态规划

设计 DP 时依次回答：

1. `dp[i]` 或 `dp[i][j]` 表示什么？
2. 当前状态从哪些旧状态转移？
3. 初始状态和遍历顺序是什么？

### 17.1 打家劫舍

```python
prev2 = 0
prev1 = 0

for money in nums:
    current = max(prev1, prev2 + money)
    prev2, prev1 = prev1, current

return prev1
```

### 17.2 完全背包：零钱兑换

```python
dp = [float('inf')] * (amount + 1)
dp[0] = 0

for coin in coins:
    for value in range(coin, amount + 1):
        dp[value] = min(dp[value], dp[value - coin] + 1)
```

- 完全背包：容量通常正序；
- 0/1 背包：容量通常倒序，避免同一物品使用多次。

### 17.3 最长递增子序列

```python
from bisect import bisect_left

tails = []

for x in nums:
    i = bisect_left(tails, x)

    if i == len(tails):
        tails.append(x)
    else:
        tails[i] = x

return len(tails)
```

`tails` 不一定是真实子序列；它保存各个长度下尽可能小的结尾值。

### 17.4 最长公共子序列

```python
m, n = len(text1), len(text2)
dp = [[0] * (n + 1) for _ in range(m + 1)]

for i in range(1, m + 1):
    for j in range(1, n + 1):
        if text1[i - 1] == text2[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
```

---

## 18. Python 冷门但高频的用法

### 18.1 `deque`

```python
from collections import deque

q = deque()
q.append(x)       # 右侧加入
q.appendleft(x)   # 左侧加入
q.pop()           # 右侧弹出
q.popleft()       # 左侧弹出
q[0]              # 队首
q[-1]             # 队尾
```

### 18.2 `defaultdict`

```python
from collections import defaultdict

graph = defaultdict(list)
count = defaultdict(int)
groups = defaultdict(set)
```

传入的是工厂函数：

```python
defaultdict(list)  # 正确
defaultdict([])    # 错误
```

### 18.3 `Counter`

```python
from collections import Counter

count = Counter(nums)
count[x]
count.most_common(3)
```

```python
c1 + c2
c1 - c2
c1 & c2  # 各键取较小计数
c1 | c2  # 各键取较大计数
```

### 18.4 排序

```python
nums.sort()                         # 原地修改，返回 None
new_nums = sorted(nums)             # 返回新列表
items.sort(key=lambda x: x[1])
items.sort(key=lambda x: (x[0], -x[1]))
nums.sort(reverse=True)
```

不要写：

```python
nums = nums.sort()  # nums 会变成 None
```

### 18.5 `enumerate` 与 `zip`

```python
for i, x in enumerate(nums):
    ...

for i, x in enumerate(nums, start=1):
    ...

for x, y in zip(a, b):
    ...
```

矩阵转置：

```python
transposed = [list(row) for row in zip(*matrix)]
```

### 18.6 字符与字符串

```python
ord('a')
chr(97)
index = ord(ch) - ord('a')

''.join(chars)
' '.join(words)

ch.isdigit()
ch.isalpha()
ch.isalnum()
```

字符串不可修改，需要先转列表：

```python
chars = list(s)
chars[i] = 'x'
s = ''.join(chars)
```

### 18.7 切片

```python
nums[left:right]  # 左闭右开
nums[::-1]        # 反转副本
nums[:k]
nums[k:]
```

切片会创建新对象；递归中频繁切片会增加时间和空间开销。

### 18.8 数学

```python
import math

math.inf
-math.inf
math.gcd(a, b)
math.lcm(a, b)
math.ceil(x)
math.floor(x)
math.isqrt(x)
```

Python 的 `//` 向负无穷取整：

```python
-3 // 2 == -2
```

若要求整数除法向零截断，可避免浮点数：

```python
sign = -1 if (a < 0) ^ (b < 0) else 1
result = sign * (abs(a) // abs(b))
```

### 18.9 位运算

```python
x & 1             # 判断奇偶
x & (x - 1)       # 去掉最低位的 1
x & -x            # 提取最低位的 1
x.bit_count()     # 二进制中 1 的数量
```

判断正整数是否为 2 的幂：

```python
x > 0 and x & (x - 1) == 0
```

异或性质：

```python
x ^ x == 0
x ^ 0 == x
```

### 18.10 递归记忆化

```python
from functools import cache

@cache
def dfs(i, state):
    if 结束条件:
        return ...
    return ...
```

缓存参数必须可哈希，不能直接传入 `list`、`set`、`dict`；可改用 `tuple` 或位掩码。

### 18.11 `itertools`

```python
from itertools import accumulate, combinations, permutations

prefix = list(accumulate(nums, initial=0))
pairs = list(combinations(nums, 2))
orders = list(permutations(nums))
```

这些函数返回迭代器。回溯题如果需要剪枝、去重或展示算法过程，通常仍应手写回溯。

---

## 19. 最容易踩的 Python 坑

### 19.1 二维数组浅拷贝

错误：

```python
grid = [[0] * cols] * rows
```

正确：

```python
grid = [[0] * cols for _ in range(rows)]
```

错误写法中所有行都指向同一个列表。

### 19.2 回溯结果必须复制

```python
answer.append(path.copy())
# 或 answer.append(path[:])
```

直接 `append(path)` 保存的是同一个列表引用。

### 19.3 不要边遍历边删除列表元素

```python
for x in nums:
    nums.remove(x)  # 容易跳过元素
```

优先使用新数组、双指针、下标或倒序遍历。

### 19.4 `None` 与假值

```python
if not x:
```

会同时匹配 `None`、`0`、空字符串和空容器。若只判断空对象：

```python
if x is None:
```

### 19.5 `and` / `or` 返回的不一定是布尔值

```python
return left or right
```

返回第一个真值对象；若都为假，则返回最后一个假值对象。

### 19.6 递归深度

Python 深递归可能触发 `RecursionError`。链状树或大图可以改用显式栈：

```python
stack = [start]

while stack:
    node = stack.pop()
```

---

## 20. 建议刷题顺序与复盘方法

推荐顺序：

1. 哈希、双指针、滑动窗口、前缀和；
2. 数组、矩阵、链表；
3. 二叉树 DFS/BFS；
4. 图、回溯、二分；
5. 栈、堆、贪心；
6. 一维 DP、背包、二维 DP。

每道题复盘只回答三个问题：

```text
识别信号：为什么看出是这个题型？
核心状态：窗口、栈、dp 或递归保存了什么？
边界陷阱：空输入、重复值、左右端点、初始化是什么？
```

真正需要熟记的是十几个模板：

- 滑动窗口；
- 前缀和 + 哈希；
- 链表反转与快慢指针；
- 树 DFS/BFS；
- 图搜索与拓扑排序；
- 回溯；
- 二分查找与二分答案；
- 单调栈；
- 堆；
- 贪心；
- 一维、背包和二维动态规划。

其余 Hot 100 题目，大多是这些模板的变形或组合。
