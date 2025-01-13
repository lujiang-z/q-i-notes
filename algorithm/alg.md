## LeetCode 622 - 设计循环队列 (Design Circular Queue)

- **题目链接**：[LeetCode 622](https://leetcode.com/problems/design-circular-queue/)
- **公司标签**：Op
- **难度**：中等（Medium）

### 题目描述：
设计并实现一个循环队列的结构。该队列应该有一个固定的最大长度，当队列为空或满时，能够正确地返回 `True` 或 `False`。
操作包括：
- `enqueue(value)`：插入一个元素到队尾。
- `dequeue()`：从队首移除一个元素。
- `front()`：获取队首元素。
- `rear()`：获取队尾元素。
- `isFull()`：检查队列是否已满。
- `isEmpty()`：检查队列是否为空。

### 考点：
- 队列（Queue）
- 环形队列（Circular Queue）
- 数组实现

### 解题思路：
- 使用固定大小的数组来存储队列的元素，并维护一个头指针和尾指针。
- 当队列满时，可以通过回绕头尾指针来插入和删除元素，从而模拟循环行为。
- 判断队列是否满或空时，需要特别注意指针的比较。


## LeetCode 641 - 设计循环双端队列 (Design Circular Deque)

- **题目链接**：[LeetCode 641](https://leetcode.com/problems/design-circular-deque/)
- **公司标签**：
- **难度**：中等（Medium）

### 题目描述：
设计一个双端队列的结构，允许从队列的两端插入和删除元素。操作包括：
- `insertFront(value)`：在队列头部插入元素。
- `insertLast(value)`：在队列尾部插入元素。
- `deleteFront()`：从队列头部删除元素。
- `deleteLast()`：从队列尾部删除元素。
- `getFront()`：获取队列头部元素。
- `getRear()`：获取队列尾部元素。
- `isEmpty()`：检查队列是否为空。
- `isFull()`：检查队列是否已满。

### 考点：
- 双端队列（Deque）
- 环形队列（Circular Queue）
- 数组实现

### 解题思路：
- 使用一个固定大小的数组来实现双端队列，维护两个指针：一个指向队头，另一个指向队尾。
- 支持从队列两端进行插入和删除操作。通过回绕队头和队尾指针来模拟环形行为。
- 判断队列是否满或空时，需要特别注意指针的比较。

## LeetCode 207 - 课程表 (Course Schedule)

- **题目链接**：[LeetCode 207](https://leetcode.com/problems/course-schedule/)
- **公司标签**：Op
- **难度**：中等（Medium）

### 题目描述：
你这个学期必须选修一些课程。给定课程的总数 `numCourses` 和一个数组 `prerequisites`，其中 `prerequisites[i] = [ai, bi]` 表示在你可以选修课程 `ai` 之前，你必须先选修课程 `bi`。
  
请你判断是否可以完成所有课程的学习任务。若能完成返回 `true`，否则返回 `false`。

### 考点：
- 拓扑排序（Topological Sorting）
- 有向图（Directed Graph）
- 广度优先搜索（BFS）

### 解题思路：
1. **建图**：使用邻接表（Adjacency List）存储图。每一门课的前置课可以看作是图中的一条有向边。
2. **入度（In-degree）**：对于每门课程，记录它的入度（即有多少门课程是它的前置课程）。如果某门课程的入度为零，表示没有前置课程，可以选修。
3. **BFS（拓扑排序）**：
   - 初始化队列，先将所有入度为零的课程入队。
   - 然后逐一弹出队列中的课程，减小其后继课程的入度。若某个后继课程的入度变为零，则将其加入队列。
   - 若最后可以遍历所有课程，则说明可以完成所有课程，返回 `true`；否则，返回 `false`。

## LeetCode 410 - 分割数组的最大值 (Split Array Largest Sum)

- **题目链接**：[LeetCode 410](https://leetcode.com/problems/split-array-largest-sum/)
- **公司标签**：JT
- **难度**：困难（Hard）

### 题目描述：
给定一个非负整数数组 `nums` 和一个整数 `m`，你需要将数组分割成 `m` 个非空的连续子数组。设计一个算法，使得这 `m` 个子数组各自和的最大值最小化。

**示例：**

```plaintext
输入: nums = [7,2,5,10,8], m = 2
输出: 18
解释:
一共有四种方法将 nums 分割为 2 个子数组。
其中最优的方式是将其分为 [7,2,5] 和 [10,8]，因为此时这两个子数组的和分别为 14 和 18，18 是所有可能中的最小最大值。
```

### 考点：
- 动态规划（Dynamic Programming）
- 数组分割（Array Partitioning）
- 最优化问题（Optimization Problems）

### 解题思路：动态规划

#### 一、状态定义
- **`dp[i][j]`**：表示将前 `i` 个元素分割成 `j` 个子数组的最小的最大子数组和。

#### 二、状态转移方程
- 对于每一个 `dp[i][j]`，尝试所有可能的前一个划分点 `k`（`j-1 <= k < i`）：
  
  $$
  dp[i][j] = \min_{k=j-1}^{i-1} \{ \max(dp[k][j-1], \text{sum}(k+1, i)) \}
  $$
  
  其中，`sum(k+1, i)` 表示从第 `k+1` 个到第 `i` 个元素的和。


#### 三、前缀和预处理
为了高效地计算任意区间的和，预先计算前缀和数组：
  
$$
prefix[i] = \sum_{t=1}^{i} nums[t-1]
$$
  
这样，任意区间 `[k+1, i]` 的和可以通过 `prefix[i] - prefix[k]` 快速得到。




## LeetCode 706 - 设计哈希映射 (Design HashMap)

- **题目链接**：[LeetCode 706](https://leetcode.com/problems/design-hashmap/)
- **公司标签**：TS
- **难度**：简单（Easy）

### 题目描述：

不使用任何内建的哈希表库，设计一个哈希映射（`HashMap`）。

实现 `MyHashMap` 类：

- `MyHashMap()` 用空映射初始化对象。
- `void put(int key, int value)` 向 HashMap 插入一个键值对 `(key, value)`。如果 key 已经存在于映射中，则更新其对应的值 `value`。
- `int get(int key)` 返回特定的 key 所映射的 value；如果映射中不包含 key 的映射，返回 `-1`。
- `void remove(key)` 如果映射中存在 key 的映射，则移除 key 和它所对应的 value。

**示例：**

```plaintext
输入：
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]

输出：
[null, null, null, 1, -1, null, 1, null, -1]

解释：
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // myHashMap 现在为 [[1,1]]
myHashMap.put(2, 2); // myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(1);    // 返回 1
myHashMap.get(3);    // 返回 -1（未找到）
myHashMap.put(2, 1); // myHashMap 现在为 [[1,1], [2,1]]（更新已有的值）
myHashMap.get(2);    // 返回 1
myHashMap.remove(2); // 删除键为 2 的数据，myHashMap 现在为 [[1,1]]
myHashMap.get(2);    // 返回 -1（未找到）
```

### 考点：
- 哈希函数（Hash Function）
- 冲突解决（Collision Resolution）
- 链地址法（Separate Chaining）



### 解题思路：

我们将使用**链地址法**来处理哈希冲突。具体步骤如下：

1. **初始化哈希表**：
   - 定义哈希表的大小（如 `769`）。
   - 创建一个包含 `size` 个空列表的列表，每个子列表作为一个桶，用于存储键值对。

2. **哈希函数**：
   - 使用简单的取模运算将键映射到桶的索引位置。

3. **`put` 方法**：
   - 计算键的哈希值，找到对应的桶。
   - 遍历桶中的所有键值对，若找到相同的键，则更新其值并返回。
   - 若未找到相同的键，则在桶中添加新的键值对。

4. **`get` 方法**：
   - 计算键的哈希值，找到对应的桶。
   - 遍历桶中的所有键值对，若找到相同的键，则返回其值。
   - 若未找到相同的键，则返回 `-1`。

5. **`remove` 方法**：
   - 计算键的哈希值，找到对应的桶。
   - 遍历桶中的所有键值对，若找到相同的键，则删除该键值对并返回。



## LeetCode 547 - 省份数量 (Number of Provinces)

- **题目链接**：[LeetCode 547](https://leetcode.com/problems/number-of-provinces/)
- **公司标签**：TS
- **难度**：中等（Medium）

### 题目描述：

有 `n` 个城市，城市之间通过一个 `n x n` 的邻接矩阵 `isConnected` 表示其直接连接关系。

- 如果 `isConnected[i][j] = 1`，则表示城市 `i` 和城市 `j` 是直接相连的。
- 如果 `isConnected[i][j] = 0`，则表示城市 `i` 和城市 `j` 之间没有直接相连。

省份被定义为一个或多个通过直接或间接连接而形成的城市组。

请你返回矩阵 `isConnected` 中省份的数量。

**示例：**

```plaintext
输入:
isConnected = [
  [1,1,0],
  [1,1,0],
  [0,0,1]
]

输出: 2

解释:
城市 0 和城市 1 直接相连，形成一个省份。
城市 2 单独成为一个省份。
```

### 考点：
- 并查集（Union-Find）
- 图的遍历（DFS/BFS）
- 图论基础（连通分量）

### 解题思路：

本题可以被视为一个图论问题，其中每个城市代表一个节点，直接连接关系代表无向边。我们的目标是找到图中的连通分量数量，每个连通分量对应一个省份。

有两种主要的方法来解决这个问题：

1. **并查集（Union-Find）**：
   - 使用并查集数据结构来管理和合并连通分量。
   - 初始化每个城市为一个独立的集合。
   - 遍历 `isConnected` 矩阵，对于每对直接连接的城市，执行合并操作。
   - 最终，不同的根节点数量即为省份的数量。

2. **深度优先搜索（DFS）或广度优先搜索（BFS）**：
   - 将城市和连接关系建模为图的邻接表。
   - 遍历所有城市，未访问的城市启动一次DFS/BFS，标记所有可达的城市为已访问。
   - 每次启动DFS/BFS时，省份计数加一。



## LeetCode 1801 - 积压订单中的订单总数 (Number of Orders in the Backlog) **(LOB)**

- **题目链接**：[LeetCode 1801](https://leetcode.com/problems/number-of-orders-in-the-backlog/)
- **公司标签**：TS
- **难度**：中等（Medium）

### 题目描述：

你有一个订单表，记录了商品的买卖订单。每个订单都包含三个整数：

- `price`：订单的价格
- `amount`：订单的数量
- `orderType`：订单类型（0 表示买单，1 表示卖单）

一个订单表是一个二维整数数组 `orders`，其中 `orders[i] = [price_i, amount_i, orderType_i]` 表示第 `i` 个订单的详细信息。

存在一个积压订单表，初始时为空。按照以下规则处理订单表中的每个订单：

1. 如果当前订单是买单，则尝试将其与积压卖单中价格最低且不高于买单价格的卖单进行匹配。如果找到这样的卖单，则匹配数量为 `min(amount_i, amount_j)`，其中 `amount_j` 是卖单的数量。匹配后，更新订单的数量。如果卖单被完全匹配，则将其从积压订单中移除。否则，减少其数量。
2. 如果当前订单是卖单，则尝试将其与积压买单中价格最高且不少于卖单价格的买单进行匹配。匹配数量为 `min(amount_i, amount_j)`，其中 `amount_j` 是买单的数量。匹配后，更新订单的数量。如果买单被完全匹配，则将其从积压订单中移除。否则，减少其数量。
3. 如果当前订单未能完全匹配，则将其余部分加入积压订单中。

在处理完所有订单后，返回积压订单中所有订单数量的总和，对 `10^9 + 7` 取模。

**示例：**

```plaintext
输入：
orders = [
  [10, 5, 0],
  [15, 2, 1],
  [25, 1, 1],
  [30, 4, 0]
]

输出：6

解释：
订单 1：买单，价格 10，数量 5 → 无可匹配的卖单，加入积压买单。
订单 2：卖单，价格 15，数量 2 → 与积压买单匹配，但买单价格 10 < 卖单价格 15，无法匹配，加入积压卖单。
订单 3：卖单，价格 25，数量 1 → 无可匹配的买单，加入积压卖单。
订单 4：买单，价格 30，数量 4 → 匹配卖单价格最低的 15，匹配数量 2。剩余买单数量 2 → 继续匹配卖单价格 25，匹配数量 1。剩余买单数量 1，加入积压买单。

积压订单总数 = 5（买单） + 1（卖单） = 6
```

### 考点：

- **优先队列（堆）**：用于高效地获取当前最佳匹配订单（最低卖价或最高买价）。
- **贪心算法**：每次选择最优的匹配以尽可能多地成交订单。
- **模拟**：逐步处理订单并维护积压订单的状态。

### 解题思路：

本题要求模拟订单的处理过程，维护一个积压订单表。为了高效地匹配订单，我们可以使用两个优先队列：

1. **买单堆**：一个最大堆，按照价格从高到低排序，用于快速获取价格最高的买单。
2. **卖单堆**：一个最小堆，按照价格从低到高排序，用于快速获取价格最低的卖单。

处理每个订单时：

- **买单**：
  - 尝试与卖单堆中价格最低且不高于买单价格的卖单进行匹配。
  - 进行匹配后，更新买单和卖单的数量，必要时移除完全匹配的卖单。
  - 如果买单未完全匹配，则将其余部分加入买单堆。

- **卖单**：
  - 尝试与买单堆中价格最高且不少于卖单价格的买单进行匹配。
  - 进行匹配后，更新卖单和买单的数量，必要时移除完全匹配的买单。
  - 如果卖单未完全匹配，则将其余部分加入卖单堆。

最后，遍历买单堆和卖单堆，计算所有积压订单的总数量。

### 代码实现：

```
class Solution {
public:
    int getNumberOfBacklogOrders(vector<vector<int>>& orders) {
        const int MOD = 1000000007;
        priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>> buyOrders;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> sellOrders;
        for (auto &&order : orders) {
            int price = order[0], amount = order[1], orderType = order[2];
            if (orderType == 0) {
                while (amount > 0 && !sellOrders.empty() && sellOrders.top().first <= price) {
                    auto sellOrder = sellOrders.top();
                    sellOrders.pop();
                    int sellAmount = min(amount, sellOrder.second);
                    amount -= sellAmount;
                    sellOrder.second -= sellAmount;
                    if (sellOrder.second > 0) {
                        sellOrders.push(sellOrder);
                    }
                }
                if (amount > 0) {
                    buyOrders.emplace(price, amount);
                }
            } else {
                while (amount > 0 && !buyOrders.empty() && buyOrders.top().first >= price) {
                    auto buyOrder = buyOrders.top();
                    buyOrders.pop();
                    int buyAmount = min(amount, buyOrder.second);
                    amount -= buyAmount;
                    buyOrder.second -= buyAmount;
                    if (buyOrder.second > 0) {
                        buyOrders.push(buyOrder);
                    }
                }
                if (amount > 0) {
                    sellOrders.emplace(price, amount);
                }
            }
        }
        int total = 0;
        while (!buyOrders.empty()) {
            total = (total + buyOrders.top().second) % MOD;
            buyOrders.pop();
        }
        while (!sellOrders.empty()) {
            total = (total + sellOrders.top().second) % MOD;
            sellOrders.pop();
        }
        return total;
    }
};


```

### 复杂度分析：

- **时间复杂度**：O(N log N)，其中 N 是订单的数量。每个订单最多会进行一次堆的插入和若干次堆的弹出操作，每次堆操作的时间复杂度为 O(log N)。
- **空间复杂度**：O(N)，用于存储买单堆和卖单堆