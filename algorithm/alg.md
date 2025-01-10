## LeetCode 622 - 设计循环队列 (Design Circular Queue)

- **题目链接**：[LeetCode 622](https://leetcode.com/problems/design-circular-queue/)
- **公司标签**：Optiver
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
- **公司标签**：Optiver
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
