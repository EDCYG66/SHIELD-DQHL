import numpy as np

class ReplayMemory:
    def __init__(self, model_dir, memory_size=500000, state_dim=102, batch_size=2000):
        """
        经验回放池
        - model_dir: 模型路径占位（与历史代码兼容）
        - memory_size: 回放池容量（默认 50 万）
        - state_dim: 状态维度（Agent 中拼接 GNN 嵌入20 + 原始状态82 = 102）
        - batch_size: 采样批量（默认 2000；在 speed_mode 下 Agent 会改为 256）
        """
        self.model_dir = model_dir
        self.memory_size = int(memory_size)
        self.state_dim = int(state_dim)

        # 为节省显存/内存，底层用 float16 存状态；采样时转为 float32
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.prestate = np.empty((self.memory_size, self.state_dim), dtype=np.float16)
        self.poststate = np.empty((self.memory_size, self.state_dim), dtype=np.float16)

        self.batch_size = int(batch_size)
        self.count = 0
        self.current = 0

    def size(self):
        return self.count

    def add(self, prestate, poststate, reward, action):
        """
        写入一条经验
        - prestate/poststate: 一维数组或可转换为 (state_dim,) 的序列
        - reward: float
        - action: int
        """
        self.actions[self.current] = int(action)
        self.rewards[self.current] = np.float32(reward)
        self.prestate[self.current] = np.asarray(prestate, dtype=np.float16)
        self.poststate[self.current] = np.asarray(poststate, dtype=np.float16)

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_batch(self, prestate, poststate, rewards, actions):
        """
        批量写入经验；保持环形缓冲区语义。
        """
        prestate = np.asarray(prestate, dtype=np.float16).reshape(-1, self.state_dim)
        poststate = np.asarray(poststate, dtype=np.float16).reshape(-1, self.state_dim)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        n = min(prestate.shape[0], poststate.shape[0], rewards.shape[0], actions.shape[0])
        if n <= 0:
            return

        remaining = n
        src_start = 0
        while remaining > 0:
            dst_end = min(self.memory_size, self.current + remaining)
            chunk = dst_end - self.current
            src_end = src_start + chunk
            self.prestate[self.current:dst_end] = prestate[src_start:src_end]
            self.poststate[self.current:dst_end] = poststate[src_start:src_end]
            self.rewards[self.current:dst_end] = rewards[src_start:src_end]
            self.actions[self.current:dst_end] = actions[src_start:src_end]

            self.count = min(self.memory_size, max(self.count, dst_end))
            self.current = dst_end % self.memory_size
            remaining -= chunk
            src_start = src_end

    def sample(self):
        """
        随机采样 batch_size 条经验，返回 float32 的状态与奖励，int32 的动作
        """
        if self.count == 0:
            raise RuntimeError("ReplayMemory is empty.")
        idxs = np.random.randint(self.count, size=self.batch_size)

        prestate = self.prestate[idxs].astype(np.float32, copy=False)   # (B, state_dim)
        poststate = self.poststate[idxs].astype(np.float32, copy=False) # (B, state_dim)
        actions = self.actions[idxs].astype(np.int32, copy=False)       # (B,)
        rewards = self.rewards[idxs].astype(np.float32, copy=False)     # (B,)

        return prestate, poststate, actions, rewards
