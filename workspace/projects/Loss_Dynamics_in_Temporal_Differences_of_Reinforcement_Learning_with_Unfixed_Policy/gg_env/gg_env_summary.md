# Goblets & Ghouls (GG) 环境与 DMFT 记号对照表

本表的目标：把 GG 代码里的对象（GameState、Action、Q-table、线性特征等）
和 DMFT 推导中的记号（\(S_t, A_t, R_{t+1}, \phi, w, Q_w\) 等）一一对应，方便在阅读
和写作 DMFT 推导时始终对齐同一个 toy model。

---

## 1. 环境 / 状态 / 动作 / 奖励

### 1.1 状态 State

- **GG 代码侧**
  - 类型：`gg_core.GameState`
  - 关键字段：
    - `state.board.agent_position  -> (ax, ay)`
    - `state.board.ghost_position  -> (gx, gy)`（可能为 `None`，代码中通常用 `(0, 0)` 占位）
    - `state.board.entities`：墙 (`Wall`)、空格 (`Empty`)、圣杯 (`Goblet`) 等
    - `state.done`：episode 是否结束
    - `state.reward`：最近一步 transition 获得的 reward

- **DMFT 记号**
  - 状态：\(S_t \in \mathcal{S}\)
  - 在 GG toy 中，可以把
    \[
      S_t \equiv (ax_t, ay_t, gx_t, gy_t, \text{board configuration}, \text{goblets})
    \]
  - 在 DMFT 极限中，通常不会直接用到网格的几何细节，而是只通过特征向量 \(\phi(S_t, A_t)\) 进入公式。

### 1.2 动作 Action

- **GG 代码侧**
  - 类型：`gg_core.Action`；在 `q_learning.py` / `q_learning_linear.py` 中有常量：
    ```python
    ACTIONS = [Action.Up, Action.Right, Action.Down, Action.Left]
    ```
  - 动作索引：`0..3` 分别对应 (Up, Right, Down, Left)。

- **DMFT 记号**
  - 动作：\(A_t \in \mathcal{A}\)，其中
    \[
      \mathcal{A} = \{\text{Up}, \text{Right}, \text{Down}, \text{Left}\}
    \]
  - 在特征表示中，动作往往通过 one-hot 或特征张量索引进入 \(\phi(S_t, A_t)\)。

### 1.3 转移核 Transition Kernel

- **GG 代码侧**
  - 配置文件：`default.yaml` 中
    ```yaml
    agent:
        transition: [0.7, 0.1, 0.1, 0.1]
    ```
    表示 slip kernel \([a,b,c,d]\)。
  - README 中给出了含义：  
    - \(a\)：执行**期望动作**的概率；
    - \(b\)：顺时针偏移一格；
    - \(c\)：反向（偏移两格）；
    - \(d\)：逆时针偏移一格。  
  - 在 `q_learning.py` / `q_learning_linear.py` 中，函数
    `sample_actual_action_index(intended_idx, transition, rng)` 使用这组概率矩阵把「期望动作」映射成「真正执行动作」。

- **DMFT 记号**
  - 转移核：\(P(S_{t+1}\mid S_t, A_t)\) 由 slip kernel 和 ghost policy 等共同决定。
  - 在 DMFT 中，我们把它作为「环境随机性」的一部分进行平均。

### 1.4 奖励 Reward

- **GG 代码侧**
  - 奖励来自 `state.reward`，主要由捡到的 goblet (正/负奖励) 和撞到 ghost / 障碍决定。
  - 每一步 Q-learning 更新中使用 `reward_next = float(next_state.reward)`。

- **DMFT 记号**
  - 奖励：\(R_{t+1}\)，作为 TD 误差的一部分：
    \[
      \delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t).
    \]

---

## 2. Tabular Q-Learning（teacher-baseline）与 DMFT 对应

### 2.1 Tabular Q-table

- **GG 代码侧**
  - 函数：`initialize_q_table(config: GGConfig)`  
    返回一个维度约为 `(width, height, width, height, |A|)` 的 Q-table：
    \[
      Q_{\text{tab}}(ax, ay, gx, gy, a)
    \]
  - 更新规则（简化）：
    \[
      Q(S_t, A_t) \leftarrow Q(S_t, A_t)
        + \alpha \bigl[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\bigr].
    \]

- **DMFT 记号**
  - Tabular Q 可以看作用 one-hot 特征 \(\phi_{\text{tab}}(S_t, A_t)\) 做函数逼近：
    \[
      Q_{\text{tab}}(S_t, A_t)
        = w^\top \phi_{\text{tab}}(S_t, A_t),
    \]
    其中 \(w\) 是超高维 one-hot 权重向量。
  - 在 DMFT 分析中，tabular 版本通常作为「完备函数逼近 / teacher」的极限模型。

### 2.2 行为策略与 on-policy / off-policy

- **GG 代码侧**
  - Tabular Q-learning 中使用 `epsilon_greedy`：
    \[
      \pi(a\mid s) =
      \begin{cases}
      1-\varepsilon, & a = \arg\max_a Q(s,a)\\
      \varepsilon/(|\mathcal{A}|-1), & \text{otherwise}.
      \end{cases}
    \]
  - GG 实验中常见两种使用方式：
    - teacher Q-learning：用自己的 Q-table 做 ε-greedy → **on-policy**；
    - 用某个 teacher Q-table 作为行为策略，而让别的模型在这个策略下训练 → **off-policy**。

- **DMFT 记号**
  - 行为策略：\(\pi_{\text{beh}}(a\mid s)\)；
  - 目标策略：\(\pi_{\text{tar}}(a\mid s)\)（比如 greedy policy w.r.t. teacher 或 student Q）。
  - on-policy：\(\pi_{\text{beh}} = \pi_{\text{tar}}\)；  
    off-policy：\(\pi_{\text{beh}} \neq \pi_{\text{tar}}\)，是“死亡三角”里的关键一角。

---

## 3. 线性函数逼近 Q-learning（toy student）与 DMFT 对应

### 3.1 高斯特征张量 \(\phi(s,a)\)

- **GG 代码侧**
  - 函数：`initialize_gaussian_features(config, feature_dim, rng)`  
    构造一个张量 `phi`：
    ```python
    phi.shape = (width, height, width, height, len(ACTIONS), feature_dim)
    ```
  - 采样方式（DMFT 风格）：
    \[
      \phi_i(s,a) \sim \mathcal{N}\bigl(0, 1/\sqrt{N}\bigr),
    \]
    其中 \(N = \text{feature\_dim}\)。  
    这保证每个特征向量 \(\phi(s,a)\) 的范数是 \(O(1)\)。

- **DMFT 记号**
  - 特征向量：\(\phi(S_t, A_t) \in \mathbb{R}^N\)，成分：
    \[
      \phi_i(S_t, A_t) \sim \mathcal{N}\bigl(0, 1/\sqrt{N}\bigr),
    \]
    正是高斯等价假设中的典型特征构造方式。
  - 在 DMFT 极限中，我们做 \(N \to \infty\) 的极限，并使用高斯等价来求解生成函数和序参量。

### 3.2 线性 Q 函数与参数 \(w\)

- **GG 代码侧**
  - 学生参数：`w: np.ndarray`，形状 `(feature_dim,)`。
  - Q 函数：
    ```python
    Q_w(s, a) = phi[s, a] @ w
    ```
    在代码中通过各种 `q_values_linear_for_state` / `_q_linear_for_indices` 实现。
  - TD 目标函数（文档字符串中明确）：
    \[
      \delta_t^{QL}(w) = R_{t+1} + \gamma \max_a Q_w(S_{t+1}, a) - Q_w(S_t, A_t),
    \]
    半梯度更新：
    \[
      w_{t+1} = w_t + \alpha\, \delta_t^{QL}(w_t)\, \phi(S_t, A_t).
    \]

- **DMFT 记号**
  - 参数向量：\(w(t) \in \mathbb{R}^N\)；
  - 线性 Q 函数：\(Q_w(S_t, A_t) = w(t)^\top \phi(S_t, A_t)\)；
  - 这是 Pehlevan paper 中 TD with linear function approximation 的标准设定。

### 3.3 行为模式 `behaviour="fixed"` vs `"unfixed"`

- **GG 代码侧**
  - 在 `q_learning_step_linear(...)` 中：
    - 参数 `behaviour: str = "unfixed"`；
    - 当 `behaviour == "fixed" and q_teacher is not None` 时：  
      - 行为策略用 teacher Q-table：
        \[
          \pi_{\text{beh}}(a\mid s) \propto \epsilon\text{-greedy}(Q_{\text{teacher}}(s,a));
        \]
    - 否则（`behaviour != "fixed"`）：  
      - 行为策略用当前学生 Q 函数 \(Q_w\)（ε-greedy）。

- **DMFT 记号**
  - `behaviour="fixed"`：
    - 行为策略 = teacher policy（固定不随 \(w\) 变化），学生在此策略下学习 → 典型 **off-policy 固定策略死亡三角**（任务 2）。
  - `behaviour="unfixed"`：
    - 行为策略由学生的 Q 函数决定，随 \(w\) 变化 → **off-policy + 非固定策略死亡三角**（任务 3）。

---

## 4. 预测误差损失 \(L_{\text{pred}}\) 与参考分布 \(d_{\text{ref}}\)

- **GG 代码侧**
  - 函数 `_build_reference_indices(config, stride)`：
    - 构造一组参考索引集合 \((ax, ay, gx, gy, a)\)，定义在整个网格上（可带 stride），对应一个参考分布 \(d_{\text{ref}}(s,a)\)。
  - 预测误差损失：
    \[
      L_{\text{pred}}(w) = \mathbb{E}_{(s,a) \sim d_{\text{ref}}}
        \bigl[ (Q_{\text{teacher}}(s,a) - Q_w(s,a))^2 \bigr].
    \]
  - 实现中通过 `_q_linear_for_indices` 等函数在参考集合上评估 teacher Q-table 和学生 Q 函数。

- **DMFT 记号**
  - 参考分布：\(d_{\text{ref}}(s,a)\)；
  - 预测误差损失：\(L_{\text{pred}}(w)\)；
  - 在 DMFT 分析中，可以用这个损失刻画「学生 Q 与 teacher Q 的逼近质量」，并在 order parameters 中引入相关的重叠量，比如
    \[
      R(t) = \frac{1}{N} w(t)^\top w^\*,
    \]
    或类似的 teacher–student overlap。

---

## 5. 总结：从 GG 到 DMFT 的映射

一个典型的「一步更新」可以在 GG 代码和 DMFT 记号之间这样对应：

- 在 GG 中：
  1. 当前状态 `state` 对应 \(S_t\)；
  2. 根据行为策略（由 teacher 或学生 Q 决定）和 slip kernel 采样动作 \(A_t\)；
  3. 环境根据 \(P(S_{t+1}\mid S_t, A_t)\) 给出 `next_state` 和 `reward_next` → 对应 \(S_{t+1}, R_{t+1}\)；
  4. 用 TD 误差 \(\delta_t^{QL}(w)\) 更新参数 `w`。

- 在 DMFT 中：
  1. 把 \((S_t, A_t)\) 映射到高斯特征向量 \(\phi(S_t, A_t)\)；
  2. 将环境随机性、策略随机性、初始权重随机性等做平均；
  3. 通过 generating function + 高斯等价 + 序参量 + 鞍点 + HS 变换，得到关于 \(\{Q(t,t'), R(t,t'), \dots\}\) 的有效闭合方程；
  4. 这些闭合方程刻画了在 GG 这一类环境族下，线性 TD / Q-learning 在「死亡三角」设置中的典型学习曲线和可能的发散行为。

在后续的 DMFT 推导中，如果出现记号歧义或想确认“某个变量在 GG 里实际对应什么”，可以随时回到本词汇表对照。

