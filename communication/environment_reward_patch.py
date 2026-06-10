import numpy as np

def compute_rb_gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    s = counts.sum()
    if s <= 0:
        return 0.0
    diffs = np.abs(counts[:, None] - counts[None, :])
    return float(diffs.sum() / (2 * counts.size * s))

def apply_reward_adjustments(
    base_reward_matrix: np.ndarray,
    actions_rb: np.ndarray,
    actions_pw: np.ndarray,
    individual_time_limit: np.ndarray,
    V2V_limit: float,
    rb_anti_conc_alpha: float = 0.02,
    rb_hot_threshold: float = 0.18,
    rb_softmask_alpha: float = 0.25,
    urgency_threshold: float = 0.30,
    beta_urgency_pos: float = 0.02,
    beta_urgency_neg: float = 0.03,
    n_rb: int = 20,
):
    actions_rb = np.asarray(actions_rb, dtype=np.int32)
    actions_pw = np.asarray(actions_pw, dtype=np.int32)
    individual_time_limit = np.asarray(individual_time_limit, dtype=np.float64)
    n_rb = int(max(1, n_rb))

    # 1) RB 反集中化惩罚
    valid_rb = (actions_rb >= 0) & (actions_rb < n_rb)
    rb_counts = np.bincount(actions_rb[valid_rb].reshape(-1), minlength=n_rb).astype(np.int32)
    rb_gini = compute_rb_gini(rb_counts)
    penalty_gini = rb_anti_conc_alpha * rb_gini

    total_links = max(1, int(np.sum(actions_rb >= 0)))
    frac = rb_counts.astype(np.float32) / float(total_links)
    is_hot = (frac > rb_hot_threshold)

    # 2) 功率-剩余时间耦合
    time_left_norm = np.clip(individual_time_limit / float(V2V_limit), 0.0, 1.0)
    urgent_mask = (time_left_norm <= urgency_threshold)
    high_power_mask = (actions_pw == 0)  # 0=23dB

    reward = base_reward_matrix.copy()

    # 全局 gini 惩罚
    if penalty_gini > 0:
        reward -= penalty_gini

    # 每链路的 hot-RB 惩罚与功率-时间耦合
    hot_link_mask = np.zeros(actions_rb.shape, dtype=bool)
    hot_link_mask[valid_rb] = is_hot[actions_rb[valid_rb]]
    reward[hot_link_mask] -= rb_softmask_alpha

    valid_urgent = valid_rb & urgent_mask
    valid_relaxed = valid_rb & (~urgent_mask)
    reward[valid_urgent & high_power_mask] += beta_urgency_pos
    reward[valid_urgent & (~high_power_mask)] -= 0.5 * beta_urgency_pos
    reward[valid_relaxed & high_power_mask] -= beta_urgency_neg
    reward[valid_relaxed & (~high_power_mask)] += 0.5 * beta_urgency_neg

    return reward
