"""单障碍物圆形 PLT-CBF。

PowerShell 安装依赖：
    python -m pip install numpy casadi
"""

import time

import casadi as ca
import numpy as np

#  实验参数
TLIM = 0.0         # 🔴允许在圆形安全区内停留的总时间 [s]；0 表示不允许进入，从0开始调试
V_REF = 0.3        # 期望巡航速度 [m/s]
A_MAX = 1.0        # 各轴名义加速度上限 [m/s²]
K1 = K2 = K3 = 1.0 # CBF1、CBF2、CBF3 的增益
EPS_B = 3e-3       # 靠近安全边界时启用 CBF3 的阈值 [m²]
USE_B3 = True      # 是否在安全边界附近启用 CBF3
USE_Z3 = True      # 相对速度为零时是否启用 CBF3
R_OFF = 0.0        # 距离终点小于该值时关闭 CBF [m]


def plt_control(
    state: np.ndarray,
    obstacle_pos: np.ndarray,
    obstacle_vel: np.ndarray,
    current_time: float,
    *,
    reset: bool = False,
    goal: np.ndarray | None = None,
    obstacle_radius: float | None = None,
    safety_margin: float | None = None,
) -> np.ndarray:
    """计算一步 PLT-CBF 安全加速度。

    参数：
        state: 无人机当前状态 [x, y, z, vx, vy, vz]。
        obstacle_pos: 圆形障碍物当前位置 [x, y, z]。
        obstacle_vel: 障碍物速度 [vx, vy, vz]。
        reset: 新实验首次调用设为 True，清空上一次实验的内部记忆。
        current_time: 当前时刻
        goal、obstacle_radius、safety_margin: 新实验首帧传入，后续自动保持。

    返回：
        u: 安全加速度 [ax, ay, az]；CBF/QP 只修改 x、y，az = 名义 az。
        调试信息可从 plt_control.last_info 读取。

    # 当前接口仅支持单障碍物。
    # 多障碍物场景应先构造所有障碍物的 CBF 约束，
    # 再将全部约束合并到同一个 QP 中统一求解。
    """
    try:
        state = np.asarray(state, dtype=float).reshape(6)
        obstacle_pos = np.asarray(obstacle_pos, dtype=float).reshape(3)
        obstacle_vel = np.asarray(obstacle_vel, dtype=float).reshape(3)
    except ValueError as exc:
        raise ValueError("state 必须为 6 维；obstacle_pos、obstacle_vel 必须为 3 维") from exc
    if not np.isfinite(current_time):
        raise ValueError("current_time 必须为有限数值")

    memory = {} if reset else getattr(plt_control, "_memory", {})
    last_time = memory.get("last_time")
    if last_time is not None and current_time < last_time:
        raise ValueError("current_time 不能小于上一次调用时刻")
    dt = 0.0 if last_time is None else current_time - last_time
    if goal is None:
        if "goal" not in memory:
            raise ValueError("新实验首帧必须传入 goal")
        goal = np.asarray(memory["goal"], dtype=float).reshape(3)
    else:
        goal = np.asarray(goal, dtype=float).reshape(3)
    if obstacle_radius is None:
        if "obstacle_radius" not in memory:
            raise ValueError("新实验首帧必须传入 obstacle_radius")
        obstacle_radius = float(memory["obstacle_radius"])
    else:
        obstacle_radius = float(obstacle_radius)
    if safety_margin is None:
        if "safety_margin" not in memory:
            raise ValueError("新实验首帧必须传入 safety_margin")
        safety_margin = float(memory["safety_margin"])
    else:
        safety_margin = float(safety_margin)
    v_ref, a_max = float(V_REF), float(A_MAX)
    k1, k2, k3, eps_b = float(K1), float(K2), float(K3), float(EPS_B)
    use_b3, use_z3, r_off = bool(USE_B3), bool(USE_Z3), float(R_OFF)
    if obstacle_radius < 0 or safety_margin < 0:
        raise ValueError("obstacle_radius 和 safety_margin 必须非负")
    if a_max <= 0:
        raise ValueError("A_MAX 必须为正")
    safety_radius = obstacle_radius + safety_margin
    t_lim = float(memory.get("t_lim", TLIM))
    if t_lim < 0:
        raise ValueError("文件开头的 TLIM 不能为负")

    # 名义控制：目标速度跟踪。
    delta = goal - state[:3]
    r_goal = np.linalg.norm(delta)
    v_des = np.zeros(3) if r_goal < 1e-5 else v_ref * np.tanh(0.7 * r_goal) * delta / r_goal
    u_ref = np.clip(5.0 * (v_des - state[3:]), -a_max, a_max)

    # 与 N1 一致：进入时刻不插值，但首个区内采样点先计入一个控制周期。
    inside_prev = bool(memory.get("inside_prev", False))
    entry_time = memory.get("entry_time")
    entry_dt = float(memory.get("entry_dt", 0.0))
    completed_inside_time = float(memory.get("completed_inside_time", 0.0))
    t_init = float(memory.get("t_init", t_lim))

    p_t = obstacle_pos[:2] - state[:2]
    v_t = obstacle_vel[:2] - state[3:5]
    c_val = p_t @ p_t - safety_radius**2
    dist = np.linalg.norm(p_t) - safety_radius
    inside = c_val <= 0.0
    entered, exited = not inside_prev and inside, inside_prev and not inside
    if inside:
        if entered or entry_time is None:
            entry_time, entry_dt = current_time, dt
        inside_time = entry_dt + current_time - entry_time
        t_rem = max(t_init - inside_time, 0.0)
        total_inside_time = completed_inside_time + inside_time
    else:
        if exited and entry_time is not None:
            completed_inside_time += entry_dt + last_time - entry_time
        entry_time, entry_dt, inside_time = None, 0.0, 0.0
        total_inside_time, t_rem = completed_inside_time, t_init
    if exited:
        t_init, t_rem = 0.0, 0.0

    # 构造当前激活的 CBF 约束：a_xy @ u_xy <= b。
    # 为保持与理论/仿真实现一致，这里按严格零相对速度判断。
    # 实机若速度测量存在噪声，可由实验人员将 0.0 替换为合适的小阈值，例如 0.01
    moving = np.linalg.norm(v_t) > 0.0
    outside = c_val > 0.0 or (abs(c_val) <= 1e-8 and v_t @ p_t >= 0.0)
    out0 = outside and t_init == 0.0 and moving
    outt = outside and t_init > 0.0 and moving
    inside_cbf = (c_val < 0.0 or (abs(c_val) <= 1e-8 and v_t @ p_t < 0.0)) and t_lim != 0.0 and moving
    use_boundary = use_b3 and c_val < eps_b and r_goal >= r_off and t_init == 0.0
    use_zero = use_z3 and not moving and r_goal >= r_off

    constraints: list[tuple[np.ndarray, float]] = []
    stage_on = np.zeros(3, dtype=bool)
    h_stage = np.full(3, np.nan)
    stage, h, lf, b = 0, np.nan, np.nan, np.nan
    a_last = np.zeros(3)

    if r_goal >= r_off:
        if inside_cbf:
            s = p_t + v_t * t_rem
            h = s @ s - safety_radius**2
            eta = 2.0 * t_rem * s
            a2, b, lf, stage = eta, k2 * h, 0.0, 2
        elif out0 or outt:
            t_use = t_init if out0 else t_init - dt
            v_norm = np.linalg.norm(v_t) + 1e-10
            q = max(p_t @ p_t - (safety_radius**2 - (v_t @ v_t) * t_use**2 / 4.0), 1e-6)
            q_sqrt = np.sqrt(q)
            h = p_t @ v_t + v_norm * q_sqrt
            lf = v_t @ v_t + v_norm * (p_t @ v_t) / q_sqrt
            psi = p_t + (q_sqrt / v_norm + v_norm * t_use**2 / (4.0 * q_sqrt)) * v_t
            a2 = psi
            b, stage = lf + k1 * h, 1
        else:
            a2 = None

        if a2 is not None and np.all(np.isfinite(a2)) and np.isfinite(b):
            constraints.append((a2, b))
            a_last, stage_on[stage - 1], h_stage[stage - 1] = np.r_[a2, 0.0], True, h

    if use_boundary or use_zero:
        h3 = p_t @ v_t
        lf3 = v_t @ v_t
        a3_2d = p_t
        b3 = lf3 + k3 * h3
        if np.all(np.isfinite(a3_2d)) and np.isfinite(b3):
            constraints.append((a3_2d, b3))
            h, lf, b, stage = h3, lf3, b3, 3
            a_last, stage_on[2], h_stage[2] = np.r_[a3_2d, 0.0], True, h3

    # QP 只优化 x、y；z 轴直接保持名义控制量。
    u = np.r_[np.clip(u_ref[:2], -a_max, a_max), u_ref[2]]
    exitflag, qp_ms = 0, np.nan
    if constraints:
        a_ineq = np.vstack([item[0] for item in constraints])
        b_ineq = np.array([item[1] for item in constraints])
        a_full = np.vstack((a_ineq, np.eye(2)))
        lower = np.r_[np.full(len(b_ineq), -np.inf), [-a_max, -a_max]]
        upper = np.r_[b_ineq, [a_max, a_max]]
        qp_start = time.perf_counter()
        cache = getattr(plt_control, "_qpoases_cache", {})
        key = a_full.shape[0]
        if key not in cache:
            cache[key] = ca.conic(
                f"plt_qp_{key}", "qpoases",
                {"h": ca.Sparsity.dense(2, 2), "a": ca.Sparsity.dense(key, 2)},
                {"printLevel": "none"},
            )
            plt_control._qpoases_cache = cache
        try:
            result = cache[key](
                h=ca.DM(2.0 * np.eye(2)), g=ca.DM(-2.0 * u_ref[:2]), a=ca.DM(a_full),
                lba=ca.DM(lower), uba=ca.DM(upper),
            )
            u_xy = np.asarray(result["x"], dtype=float).reshape(2)
            solved = bool(cache[key].stats()["success"]) and np.all(np.isfinite(u_xy))
        except RuntimeError:
            solved = False
        qp_ms = 1000.0 * (time.perf_counter() - qp_start)
        exitflag = 1 if solved else -1
        if solved:
            u[:2] = u_xy

    next_memory = {
        "goal": goal.copy(),
        "obstacle_radius": obstacle_radius,
        "safety_margin": safety_margin,
        "t_lim": t_lim,
        "last_time": current_time,
        "entry_time": entry_time,
        "entry_dt": entry_dt,
        "completed_inside_time": completed_inside_time,
        "inside_prev": inside,
        "inside_time": inside_time,
        "total_inside_time": total_inside_time,
        "t_init": t_init,
        "t_rem": t_rem,
    }
    plt_control._memory = next_memory
    plt_control.last_info = {
        "dist": dist,
        "inside": inside,
        "entered": entered,
        "exited": exited,
        "t_rem": t_rem,
        "stage": stage,
        "exitflag": exitflag,
        "qp_ms": qp_ms,
    }
    return u

# ============================ 实验调用参考 ============================
#
# 本文件只提供一步 PLT-CBF 控制器。
# 实验平台需要自行提供：
#   1) 无人机状态 state = [x, y, z, vx, vy, vz]
#   2) 障碍物位置 obstacle_pos = [x, y, z]
#   3) 障碍物速度 obstacle_vel = [vx, vy, vz]
#   4) 无人机当前单调时刻 current_time
#   5) 将返回的加速度指令 u = [ax, ay, az] 发送给底层飞控
#
# 首次调用时：
#   reset = True
#   传入 goal、obstacle_radius、safety_margin

# 后续调用时：
#   只传 state、obstacle_pos、obstacle_vel、current_time
#
# 注意：
#   - 当前版本为单障碍物圆形安全区接口。
#   - CBF/QP 只修正 x、y 方向，z 方向保持名义控制器输出。
#   - 本文件不包含 mocap、ROS/PX4 通信、坐标系转换、起降和 failsafe。
#
#
# 
