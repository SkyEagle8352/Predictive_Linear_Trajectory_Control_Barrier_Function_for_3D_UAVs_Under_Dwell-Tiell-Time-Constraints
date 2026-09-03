"""恒定高度无人机的二维 PLT-CBF 仿真。

运行方式：
    python N1_PLT_ellipse.py

脚本会将 simulation_data.npz 和六张 PNG 图保存到 python_output/ 文件夹。
"""

# ============================ PowerShell 使用说明 ============================
# 1. 安装 Python 依赖（在 PowerShell 中执行）：
#    python -m pip install --upgrade pip
#    python -m pip install numpy scipy matplotlib osqp casadi
#
# 2. 选择 QP 求解器：
#    QP_SOLVER = 1  -> OSQP
#    QP_SOLVER = 2  -> CasADi + qpOASES
#
# 3. 运行仿真（在当前文件夹的 PowerShell 中执行）：
#    python N1_PLT_ellipse.py

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import osqp
import casadi as ca
from matplotlib.patches import Circle, Ellipse
from scipy import sparse


# ============================ 参数 ============================
# 禁飞区（TLZ）
P_OBS = np.array([-0.5, 0.1, 0.0]) # 中心位置 [m]
R_OBS = 0.2                          # 实体圆形障碍物半径 [m]
D_SAFE = 0.5                         # 仅作为默认安全裕度参考 [m]
SAFE_AXES = np.array([R_OBS + D_SAFE, R_OBS + D_SAFE])  # 椭圆安全区半轴 [a, b] [m]
V_OBS = np.array([0.0, 0.0, 0.0])    # 障碍物速度 [m/s]
ANG_OBS = 0.0                         # 椭圆长轴相对全局 x 轴旋转角 [rad]
H_OBS = 10.0                          # 圆柱高度 [m]
TLIM = 0.0                            # 🔴【实验时需要修改】允许在禁飞区内停留的时间 [s]

# 任务与控制器参数
P0 = np.array([-2.5, 0.0, 1.0])      # 起点 [m]
PG = np.array([2.5, 0.0, 1.0])       # 目标点 [m]
V0 = np.zeros(3)                      # 初始速度 [m/s]
V_REF = 0.3                           # 期望巡航速度 [m/s]
A_MAX = 1.0                           # 各轴最大加速度 [m/s^2]
FS = 20.0                             # 控制频率 [Hz]
T_MAX = 120.0                         # 最大仿真时间 [s]
R_GOAL = 0.05                         # 到达目标的距离容差 [m]
R_OFF = 0.0                           # 在该目标距离内关闭 CBF [m]

# h1、h2、h3 的 PLT-CBF 增益
K1 = 1.0
K2 = 1.0
K3 = 1.0
EPS_B = 3*1e-3                        # h3 边界激活阈值 [m^2]
USE_B3 = True                         # 是否在边界附近启用 h3
USE_Z3 = True                         # 相对速度为零时是否启用 h3

# QP 求解器：1 = OSQP，2 = CasADi + qpOASES 建议选2比较快
QP_SOLVER = 2

OUT_DIR = Path(__file__).with_name("python_output")


def make_ellipse_transform(axes: np.ndarray, angle: float) -> tuple[np.ndarray, float]:
    """将旋转椭圆映射到半径 a 的等效圆，与 MATLAB make_cylinder_transform 一致。"""
    axes = np.asarray(axes, dtype=float).reshape(2)
    a_axis, b_axis = float(axes[0]), float(axes[1])
    if not (a_axis > 0 and b_axis > 0 and a_axis >= b_axis):
        raise ValueError("SAFE_AXES 必须满足 a >= b > 0，其中 [a, b] 为长、短半轴")
    if b_axis < R_OBS:
        raise ValueError("SAFE_AXES 的短半轴 b 必须不小于实体障碍物半径 R_OBS")

    ct, st = np.cos(angle), np.sin(angle)
    rot = np.array([[ct, -st], [st, ct]])
    scale = np.diag([1.0, a_axis / b_axis])
    transform = scale @ rot.T
    return transform, a_axis


def guidance(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    delta = PG - pos
    r = np.linalg.norm(delta)
    v_des = np.zeros(3) if r < 1e-5 else V_REF * np.tanh(0.7 * r) * delta / r
    u_ref = np.clip(5.0 * (v_des - vel), -A_MAX, A_MAX)
    return u_ref, r, v_des


_casadi_solvers: dict[tuple[int, int], ca.Function] = {}


def solve_qp_osqp(u_ref: np.ndarray, a_ineq: np.ndarray, b_ineq: np.ndarray) -> tuple[np.ndarray, int]:
    """使用 OSQP 求解带 CBF 约束和输入边界的二次规划。"""
    n_vars = u_ref.size
    p = sparse.eye(n_vars, format="csc") * 2.0
    q = -2.0 * u_ref
    a = sparse.vstack((sparse.csc_matrix(a_ineq), sparse.eye(n_vars, format="csc")), format="csc")
    lower = np.r_[np.full(len(b_ineq), -np.inf), np.full(n_vars, -A_MAX)]
    upper = np.r_[b_ineq, np.full(n_vars, A_MAX)]

    solver = osqp.OSQP()
    solver.setup(p, q, a, lower, upper, verbose=False, polishing=True,
                 eps_abs=1e-8, eps_rel=1e-8)
    result = solver.solve()
    solved = result.info.status.lower().startswith("solved")
    return (result.x, 1) if solved and np.all(np.isfinite(result.x)) else (np.zeros(n_vars), -1)


def solve_qp_casadi(u_ref: np.ndarray, a_ineq: np.ndarray, b_ineq: np.ndarray) -> tuple[np.ndarray, int]:
    """使用 CasADi + qpOASES 求解相同的二次规划。"""
    n_vars = u_ref.size
    a_full = np.vstack((a_ineq, np.eye(n_vars)))
    lower = np.r_[np.full(len(b_ineq), -np.inf), np.full(n_vars, -A_MAX)]
    upper = np.r_[b_ineq, np.full(n_vars, A_MAX)]
    n_rows = a_full.shape[0]
    solver_key = (n_vars, n_rows)

    if solver_key not in _casadi_solvers:
        _casadi_solvers[solver_key] = ca.conic(
            f"plt_qp_{n_vars}_{n_rows}", "qpoases",
            {"h": ca.Sparsity.dense(n_vars, n_vars), "a": ca.Sparsity.dense(n_rows, n_vars)},
            {"printLevel": "none"},
        )

    try:
        result = _casadi_solvers[solver_key](
            h=ca.DM(2.0 * np.eye(n_vars)), g=ca.DM(-2.0 * u_ref), a=ca.DM(a_full),
            lba=ca.DM(lower), uba=ca.DM(upper),
        )
        u = np.asarray(result["x"], dtype=float).reshape(n_vars)
        solved = bool(_casadi_solvers[solver_key].stats()["success"])
        return (u, 1) if solved and np.all(np.isfinite(u)) else (np.zeros(n_vars), -1)
    except RuntimeError:
        return np.zeros(n_vars), -1


def cbf_control(
    obs: dict,
    u_ref: np.ndarray,
    p_t: np.ndarray,
    v_t: np.ndarray,
    c_val: float,
    t_init: float,
    t_rem: float,
    dt: float,
    r_goal: float,
) -> dict:
    """生成当前激活的 PLT-CBF 约束并求解安全二次规划。"""
    moving = np.linalg.norm(v_t) > 0.0
    is_zero = not moving
    outside = c_val > 0.0 or (abs(c_val) <= 1e-8 and v_t @ p_t >= 0.0)
    out0 = outside and t_init == 0.0 and moving
    outt = outside and t_init > 0.0 and moving
    inside_cbf = (c_val < 0.0 or (abs(c_val) <= 1e-8 and v_t @ p_t < 0.0)) and TLIM != 0.0 and moving
    use_boundary = USE_B3 and c_val < EPS_B and r_goal >= R_OFF and t_init == 0.0
    use_zero = USE_Z3 and is_zero and r_goal >= R_OFF

    constraints: list[tuple[np.ndarray, float]] = []
    result = {
        "u": np.r_[np.clip(u_ref[:2], -A_MAX, A_MAX), u_ref[2]],
        "exitflag": 0,
        "qp_ms": np.nan,
        "zero_rel": is_zero,
        "stage": 0,
        "stage_on": np.zeros(3, dtype=bool),
        "h_stage": np.full(3, np.nan),
        "h": np.nan,
        "a": np.zeros(3),
        "lf": np.nan,
        "b": np.nan,
    }

    last: tuple[np.ndarray, float, float, float, int] | None = None
    if r_goal >= R_OFF:
        if inside_cbf:
            s = p_t + v_t * t_rem
            h = s @ s - obs["r"] ** 2
            a2, b, lf = obs["T"].T @ (2.0 * t_rem * s), K2 * h, 0.0
            last = (a2, b, h, lf, 2)
        elif out0 or outt:
            t_lim = t_init if out0 else t_init - dt
            v_norm = np.linalg.norm(v_t) + 1e-10
            q = max(p_t @ p_t - (obs["r"] ** 2 - (v_t @ v_t) * t_lim**2 / 4.0), 1e-6)
            q_sqrt = np.sqrt(q)
            h = p_t @ v_t + v_norm * q_sqrt
            lf = v_t @ v_t + v_norm * (p_t @ v_t) / q_sqrt
            psi = p_t + (q_sqrt / v_norm + v_norm * t_lim**2 / (4.0 * q_sqrt)) * v_t
            a2, b = obs["T"].T @ psi, lf + K1 * h
            last = (a2, b, h, lf, 1)

    if last is not None and np.all(np.isfinite(last[0])) and np.isfinite(last[1]):
        a2, b, h, lf, stage = last
        a3 = np.r_[a2, 0.0]
        constraints.append((a3, b))
        result.update({"h": h, "a": a3, "lf": lf, "b": b, "stage": stage})
        result["h_stage"][stage - 1] = h
        result["stage_on"][stage - 1] = True

    if use_boundary or use_zero:
        h = p_t @ v_t
        lf = v_t @ v_t
        a2, b = obs["T"].T @ p_t, lf + K3 * h
        if np.all(np.isfinite(a2)) and np.isfinite(b):
            a3 = np.r_[a2, 0.0]
            constraints.append((a3, b))
            result.update({"h": h, "a": a3, "lf": lf, "b": b, "stage": 3})
            result["h_stage"][2] = h
            result["stage_on"][2] = True

    if constraints:
        a_ineq = np.vstack([item[0] for item in constraints])
        b_ineq = np.array([item[1] for item in constraints])
        qp_start = time.perf_counter()
        # QP 目标函数和 CBF 约束只处理 x、y 轴，z 轴直接使用名义控制量。
        if QP_SOLVER == 1:
            u_xy, exitflag = solve_qp_osqp(u_ref[:2], a_ineq[:, :2], b_ineq)
        elif QP_SOLVER == 2:
            u_xy, exitflag = solve_qp_casadi(u_ref[:2], a_ineq[:, :2], b_ineq)
        else:
            raise ValueError("QP_SOLVER 必须是 1（OSQP）或 2（qpOASES）")
        result["qp_ms"] = 1000.0 * (time.perf_counter() - qp_start)
        result["exitflag"] = exitflag
        if exitflag > 0:
            result["u"] = np.r_[u_xy, u_ref[2]]

    return result


def dynamics(state: np.ndarray, u: np.ndarray, obs: dict, dt: float) -> tuple[np.ndarray, dict]:
    """使用四阶 Runge-Kutta 方法更新无人机和运动障碍物的状态。"""
    k1 = np.r_[state[3:], u]
    k2 = np.r_[state[3:] + 0.5 * dt * k1[3:], u]
    k3 = np.r_[state[3:] + 0.5 * dt * k2[3:], u]
    k4 = np.r_[state[3:] + dt * k3[3:], u]
    next_state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_obs = obs.copy()
    next_obs["p"] = obs["p"] + obs["v"] * dt
    return next_state, next_obs


def simulate() -> dict:
    dt = 1.0 / FS
    max_steps = int(np.ceil(T_MAX / dt))
    transform, r_eq = make_ellipse_transform(SAFE_AXES, ANG_OBS)
    obs = {"p": P_OBS.copy(), "v": V_OBS.copy(), "r": r_eq, "T": transform,
           "axes": SAFE_AXES.copy(), "ang": ANG_OBS, "height": H_OBS}

    state = np.r_[P0, V0]
    inside_prev = False
    inside_time = total_inside_time = 0.0
    t_init = TLIM
    t_rem = TLIM
    entries: list[list[float]] = []
    exits: list[list[float]] = []
    entry_crossings: list[float] = []
    exit_crossings: list[float] = []
    dist_prev = np.linalg.norm(transform @ (obs["p"][:2] - state[:2])) - obs["r"]

    pos = np.zeros((max_steps + 1, 3))
    vel = np.zeros((max_steps + 1, 3))
    obs_pos = np.zeros((max_steps + 1, 3))
    u_ref_log = np.zeros((max_steps, 3))
    u_log = np.zeros((max_steps, 3))
    v_des_log = np.zeros((max_steps, 3))
    r_goal_log = np.zeros(max_steps)
    exitflag = np.zeros(max_steps, dtype=int)
    solve_ms = np.zeros(max_steps)
    qp_ms = np.full(max_steps, np.nan)
    dist_log = np.zeros(max_steps + 1)
    region_log = np.zeros(max_steps, dtype=int)
    t_rem_log = np.zeros(max_steps)
    inside_log = np.zeros(max_steps)
    total_inside_log = np.zeros(max_steps)
    stage_last = np.zeros(max_steps, dtype=int)
    stage_on = np.zeros((3, max_steps), dtype=bool)
    zero_rel_log = np.zeros(max_steps, dtype=bool)
    h_log = np.full(max_steps, np.nan)
    h_stage = np.full((3, max_steps), np.nan)
    h_dot_log = np.full(max_steps, np.nan)
    ratio_log = np.full(max_steps, np.nan)
    a_last = np.zeros((max_steps, 3))
    lf_last = np.full(max_steps, np.nan)
    b_last = np.full(max_steps, np.nan)

    steps = max_steps
    for i in range(max_steps):
        pos[i], vel[i], obs_pos[i] = state[:3], state[3:], obs["p"]
        u_ref, r_goal_now, v_des = guidance(state[:3], state[3:])
        u_ref_log[i], v_des_log[i], r_goal_log[i] = u_ref, v_des, r_goal_now
        t0 = time.perf_counter()

        p_rel = obs["p"][:2] - state[:2]
        v_rel = obs["v"][:2] - state[3:5]
        p_t = obs["T"] @ p_rel
        v_t = obs["T"] @ v_rel
        c_val = p_t @ p_t - obs["r"] ** 2
        dist = np.linalg.norm(p_t) - obs["r"]
        dist_log[i] = dist
        region_log[i] = 1 if c_val > 1e-8 else (-1 if c_val < -1e-8 else 0)

        inside = c_val <= 0.0
        if not inside_prev and inside:
            entries.append([*state[:3], i * dt])
        if inside_prev and not inside:
            exits.append([*state[:3], i * dt])
        if inside:
            inside_time = dt if not inside_prev else inside_time + dt
            t_rem = max(t_init - inside_time, 0.0)
            total_inside_time += dt
        else:
            inside_time, t_rem = 0.0, t_init
        if inside_prev and not inside:
            t_init, t_rem = 0.0, 0.0
        inside_prev = inside

        if dist <= 0 < dist_prev:
            ratio = (-dist_prev) / (dist - dist_prev + 1e-10)
            entry_crossings.append(((i - 1) + np.clip(ratio, 0.0, 1.0)) * dt)
        elif dist > 0 >= dist_prev:
            ratio = (-dist_prev) / (dist - dist_prev + 1e-10)
            exit_crossings.append(((i - 1) + np.clip(ratio, 0.0, 1.0)) * dt)
        dist_prev = dist
        inside_log[i], total_inside_log[i], t_rem_log[i] = inside_time, total_inside_time, t_rem

        cbf = cbf_control(obs, u_ref, p_t, v_t, c_val,
                          t_init, t_rem, dt, r_goal_now)
        u = cbf["u"]
        zero_rel_log[i] = cbf["zero_rel"]
        exitflag[i], qp_ms[i] = cbf["exitflag"], cbf["qp_ms"]
        stage_last[i] = cbf["stage"]
        stage_on[:, i] = cbf["stage_on"]
        h_log[i], a_last[i], lf_last[i], b_last[i] = cbf["h"], cbf["a"], cbf["lf"], cbf["b"]
        h_stage[:, i] = cbf["h_stage"]
        solve_ms[i] = 1000.0 * (time.perf_counter() - t0)
        u_log[i] = u

        if stage_last[i] != 0:
            h_dot_log[i] = a_last[i] @ (-u) + lf_last[i]
            ratio_log[i] = 0.0 if abs(b_last[i]) < 1e-10 else (a_last[i] @ u) / b_last[i]

        state, obs = dynamics(state, u, obs, dt)
        if np.linalg.norm(state[:3] - PG) <= R_GOAL:
            steps = i + 1
            break

    pos[steps], vel[steps], obs_pos[steps] = state[:3], state[3:], obs["p"]
    dist_log[steps] = np.linalg.norm(obs["T"] @ (obs["p"][:2] - state[:2])) - obs["r"]
    t_ctrl = np.arange(steps) * dt
    return {
        "dt": dt, "steps": steps, "state": state, "obs": obs, "r_safe": r_eq, "safe_axes": SAFE_AXES.copy(),
        "pos": pos[: steps + 1], "vel": vel[: steps + 1], "obs_pos": obs_pos[: steps + 1],
        "t": t_ctrl, "u_ref": u_ref_log[:steps], "u": u_log[:steps], "v_des": v_des_log[:steps],
        "r_goal": r_goal_log[:steps], "exitflag": exitflag[:steps], "solve_ms": solve_ms[:steps],
        "qp_ms": qp_ms[:steps],
        "dist": dist_log[: steps + 1], "region": region_log[:steps], "t_rem": t_rem_log[:steps],
        "inside": inside_log[:steps], "inside_total": total_inside_log[:steps],
        "stage_last": stage_last[:steps], "stage_on": stage_on[:, :steps],
        "zero_rel": zero_rel_log[:steps], "h": h_log[:steps], "h_stage": h_stage[:, :steps],
        "h_dot": h_dot_log[:steps],
        "ratio": ratio_log[:steps], "entries": np.asarray(entries).reshape(-1, 4),
        "exits": np.asarray(exits).reshape(-1, 4),
        "entry_crossings": np.asarray(entry_crossings), "exit_crossings": np.asarray(exit_crossings),
        "inside_total_final": total_inside_time,
    }


def style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def save_results(data: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(OUT_DIR / "simulation_data.npz", **data)
    config = {name: value for name, value in globals().items() if name in {
        "P_OBS", "R_OBS", "D_SAFE", "SAFE_AXES", "V_OBS", "ANG_OBS", "H_OBS", "P0", "PG", "V0",
        "V_REF", "A_MAX", "FS", "T_MAX", "R_GOAL", "R_OFF", "TLIM", "K1", "K2", "K3",
        "EPS_B", "USE_B3", "USE_Z3", "QP_SOLVER",
    }}
    config = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in config.items()}
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def plot_all(data: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    t, pos, vel, obs = data["t"], data["pos"], data["vel"], data["obs"]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(pos[:, 0], pos[:, 1], lw=2, label="PLT-CBF")
    ax.plot(P0[0], P0[1], "*", ms=15, mfc="#ffd60a", mec="k", mew=0.8, label="Start")
    ax.plot(PG[0], PG[1], "p", ms=11, color="#e63946", mec="k", label="Goal")
    center = obs["p"][:2]
    safe_tlz = Ellipse(center, 2 * obs["axes"][0], 2 * obs["axes"][1],
                       angle=np.degrees(obs["ang"]), facecolor="#4c78a8", alpha=0.15,
                       edgecolor="#4c78a8", lw=2, label="Safety TLZ")
    real_tlz = Circle(center, R_OBS, facecolor="#f4a261", alpha=0.35,
                      edgecolor="#e76f51", lw=2, label="Physical TLZ")
    ax.add_patch(safe_tlz)
    ax.add_patch(real_tlz)
    ax.annotate(f"Safety axes: a={obs['axes'][0]:.2f} m, b={obs['axes'][1]:.2f} m",
                xy=center + [obs["axes"][0], 0], xytext=(10, 10), textcoords="offset points",
                color="#2563a8", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#2563a8"})
    ax.annotate(f"Physical radius: {R_OBS:.2f} m", xy=center + [R_OBS, 0],
                xytext=(10, -16), textcoords="offset points", color="#c85a3a", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#c85a3a"})
    if data["entries"].size:
        ax.plot(data["entries"][:, 0], data["entries"][:, 1], "o", ms=8, color="#c51b8a", label="Entry")
        for idx, point in enumerate(data["entries"], start=1):
            ax.annotate(f"In {idx}: {point[3]:.2f} s", xy=point[:2], xytext=(6, 8),
                        textcoords="offset points", color="#9c1776", fontsize=9)
    if data["exits"].size:
        ax.plot(data["exits"][:, 0], data["exits"][:, 1], "ks", ms=7, label="Exit")
        for idx, point in enumerate(data["exits"], start=1):
            ax.annotate(f"Out {idx}: {point[3]:.2f} s", xy=point[:2], xytext=(6, -14),
                        textcoords="offset points", color="black", fontsize=9)
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="2D trajectory", aspect="equal")
    style(ax); ax.legend(loc="best")
    fig.savefig(OUT_DIR / "01_trajectory.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        ax.plot(t, data["u_ref"][:, j], "k--", lw=1.1, label=r"$u_{\mathrm{nom}}$")
        ax.plot(t, data["u"][:, j], "b", lw=1.5, label=r"$u_{\mathrm{safety}}$")
        bound = A_MAX
        ax.axhline(bound, color="#8ecae6", ls="-.", label="Boundary" if j == 0 else "_nolegend_")
        ax.axhline(-bound, color="#8ecae6", ls="-.", label="_nolegend_")
        ax.set_ylabel(rf"$a_{{{'xyz'[j]}}}$ (m/s²)"); style(ax)
    axes[0].legend(); axes[-1].set_xlabel("Time (s)")
    fig.savefig(OUT_DIR / "02_controls.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    t_state = np.arange(len(vel)) * data["dt"]
    for j, ax in enumerate(axes[:3]):
        ax.plot(t_state, vel[:, j], "b", lw=1.5)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel(f"v{'xyz'[j]} (m/s)"); style(ax)
    axes[3].plot(t_state, np.linalg.norm(vel, axis=1), "b", lw=1.5, label="Speed")
    axes[3].axhline(V_REF, color="r", ls="--", label="Cruise speed")
    axes[3].set(xlabel="Time (s)", ylabel="Speed (m/s)"); style(axes[3]); axes[3].legend()
    fig.savefig(OUT_DIR / "03_speed.png", dpi=200); plt.close(fig)

    colors = ["#0057d9", "#d62828", "#1b9e3e"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    active = data["stage_on"]
    none = ~np.any(active, axis=0)
    ax.step(t, np.where(none, 0.0, np.nan), where="post", color="0.5", label="None")
    for stage, color in enumerate(colors, start=1):
        ax.step(t, np.where(active[stage - 1], stage, np.nan), where="post", color=color, lw=2, label=f"CBF{stage}")
    ax.plot(t[data["zero_rel"]], np.full(np.count_nonzero(data["zero_rel"]), 3), "o", color=colors[2], ms=5, label="CBF3: v_rel=0")
    ax.set(yticks=[0, 1, 2, 3], yticklabels=["None", "CBF1", "CBF2", "CBF3"], ylim=(-0.2, 3.2),
           xlabel="Time (s)", title="CBF activation")
    style(ax); ax.legend(ncol=3, loc="best")
    fig.savefig(OUT_DIR / "04_cbf_activation.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    flag = data["exitflag"]
    ax.step(t, flag, where="post", color="k", label="exitflag")
    ax.plot(t[flag > 0], flag[flag > 0], "go", ms=4, label="Solved")
    ax.plot(t[flag < 0], flag[flag < 0], "rx", ms=6, label="Failed")
    ax.plot(t[flag == 0], flag[flag == 0], "ks", ms=3, label="QP not called")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Time (s)", ylabel="QP exitflag", title="QP status")
    style(ax); ax.legend(loc="best")
    fig.savefig(OUT_DIR / "05_qp_status.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for stage, color in enumerate(colors, start=1):
        values = data["h_stage"][stage - 1]
        ax.plot(t, values, color=color, lw=1.5, label=f"h{stage}")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Time (s)", ylabel="h", title="CBF h values")
    style(ax); ax.legend(loc="best")
    fig.savefig(OUT_DIR / "06_h_values.png", dpi=200); plt.close(fig)


def print_summary(data: dict) -> None:
    positive = np.count_nonzero(data["exitflag"] > 0)
    negative = np.count_nonzero(data["exitflag"] < 0)
    print("Simulation finished")
    print(f"Steps: {data['steps']}, time: {data['steps'] * data['dt']:.2f} s")
    print(f"Final position: {data['state'][:3]}")
    print(f"Goal error: {np.linalg.norm(data['state'][:3] - PG):.4f} m")
    print(f"Minimum signed distance: {np.min(data['dist']):.4f} m")
    print(f"QP: {positive} solved, {negative} failed, {data['steps'] - positive - negative} not called")
    qp_ms = data["qp_ms"]
    qp_ms = qp_ms[np.isfinite(qp_ms)]
    solver_label = {1: "OSQP", 2: "qpOASES"}.get(QP_SOLVER, "Unknown solver")
    if len(qp_ms):
        print(f"{solver_label} solve time ({len(qp_ms)} calls): mean={np.mean(qp_ms):.3f} ms, "
              f"p95={np.percentile(qp_ms, 95):.3f} ms, max={np.max(qp_ms):.3f} ms")
    else:
        print(f"{solver_label} solve time: None (QP not called)")
    entries = data["entries"]
    exits = data["exits"]
    entry_crossings = data["entry_crossings"]
    exit_crossings = data["exit_crossings"]
    n_passes = min(len(entries), len(exits))
    if n_passes == 0:
        if len(entries) == 0:
            print(f"Dwell: tin=None, tout=None, Tdwell=None, Tlim={TLIM:.3f} s, satisfied=True (no traversal)")
        else:
            tin = entries[0, 3]
            tin_interp = entry_crossings[0] if len(entry_crossings) else None
            tin_text = f"{tin:.2f} s ({tin_interp:.4f} s)" if tin_interp is not None else f"{tin:.2f} s (None)"
            print(f"Dwell 1: tin={tin_text}, tout=None, Tdwell=None, Tlim={TLIM:.3f} s, satisfied=None (not exited)")
    else:
        for k in range(n_passes):
            tin = entries[k, 3]
            tout = exits[k, 3]
            tin_interp = entry_crossings[k] if k < len(entry_crossings) else None
            tout_interp = exit_crossings[k] if k < len(exit_crossings) else None
            tdwell = tout - tin
            tdwell_interp = tout_interp - tin_interp if tin_interp is not None and tout_interp is not None else None
            check_time = tdwell_interp if tdwell_interp is not None else tdwell
            satisfied = check_time <= TLIM + 1e-10
            tin_text = f"{tin:.2f} s ({tin_interp:.4f} s)" if tin_interp is not None else f"{tin:.2f} s (None)"
            tout_text = f"{tout:.2f} s ({tout_interp:.4f} s)" if tout_interp is not None else f"{tout:.2f} s (None)"
            dwell_text = f"{tdwell:.2f} s ({tdwell_interp:.4f} s)" if tdwell_interp is not None else f"{tdwell:.2f} s (None)"
            print(f"Dwell {k + 1}: tin={tin_text}, tout={tout_text}, "
                  f"Tdwell={dwell_text}, Tlim={TLIM:.3f} s, satisfied={satisfied}")
        if len(entries) > n_passes:
            tin = entries[n_passes, 3]
            tin_interp = entry_crossings[n_passes] if n_passes < len(entry_crossings) else None
            tin_text = f"{tin:.2f} s ({tin_interp:.4f} s)" if tin_interp is not None else f"{tin:.2f} s (None)"
            print(f"Dwell {n_passes + 1}: tin={tin_text}, tout=None, "
                  f"Tdwell=None, Tlim={TLIM:.3f} s, satisfied=None (not exited)")
    print(f"Figures and data saved to: {OUT_DIR}")


if __name__ == "__main__":
    # 每次仿真先清空旧输出，避免遗留图片或数据混入本次结果。
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    result = simulate()
    save_results(result)
    plot_all(result)
    print_summary(result)
