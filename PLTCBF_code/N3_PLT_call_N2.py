"""使用 N2 圆形 PLT-CBF 控制器复现 N1 场景。

运行：python N3_PLT_call_N2.py
输出：python_output2/simulation_data.npz 和六张与 N1 同类的图。
"""

from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import N2_PLT_ellipse_experiment as controller


# ============================ N1 场景参数 ============================
P_OBS = np.array([-0.5, 0.1, 0.0])
R_OBS, D_SAFE = 0.2, 0.5
V_OBS = np.zeros(3)
P0, PG, V0 = np.array([-2.5, 0.0, 1.0]), np.array([2.5, 0.0, 1.0]), np.zeros(3)
V_REF, A_MAX, FS, T_MAX, R_GOAL = 0.3, 1.0, 20.0, 120.0, 0.05
TLIM, K1, K2, K3, EPS_B, USE_B3, USE_Z3, R_OFF = controller.TLIM, 1.0, 1.0, 1.0, 3e-3, True, True, 0.0
OUT_DIR = Path(__file__).with_name("python_output2")


def rk4_step(state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    k1 = np.r_[state[3:], u]
    k2 = np.r_[state[3:] + 0.5 * dt * k1[3:], u]
    k3 = np.r_[state[3:] + 0.5 * dt * k2[3:], u]
    k4 = np.r_[state[3:] + dt * k3[3:], u]
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def main() -> None:
    # 每次仿真先清空旧输出，避免遗留图片或数据混入本次结果。
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(exist_ok=True)
    dt, max_steps = 1.0 / FS, int(np.ceil(T_MAX * FS))
    controller.V_REF, controller.A_MAX = V_REF, A_MAX
    controller.K1, controller.K2, controller.K3 = K1, K2, K3
    controller.EPS_B, controller.USE_B3, controller.USE_Z3, controller.R_OFF = EPS_B, USE_B3, USE_Z3, R_OFF

    state = np.r_[P0, V0]
    pos, vel = np.zeros((max_steps + 1, 3)), np.zeros((max_steps + 1, 3))
    u_ref, u = np.zeros((max_steps, 3)), np.zeros((max_steps, 3))
    v_des = np.zeros((max_steps, 3))
    dist, trem, call_ms = np.zeros(max_steps), np.zeros(max_steps), np.zeros(max_steps)
    stage, exitflag = np.zeros(max_steps, dtype=int), np.zeros(max_steps, dtype=int)
    stage_on = np.zeros((3, max_steps), dtype=bool)
    h_stage = np.full((3, max_steps), np.nan)
    zero_rel = np.zeros(max_steps, dtype=bool)
    entries, exits, entry_crossings, exit_crossings = [], [], [], []
    dist_prev = np.linalg.norm(P_OBS[:2] - P0[:2]) - (R_OBS + D_SAFE)
    plot_inside_prev, plot_entry_time, plot_entry_dt, plot_t_init, plot_last_time = False, None, 0.0, TLIM, None
    steps = max_steps

    for i in range(max_steps):
        now = i * dt
        pos[i], vel[i] = state[:3], state[3:]
        delta = PG - state[:3]
        r_goal = np.linalg.norm(delta)
        v_des[i] = np.zeros(3) if r_goal < 1e-5 else V_REF * np.tanh(0.7 * r_goal) * delta / r_goal
        u_ref[i] = np.clip(5.0 * (v_des[i] - state[3:]), -A_MAX, A_MAX)

        # N3 自行记录与 N1 同样的 CBF 激活和 h 值；不改变 N2 控制器。
        plot_dt = 0.0 if plot_last_time is None else now - plot_last_time
        p, v = P_OBS[:2] - state[:2], V_OBS[:2] - state[3:5]
        c_val = p @ p - (R_OBS + D_SAFE) ** 2
        dist_now = np.linalg.norm(p) - (R_OBS + D_SAFE)
        inside_now = c_val <= 0.0
        entered_now, exited_now = not plot_inside_prev and inside_now, plot_inside_prev and not inside_now
        if inside_now:
            if entered_now or plot_entry_time is None:
                plot_entry_time, plot_entry_dt = now, plot_dt
            trem_now = max(plot_t_init - plot_entry_dt - (now - plot_entry_time), 0.0)
        else:
            trem_now = plot_t_init
        if exited_now:
            plot_t_init, trem_now = 0.0, 0.0

        moving = np.linalg.norm(v) > 0.0
        outside = c_val > 0.0 or (abs(c_val) <= 1e-8 and v @ p >= 0.0)
        out0, outt = outside and plot_t_init == 0.0 and moving, outside and plot_t_init > 0.0 and moving
        inside_cbf = (c_val < 0.0 or (abs(c_val) <= 1e-8 and v @ p < 0.0)) and TLIM != 0.0 and moving
        if r_goal >= R_OFF:
            if inside_cbf:
                s = p + v * trem_now
                h_stage[1, i], stage_on[1, i] = s @ s - (R_OBS + D_SAFE) ** 2, True
            elif out0 or outt:
                t_use = plot_t_init if out0 else plot_t_init - plot_dt
                v_norm = np.linalg.norm(v) + 1e-10
                q_sqrt = np.sqrt(max(p @ p - ((R_OBS + D_SAFE) ** 2 - (v @ v) * t_use**2 / 4.0), 1e-6))
                h_stage[0, i], stage_on[0, i] = p @ v + v_norm * q_sqrt, True
        zero_rel[i] = not moving
        if (USE_B3 and c_val < EPS_B and r_goal >= R_OFF and plot_t_init == 0.0) or (USE_Z3 and zero_rel[i] and r_goal >= R_OFF):
            h_stage[2, i], stage_on[2, i] = p @ v, True
        if dist_now <= 0.0 < dist_prev:
            ratio = (-dist_prev) / (dist_now - dist_prev + 1e-10)
            entry_crossings.append(((i - 1) + np.clip(ratio, 0.0, 1.0)) * dt)
        elif dist_now > 0.0 >= dist_prev:
            ratio = (-dist_prev) / (dist_now - dist_prev + 1e-10)
            exit_crossings.append(((i - 1) + np.clip(ratio, 0.0, 1.0)) * dt)
        dist_prev = dist_now
        plot_inside_prev, plot_last_time = inside_now, now

        start = time.perf_counter()
        if i == 0:
            u[i] = controller.plt_control(
                state, P_OBS, V_OBS, now, reset=True,
                goal=PG, obstacle_radius=R_OBS, safety_margin=D_SAFE,
            )
        else:
            u[i] = controller.plt_control(state, P_OBS, V_OBS, now)
        call_ms[i] = 1000.0 * (time.perf_counter() - start)
        info = controller.plt_control.last_info
        dist[i], trem[i], stage[i], exitflag[i] = info["dist"], info["t_rem"], info["stage"], info["exitflag"]
        if info["entered"]:
            entries.append([*state[:3], now])
        if info["exited"]:
            exits.append([*state[:3], now])

        state = rk4_step(state, u[i], dt)
        if np.linalg.norm(state[:3] - PG) <= R_GOAL:
            steps = i + 1
            break

    pos[steps], vel[steps] = state[:3], state[3:]
    t = np.arange(steps) * dt
    valid_call_ms = call_ms[:steps]
    np.savez_compressed(
        OUT_DIR / "simulation_data.npz", t=t, dt=dt, pos=pos[:steps + 1], vel=vel[:steps + 1],
        u_ref=u_ref[:steps], u=u[:steps], v_des=v_des[:steps], dist=dist[:steps], t_rem=trem[:steps],
        stage=stage[:steps], stage_on=stage_on[:, :steps], h_stage=h_stage[:, :steps],
        zero_rel=zero_rel[:steps], exitflag=exitflag[:steps], n2_call_ms=valid_call_ms,
        entries=np.asarray(entries).reshape(-1, 4), exits=np.asarray(exits).reshape(-1, 4),
        entry_crossings=np.asarray(entry_crossings), exit_crossings=np.asarray(exit_crossings),
    )

    safe_radius = R_OBS + D_SAFE
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(pos[:steps + 1, 0], pos[:steps + 1, 1], lw=2, color="#1261a0", label="PLT-CBF")
    ax.plot(*P0[:2], "*", ms=15, color="#ffd60a", mec="k", label="Start")
    ax.plot(*PG[:2], "p", ms=11, color="#e63946", mec="k", label="Goal")
    ax.add_patch(Circle(P_OBS[:2], safe_radius, fc="#4c78a8", ec="#4c78a8", alpha=0.16, lw=2, label="Safety TLZ"))
    ax.add_patch(Circle(P_OBS[:2], R_OBS, fc="#f4a261", ec="#e76f51", alpha=0.45, lw=2, label="Physical TLZ"))
    ax.annotate(f"Safety radius: {safe_radius:.2f} m", xy=P_OBS[:2] + [safe_radius, 0], xytext=(10, 10),
                textcoords="offset points", color="#2563a8", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#2563a8"})
    ax.annotate(f"Physical radius: {R_OBS:.2f} m", xy=P_OBS[:2] + [R_OBS, 0], xytext=(10, -16),
                textcoords="offset points", color="#c85a3a", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#c85a3a"})
    if entries:
        points = np.asarray(entries)
        ax.plot(points[:, 0], points[:, 1], "o", ms=8, color="#c51b8a", label="Entry")
        for index, point in enumerate(points, start=1):
            ax.annotate(f"In {index}: {point[3]:.2f} s", xy=point[:2], xytext=(6, 8),
                        textcoords="offset points", color="#9c1776", fontsize=9)
    if exits:
        points = np.asarray(exits)
        ax.plot(points[:, 0], points[:, 1], "ks", ms=7, label="Exit")
        for index, point in enumerate(points, start=1):
            ax.annotate(f"Out {index}: {point[3]:.2f} s", xy=point[:2], xytext=(6, -14),
                        textcoords="offset points", color="black", fontsize=9)
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="2D trajectory", aspect="equal")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(OUT_DIR / "01_trajectory.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        ax.plot(t, u_ref[:steps, j], "k--", lw=1.1, label="Reference" if j == 0 else "_nolegend_")
        ax.plot(t, u[:steps, j], color="#1261a0", lw=1.5, label="Applied" if j == 0 else "_nolegend_")
        ax.axhline(A_MAX, color="#8ecae6", ls="-.", label="Saturation" if j == 0 else "_nolegend_")
        ax.axhline(-A_MAX, color="#8ecae6", ls="-."); ax.set_ylabel(f"a{'xyz'[j]} (m/s²)"); ax.grid(True, alpha=0.3)
    axes[0].legend(); axes[-1].set_xlabel("Time (s)")
    fig.savefig(OUT_DIR / "02_controls.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes[:3]):
        ax.plot(np.arange(steps + 1) * dt, vel[:steps + 1, j], color="#1261a0", lw=1.5)
        ax.axhline(0, color="k", ls="--", lw=0.8); ax.set_ylabel(f"v{'xyz'[j]} (m/s)"); ax.grid(True, alpha=0.3)
    axes[3].plot(np.arange(steps + 1) * dt, np.linalg.norm(vel[:steps + 1], axis=1), color="#1261a0", label="Speed")
    axes[3].axhline(V_REF, color="r", ls="--", label="Cruise speed")
    axes[3].set(xlabel="Time (s)", ylabel="Speed (m/s)"); axes[3].grid(True, alpha=0.3); axes[3].legend()
    fig.savefig(OUT_DIR / "03_speed.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    active = stage_on[:, :steps]
    ax.step(t, np.where(~np.any(active, axis=0), 0.0, np.nan), where="post", color="0.5", label="None")
    for index, color in enumerate(("#0057d9", "#d62828", "#1b9e3e"), start=1):
        ax.step(t, np.where(active[index - 1], index, np.nan), where="post", color=color, lw=2, label=f"CBF{index}")
    ax.plot(t[zero_rel[:steps]], np.full(np.count_nonzero(zero_rel[:steps]), 3), "o", color="#1b9e3e", ms=4, label="CBF3: v_rel=0")
    ax.set(yticks=[0, 1, 2, 3], yticklabels=["None", "CBF1", "CBF2", "CBF3"], ylim=(-0.2, 3.2), xlabel="Time (s)", title="CBF activation")
    ax.grid(True, alpha=0.3); ax.legend(ncol=3, loc="best")
    fig.savefig(OUT_DIR / "04_cbf_activation.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    flags = exitflag[:steps]
    ax.step(t, flags, where="post", color="k", label="exitflag")
    ax.plot(t[flags > 0], flags[flags > 0], "go", ms=4, label="Solved")
    ax.plot(t[flags < 0], flags[flags < 0], "rx", ms=6, label="Failed")
    ax.plot(t[flags == 0], flags[flags == 0], "ks", ms=3, label="QP not called")
    ax.axhline(0, color="k", ls="--", lw=0.8); ax.set(xlabel="Time (s)", ylabel="QP exitflag", title="QP status")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(OUT_DIR / "05_qp_status.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, color in enumerate(("#0057d9", "#d62828", "#1b9e3e"), start=1):
        ax.plot(t, h_stage[index - 1, :steps], color=color, lw=1.5, label=f"h{index}")
    ax.axhline(0, color="k", ls="--", lw=0.8); ax.set(xlabel="Time (s)", ylabel="h", title="CBF h values")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.savefig(OUT_DIR / "06_h_values.png", dpi=200); plt.close(fig)

    positive, negative = np.count_nonzero(exitflag[:steps] > 0), np.count_nonzero(exitflag[:steps] < 0)
    print("N3 simulation finished (every step calls N2 plt_control)")
    print(f"Steps: {steps}, simulation time: {steps * dt:.2f} s")
    print(f"Final position: {state[:3]}")
    print(f"Goal error: {np.linalg.norm(state[:3] - PG):.4f} m")
    print(f"Minimum signed distance: {np.min(dist[:steps]):.4f} m")
    print(f"QP: {positive} solved, {negative} failed, {steps - positive - negative} not called")
    print(f"N2 calls: {steps}, mean={np.mean(valid_call_ms):.3f} ms, p95={np.percentile(valid_call_ms, 95):.3f} ms, max={np.max(valid_call_ms):.3f} ms")
    print(f"Entries: {len(entries)}, exits: {len(exits)}, TLIM={TLIM:.3f} s")
    for index, (entry, exit_) in enumerate(zip(entries, exits), start=1):
        tin, tout = entry[3], exit_[3]
        tin_interp = entry_crossings[index - 1] if index <= len(entry_crossings) else None
        tout_interp = exit_crossings[index - 1] if index <= len(exit_crossings) else None
        dwell = tout - tin
        dwell_interp = tout_interp - tin_interp if tin_interp is not None and tout_interp is not None else None
        check_time = dwell_interp if dwell_interp is not None else dwell
        tin_text = f"{tin:.3f} s ({tin_interp:.4f} s)" if tin_interp is not None else f"{tin:.3f} s"
        tout_text = f"{tout:.3f} s ({tout_interp:.4f} s)" if tout_interp is not None else f"{tout:.3f} s"
        dwell_text = f"{dwell:.3f} s ({dwell_interp:.4f} s)" if dwell_interp is not None else f"{dwell:.3f} s"
        print(f"Dwell {index}: tin={tin_text}, tout={tout_text}, Tdwell={dwell_text}, "
              f"TLIM={TLIM:.3f} s, satisfied={check_time <= TLIM + 1e-10}")
    if len(entries) > len(exits):
        print(f"Dwell {len(exits) + 1}: tin={entries[-1][3]:.3f} s, tout=None, Tdwell=None")
    print(f"Figures and data saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
