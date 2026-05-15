import os, warnings, time
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import simulator_linear as sim_lin
import simulator_kernel as sim_ker

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": "--",
    "grid.alpha": 0.45, "grid.color": "#cccccc",
    "legend.framealpha": 0.9, "legend.fontsize": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "axes.facecolor": "#f9f9f9",
    "figure.facecolor": "white",
    "lines.linewidth": 2.0, "lines.markersize": 6,
})

PROTOCOLS = ["CoAP", "CoCoA", "CoCoA+", "ML_Linear", "ML_Kernel", "ML_Linear_BO"]
DISPLAY   = {"CoAP":"CoAP","CoCoA":"CoCoA","CoCoA+":"CoCoA+",
             "ML_Linear":"SVR (Linear)","ML_Kernel":"SVR (Kernel)",
             "ML_Linear_BO":"SVR (Linear)+BO"}
COLORS    = {"CoAP":"#2166ac","CoCoA":"#d6604d","CoCoA+":"#8b4513",
             "ML_Linear":"#1a1a1a","ML_Kernel":"#1b7837",
             "ML_Linear_BO":"#e67e00"}
MARKERS   = {"CoAP":"o","CoCoA":"s","CoCoA+":"D","ML_Linear":"^","ML_Kernel":"v",
             "ML_Linear_BO":"X"}
LINES     = {"CoAP":"-","CoCoA":"--","CoCoA+":"-.","ML_Linear":":","ML_Kernel":"-",
             "ML_Linear_BO":":"}

SAVE_DIR = "plots/extended_report"


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def _run(s_ker, s_lin, proto, model_lin, model_ker, drop_model):
    t0 = time.time()
    if proto == "ML_Linear":
        r = sim_lin.simulate_protocol(s_lin, "ML", model=model_lin, drop_model=drop_model)
    elif proto == "ML_Kernel":
        r = sim_ker.simulate_protocol(s_ker, "ML", model=model_ker, drop_model=drop_model)
    elif proto == "ML_Linear_BO":
        r = sim_lin.simulate_protocol(s_lin, "ML_BO", model=model_lin, drop_model=drop_model)
    else:
        r = sim_ker.simulate_protocol(s_ker, proto)
    print(f"    {DISPLAY[proto]:18s} done ({time.time()-t0:.1f}s) "
          f"PDR={r.pdr_percentage:.1f}%  GP={r.goodput_bytes_per_s:.0f} B/s")
    return r


def _bar_with_pct(df_c, x_col, y_col, client_counts, title, ylabel,
                  higher_better, path):
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = [COLORS[p] for p in PROTOCOLS]
    sns.barplot(x=x_col, y=y_col, hue="Protocol", data=df_c,
                hue_order=[DISPLAY[p] for p in PROTOCOLS],
                palette=palette, ax=ax, width=0.75,
                edgecolor="white", linewidth=0.5)

    coap_map = (df_c[df_c["Raw"]=="CoAP"]
                .set_index(x_col)[y_col].to_dict())

    for container, proto in zip(ax.containers, PROTOCOLS):
        for i, bar in enumerate(container):
            val  = bar.get_height()
            if val <= 0:
                continue
            c    = client_counts[i]
            base = coap_map.get(c, None)
            if base is None or base == 0:
                continue
            imp  = (val - base) / abs(base) * 100
            if not higher_better:
                imp = -imp
            xp = bar.get_x() + bar.get_width() / 2
            yp = val + df_c[y_col].max() * 0.015
            ax.text(xp, yp, f"{imp:+.1f}%", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                    color=COLORS[proto], rotation=90)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Number of Clients")
    ax.set_ylabel(ylabel)
    ax.legend(title="Protocol", bbox_to_anchor=(1.01,1), loc="upper left")
    _save(fig, path)


def run_comprehensive_evaluation():
    os.makedirs(SAVE_DIR, exist_ok=True)

    lin_path  = "models/linear_rto_model.joblib"
    ker_path  = "models/rto_model.joblib"
    drop_path = "models/drop_model.joblib"

    if not os.path.exists(lin_path) or not os.path.exists(ker_path):
        print("ERROR: models missing — run training scripts first.")
        return

    model_lin  = joblib.load(lin_path)
    model_ker  = joblib.load(ker_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None

    def R(s_ker, s_lin, proto):
        return _run(s_ker, s_lin, proto, model_lin, model_ker, drop_model)

    client_counts = [80, 100, 120, 150]
    SIM_T      = 60
    SIM_T_MAIN = 300

    print(f"\nLoad sweep ({len(client_counts)} client counts × {len(PROTOCOLS)} protocols)...")
    rows = []
    for c in client_counts:
        print(f"  clients={c}")
        s_ker = sim_ker.Scenario(clients=c, packet_size=512, pdr=0.80,
                                  jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_T_MAIN)
        s_lin = sim_lin.Scenario(clients=c, packet_size=512, pdr=0.80,
                                  jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_T_MAIN)
        for proto in PROTOCOLS:
            res = R(s_ker, s_lin, proto)
            rows.append({
                "Clients":    c,
                "Protocol":   DISPLAY[proto],
                "Raw":        proto,
                "Throughput": res.goodput_bytes_per_s,
                "Latency":    res.mean_latency_ms,
                "PDR":        res.pdr_percentage,
                "Energy":     res.energy_mj,
                "Retrans":    res.avg_retransmissions_per_transaction,
                "MissRate":   res.deadline_miss_rate,
                "RTO":        res.avg_rto_ms,
            })

    df_c = pd.DataFrame(rows)
    df_c.to_csv(f"{SAVE_DIR}/load_sweep.csv", index=False)

    _bar_with_pct(df_c, "Clients", "Throughput", client_counts,
                  "Goodput vs. Network Load", "Goodput (B/s)",
                  higher_better=True,
                  path=f"{SAVE_DIR}/1_throughput_load.png")

    _bar_with_pct(df_c, "Clients", "Latency", client_counts,
                  "Mean Latency vs. Network Load", "Mean Latency (ms)",
                  higher_better=False,
                  path=f"{SAVE_DIR}/2_latency_load.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for proto in PROTOCOLS:
        d = df_c[df_c["Raw"]==proto]
        ax.plot(d["Clients"], d["MissRate"], color=COLORS[proto],
                marker=MARKERS[proto], linestyle=LINES[proto],
                label=DISPLAY[proto])
    ax.set_title("Deadline Miss Rate vs. Network Load", pad=12)
    ax.set_xlabel("Number of Clients"); ax.set_ylabel("Miss Rate (%)")
    ax.legend(title="Protocol")
    _save(fig, f"{SAVE_DIR}/3_missrate_load.png")

    time_points = np.linspace(30, 300, 5)
    print(f"\nTime-series ({len(time_points)} points × 2 scenarios × {len(PROTOCOLS)} protocols)...")
    ts_rows = []
    for sc_name in ["Continuous", "Burst"]:
        is_b = sc_name == "Burst"
        print(f"  {sc_name} scenario")
        for t in time_points:
            s_ker = sim_ker.Scenario(clients=150, packet_size=512, pdr=0.80,
                sim_time_s=t, jitter_ms=180.0,
                burst_error_prob      = 0.20 if is_b else 0.01,
                burst_recovery_prob   = 0.10 if is_b else 0.80,
                burst_loss_multiplier = 0.40 if is_b else 1.0)
            s_lin = sim_lin.Scenario(clients=150, packet_size=512, pdr=0.80,
                sim_time_s=t, jitter_ms=180.0,
                burst_error_prob      = 0.20 if is_b else 0.01,
                burst_recovery_prob   = 0.10 if is_b else 0.80,
                burst_loss_multiplier = 0.40 if is_b else 1.0)
            for proto in PROTOCOLS:
                res = R(s_ker, s_lin, proto)
                ts_rows.append({
                    "Time": t, "Scenario": sc_name,
                    "Protocol": DISPLAY[proto], "Raw": proto,
                    "Throughput": res.goodput_bytes_per_s,
                    "Energy":     res.energy_mj,
                    "PDR":        res.pdr_percentage,
                    "RTO":        res.avg_rto_ms,
                    "Retrans":    res.avg_retransmissions_per_transaction,
                })

    df_ts = pd.DataFrame(ts_rows)

    ts_metrics = [
        ("Throughput", "Goodput (B/s)",               "4_time_tp"),
        ("Energy",     "Cumulative Energy (mJ)",       "5_time_energy"),
        ("PDR",        "Packet Delivery Ratio (%)",    "6_time_pdr"),
        ("RTO",        "Average RTO (ms)",             "7_time_rto"),
        ("Retrans",    "Avg Retransmissions / Tx",     "8_time_retrans"),
    ]
    for col, ylabel, fname in ts_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
        for i, sc in enumerate(["Continuous", "Burst"]):
            ax  = axes[i]
            sub = df_ts[df_ts["Scenario"]==sc]
            for proto in PROTOCOLS:
                d = sub[sub["Raw"]==proto]
                ax.plot(d["Time"], d[col], color=COLORS[proto],
                        marker=MARKERS[proto], linestyle=LINES[proto],
                        label=DISPLAY[proto], alpha=0.9)
            ax.set_title(f"({'ab'[i]}) {sc} Scenario — 150 clients", pad=10)
            ax.set_xlabel("Simulation Time (s)")
            ax.set_ylabel(ylabel)
            if i == 1:
                ax.legend(title="Protocol", loc="best")
        fig.suptitle(ylabel, fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save(fig, f"{SAVE_DIR}/{fname}.png")

    payloads = [128, 512, 1024]
    print(f"\nPayload sweep ({len(payloads)} sizes)...")
    ps_rows = []
    for ps in payloads:
        s_ker = sim_ker.Scenario(clients=100, packet_size=ps, pdr=0.80,
                                  jitter_ms=180.0, sim_time_s=SIM_T)
        s_lin = sim_lin.Scenario(clients=100, packet_size=ps, pdr=0.80,
                                  jitter_ms=180.0, sim_time_s=SIM_T)
        for proto in PROTOCOLS:
            res = R(s_ker, s_lin, proto)
            ps_rows.append({"Payload":ps,"Protocol":DISPLAY[proto],
                             "Raw":proto,"PDR":res.pdr_percentage})

    df_ps = pd.DataFrame(ps_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for proto in PROTOCOLS:
        d = df_ps[df_ps["Raw"]==proto]
        ax.plot(d["Payload"], d["PDR"], color=COLORS[proto],
                marker=MARKERS[proto], linestyle=LINES[proto],
                label=DISPLAY[proto])
    ax.set_title("PDR Sensitivity to Payload Size", pad=12)
    ax.set_xlabel("Payload Size (bytes)"); ax.set_ylabel("PDR (%)")
    ax.legend(title="Protocol")
    _save(fig, f"{SAVE_DIR}/9_payload_pdr.png")

    print("\nEnergy breakdown...")
    eb_rows = []
    for c in [80, 120, 150]:
        s_ker = sim_ker.Scenario(clients=c, packet_size=512, pdr=0.80,
                                  jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_T)
        s_lin = sim_lin.Scenario(clients=c, packet_size=512, pdr=0.80,
                                  jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_T)
        for proto in PROTOCOLS:
            res = R(s_ker, s_lin, proto)
            radio = (res.acks + res.retransmissions) * 0.1
            infer = max(0.0, res.energy_mj - radio)
            eb_rows.append({"Clients":c,"Protocol":DISPLAY[proto],"Raw":proto,
                             "Radio":radio,"Inference":infer,"Total":res.energy_mj})

    df_eb = pd.DataFrame(eb_rows)
    eb_clients = [80, 120, 150]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    x = np.arange(len(PROTOCOLS))
    for i, c in enumerate(eb_clients):
        ax  = axes[i]
        sub = df_eb[df_eb["Clients"]==c].set_index("Raw")
        rv  = [sub.loc[p, "Radio"]     if p in sub.index else 0.0 for p in PROTOCOLS]
        iv  = [sub.loc[p, "Inference"] if p in sub.index else 0.0 for p in PROTOCOLS]
        ax.bar(x, rv, color=[COLORS[p] for p in PROTOCOLS], alpha=0.85)
        ax.bar(x, iv, bottom=rv, color=[COLORS[p] for p in PROTOCOLS],
               alpha=0.4, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[p] for p in PROTOCOLS],
                            rotation=25, ha="right", fontsize=8)
        ax.set_title(f"({'abc'[i]}) {c} Clients", pad=8)
        ax.set_ylabel("Energy (mJ)" if i==0 else "")
    fig.suptitle("Energy Breakdown: Radio TX vs. ML Inference",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, f"{SAVE_DIR}/10_energy_breakdown.png")

    print("\nPDR heatmap...")
    pdr_vals  = [0.50, 0.60, 0.70]
    client_hm = [80, 150]
    hm_data   = {DISPLAY[p]: np.zeros((len(pdr_vals), len(client_hm)))
                 for p in PROTOCOLS}
    for pi, pdr in enumerate(pdr_vals):
        for ci, c in enumerate(client_hm):
            s_ker = sim_ker.Scenario(clients=c, packet_size=512, pdr=pdr,
                                      jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_T)
            s_lin = sim_lin.Scenario(clients=c, packet_size=512, pdr=pdr,
                                      jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_T)
            for proto in PROTOCOLS:
                res = R(s_ker, s_lin, proto)
                hm_data[DISPLAY[proto]][pi, ci] = res.pdr_percentage

    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(18, 4), sharey=True)
    for i, proto in enumerate(PROTOCOLS):
        ax  = axes[i]
        mat = hm_data[DISPLAY[proto]]
        im  = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                        vmin=50, vmax=100, origin="lower")
        ax.set_xticks(range(len(client_hm)))
        ax.set_xticklabels(client_hm, fontsize=9)
        ax.set_yticks(range(len(pdr_vals)))
        ax.set_yticklabels([f"{v:.0%}" for v in pdr_vals], fontsize=9)
        ax.set_xlabel("Clients", fontsize=9)
        if i == 0:
            ax.set_ylabel("Link PDR", fontsize=9)
        ax.set_title(DISPLAY[proto], fontsize=10,
                     fontweight="bold", color=COLORS[proto])
        for r in range(len(pdr_vals)):
            for c_i in range(len(client_hm)):
                ax.text(c_i, r, f"{mat[r,c_i]:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if mat[r,c_i]>65 else "white")
    fig.colorbar(im, ax=axes[-1], shrink=0.85, label="PDR (%)")
    fig.suptitle("PDR Heatmap: Link Quality × Network Load",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    _save(fig, f"{SAVE_DIR}/12_pdr_heatmap.png")

    print(f"\nDone. Plots saved to {SAVE_DIR}/")


if __name__ == "__main__":
    run_comprehensive_evaluation()
