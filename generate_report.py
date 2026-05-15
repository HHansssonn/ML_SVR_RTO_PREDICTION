import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulator_linear as sim_lin
import simulator_kernel as sim_ker
import joblib

OUT_DIR    = "plots/results"
SIM_TIME   = 300
CLIENTS    = [80, 100, 120, 150]

PROTOCOLS = {
    "CoAP":      "CoAP",
    "CoCoA":     "CoCoA",
    "CoCoA+":    "CoCoA+",
    "ML_Linear": "SVR (Linear)",
    "ML_Kernel": "SVR (Kernel)",
    "ML_BO":     "SVR (Linear)+BO",
}

COLORS = {
    "CoAP":      "#2166ac",
    "CoCoA":     "#d6604d",
    "CoCoA+":    "#8b4513",
    "ML_Linear": "#1a1a1a",
    "ML_Kernel": "#1b7837",
    "ML_BO":     "#e67e00",
}

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading models...")
model_lin  = joblib.load("models/linear_rto_model.joblib")
model_ker  = joblib.load("models/rto_model.joblib")
drop_model = joblib.load("models/drop_model.joblib")


def run(s_ker, s_lin, proto):
    if proto == "ML_Linear":
        return sim_lin.simulate_protocol(s_lin, "ML",
                                         model=model_lin,
                                         drop_model=drop_model)
    elif proto == "ML_Kernel":
        return sim_ker.simulate_protocol(s_ker, "ML",
                                         model=model_ker,
                                         drop_model=drop_model)
    elif proto == "ML_BO":
        return sim_lin.simulate_protocol(s_lin, "ML_BO",
                                         model=model_lin,
                                         drop_model=drop_model)
    else:
        return sim_ker.simulate_protocol(s_ker, proto)


print("Running load sweep...")
rows = []
for c in CLIENTS:
    print(f"  clients={c}")
    s_ker = sim_ker.Scenario(clients=c, packet_size=512, pdr=0.80,
                              jitter_ms=180.0, base_rtt_ms=200,
                              sim_time_s=SIM_TIME)
    s_lin = sim_lin.Scenario(clients=c, packet_size=512, pdr=0.80,
                              jitter_ms=180.0, base_rtt_ms=200,
                              sim_time_s=SIM_TIME)
    for key, label in PROTOCOLS.items():
        r = run(s_ker, s_lin, key)
        rows.append({
            "Clients":      c,
            "Protocol":     key,
            "Label":        label,
            "Goodput":      round(r.goodput_bytes_per_s, 1),
            "PDR":          round(r.pdr_percentage, 2),
            "Energy":       round(r.energy_mj, 2),
            "Retrans":      round(r.avg_retransmissions_per_transaction, 3),
            "Latency":      round(r.mean_latency_ms, 1),
            "MissRate":     round(r.deadline_miss_rate, 2),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "results_load_sweep.csv"), index=False)
print("Saved results_load_sweep.csv")


print("Running burst/continuous comparison...")
bc_rows = []
for scenario in ["Continuous", "Burst"]:
    is_burst = scenario == "Burst"
    s_ker = sim_ker.Scenario(
        clients=100, packet_size=512, pdr=0.60,
        jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_TIME,
        burst_error_prob      = 0.20 if is_burst else 0.01,
        burst_recovery_prob   = 0.10 if is_burst else 0.80,
        burst_loss_multiplier = 0.40 if is_burst else 1.0,
    )
    s_lin = sim_lin.Scenario(
        clients=100, packet_size=512, pdr=0.60,
        jitter_ms=180.0, base_rtt_ms=200, sim_time_s=SIM_TIME,
        burst_error_prob      = 0.20 if is_burst else 0.01,
        burst_recovery_prob   = 0.10 if is_burst else 0.80,
        burst_loss_multiplier = 0.40 if is_burst else 1.0,
    )
    for key, label in PROTOCOLS.items():
        r = run(s_ker, s_lin, key)
        bc_rows.append({
            "Scenario":  scenario,
            "Protocol":  key,
            "Label":     label,
            "Goodput":   round(r.goodput_bytes_per_s, 1),
            "PDR":       round(r.pdr_percentage, 2),
            "Energy":    round(r.energy_mj, 2),
            "Retrans":   round(r.avg_retransmissions_per_transaction, 3),
        })

df_bc = pd.DataFrame(bc_rows)
df_bc.to_csv(os.path.join(OUT_DIR, "results_burst_continuous.csv"), index=False)
print("Saved results_burst_continuous.csv")


def bar_chart(df, metric, ylabel, title, filename, higher_better=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(len(CLIENTS))
    n      = len(PROTOCOLS)
    width  = 0.13
    offset = -(n - 1) / 2 * width

    for i, (key, label) in enumerate(PROTOCOLS.items()):
        vals = [df[(df["Clients"] == c) & (df["Protocol"] == key)][metric].values[0]
                for c in CLIENTS]
        ax.bar(x + offset + i * width, vals, width,
               label=label, color=COLORS[key])

    ax.set_xlabel("Number of Clients")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(CLIENTS)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {filename}")


bar_chart(df, "Goodput", "Goodput (B/s)",
          "Goodput vs. Number of Clients", "plot_goodput.png")

bar_chart(df, "PDR", "PDR (%)",
          "Packet Delivery Ratio vs. Number of Clients", "plot_pdr.png")

bar_chart(df, "Energy", "Energy (mJ)",
          "Energy Consumption vs. Number of Clients", "plot_energy.png",
          higher_better=False)

bar_chart(df, "Retrans", "Avg Retransmissions / Tx",
          "Retransmissions vs. Number of Clients", "plot_retrans.png",
          higher_better=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, scenario in zip(axes, ["Continuous", "Burst"]):
    sub = df_bc[df_bc["Scenario"] == scenario]
    labels = [PROTOCOLS[k] for k in PROTOCOLS]
    vals   = [sub[sub["Protocol"] == k]["PDR"].values[0] for k in PROTOCOLS]
    colors = [COLORS[k] for k in PROTOCOLS]
    ax.bar(labels, vals, color=colors)
    ax.set_title(f"{scenario} Scenario — PDR (%)")
    ax.set_ylabel("PDR (%)")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(min(vals) - 5, 100)

plt.suptitle("PDR: Continuous vs Burst (100 clients, PDR=60%)")
plt.tight_layout()
path = os.path.join(OUT_DIR, "plot_burst_vs_continuous.png")
plt.savefig(path, dpi=150)
plt.close()
print("Saved plot_burst_vs_continuous.png")

print("\n=== PDR (%) by Client Count ===")
pdr_pivot = df.pivot(index="Clients", columns="Label", values="PDR")
print(pdr_pivot.to_string())

print("\n=== Goodput (B/s) by Client Count ===")
gp_pivot = df.pivot(index="Clients", columns="Label", values="Goodput")
print(gp_pivot.to_string())

pdr_pivot.to_csv(os.path.join(OUT_DIR, "table_pdr.csv"))
gp_pivot.to_csv(os.path.join(OUT_DIR, "table_goodput.csv"))

print(f"\nDone. All outputs saved to {OUT_DIR}/")
