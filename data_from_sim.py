import warnings
warnings.filterwarnings("ignore")

import os
import math
import random
import numpy as np
import pandas as pd

import simulator_kernel as sim_ker

PENDING_TIMEOUT_S = 5.0


def collect_from_scenario(scenario: sim_ker.Scenario,
                           seed: int = 1,
                           collect_protocol: str = "CoAP") -> list:
    s   = scenario
    rng = random.Random(seed)
    ch  = sim_ker.ChannelModel(s, rng)

    current_time = 0.0
    features     = []

    srtt      = [s.base_rtt_ms       for _ in range(s.clients)]
    rttvar    = [s.base_rtt_ms / 2.0 for _ in range(s.clients)]
    hist      = [[1] * 8             for _ in range(s.clients)]
    rto_state = [rng.uniform(2000.0, 3000.0) for _ in range(s.clients)]
    srtt_weak = [s.base_rtt_ms       for _ in range(s.clients)]
    rv_weak   = [s.base_rtt_ms / 4.0 for _ in range(s.clients)]

    next_event    = [rng.uniform(0.0, s.mean_interarrival_ms / 1000.0)
                     for _ in range(s.clients)]
    last_app_time = list(next_event)
    last_tx_time  = [-1.0] * s.clients
    pending       = [False] * s.clients
    tx_start_time = [0.0]   * s.clients
    attempt_no    = [0]     * s.clients
    pending_srtt  = [0.0]   * s.clients
    pending_rttv  = [0.0]   * s.clients
    pending_sr    = [1.0]   * s.clients
    pending_iat   = [0.0]   * s.clients

    prev_tx_end   = [-1.0] * s.clients

    channel_busy_until = 0.0
    airtime_s          = sim_ker._packet_airtime_s(s.packet_size)
    hidden_coll_prob   = sim_ker._hidden_collision_probability(s)

    def success_rate(idx):
        h = hist[idx][-10:]
        return sum(h) / len(h) if h else 1.0

    def inter_arrival(idx, now):
        if prev_tx_end[idx] < 0:
            return float(s.mean_interarrival_ms)
        gap = (now - prev_tx_end[idx]) * 1000.0
        return float(min(max(0.0, gap), 35000.0))

    def schedule_next(idx, not_before):
        gap = rng.expovariate(1.0 / s.mean_interarrival_ms) / 1000.0
        t   = max(not_before, last_app_time[idx]) + gap
        last_app_time[idx] = t
        next_event[idx]    = min(t, s.sim_time_s)

    def compute_rto(idx, attempt):
        if collect_protocol == "CoCoA":
            vbf = 3.0 if rto_state[idx] < 1000 else (
                  1.5 if rto_state[idx] > 3000 else 2.0)
            return max(sim_ker.STD_RTO_MIN_MS,
                       min(sim_ker.STD_RTO_MAX_MS,
                           rto_state[idx] * (vbf ** attempt)))
        else:
            return max(sim_ker.STD_RTO_MIN_MS,
                       min(sim_ker.STD_RTO_MAX_MS,
                           rto_state[idx] * (2.0 ** attempt)))

    def update_srtt(idx, rtt_ms, attempt):
        alpha, beta = 0.125, 0.25
        if collect_protocol == "CoCoA" and attempt > 0:
            srtt_weak[idx] = (1-alpha)*srtt_weak[idx] + alpha*rtt_ms
            rv_weak[idx]   = 0.75*rv_weak[idx] + 0.25*abs(srtt_weak[idx]-rtt_ms)
            rto_state[idx] = 0.5*(srtt[idx]+4*rttvar[idx]) + \
                             0.5*(srtt_weak[idx]+4*rv_weak[idx])
        else:
            srtt[idx]      = (1-alpha)*srtt[idx]   + alpha*rtt_ms
            rttvar[idx]    = (1-beta) *rttvar[idx] + beta*abs(srtt[idx]-rtt_ms)
            rto_state[idx] = srtt[idx] + 4*rttvar[idx]
        rto_state[idx] = max(sim_ker.STD_RTO_MIN_MS,
                             min(sim_ker.STD_RTO_MAX_MS, rto_state[idx]))

    while current_time < s.sim_time_s:
        next_t = min(next_event)
        if next_t >= s.sim_time_s:
            break
        current_time = next_t
        ready = [i for i, t in enumerate(next_event)
                 if t <= current_time + 1e-9]
        ch.step()
        p_base = ch.current_success_probability()

        for idx in ready:
            if current_time >= s.sim_time_s:
                next_event[idx] = s.sim_time_s
                continue

            if not pending[idx]:
                pending[idx]       = True
                tx_start_time[idx] = current_time
                attempt_no[idx]    = 0
                pending_srtt[idx]  = srtt[idx]
                pending_rttv[idx]  = rttvar[idx]
                pending_sr[idx]    = success_rate(idx)
                pending_iat[idx]   = inter_arrival(idx, current_time)
                last_tx_time[idx]  = current_time

            attempt = attempt_no[idx]

            if attempt > s.max_retx:
                features.append({
                    'srtt_ms':             pending_srtt[idx],
                    'rttvar_ms':           pending_rttv[idx],
                    'attempt':             attempt,
                    'success_rate_window': pending_sr[idx],
                    'inter_arrival_ms':    pending_iat[idx],
                    'target_rto_ms':       float(np.clip(
                                               pending_srtt[idx] + 4 * pending_rttv[idx],
                                               100.0,
                                               min(60000.0, 2000.0 * (2.0 ** attempt)))),
                    'drop_label':          1,
                    'is_timeout':          1,
                })
                hist[idx] = (hist[idx] + [0])[-10:]
                pending[idx]     = False
                prev_tx_end[idx] = current_time
                schedule_next(idx, current_time)
                continue

            local_rto  = compute_rto(idx, attempt)
            client_time = current_time

            if client_time < channel_busy_until:
                client_time = channel_busy_until + rng.uniform(
                    sim_ker.CCA_BACKOFF_MIN_MS,
                    sim_ker.CCA_BACKOFF_MAX_MS) / 1000.0

            if client_time >= s.sim_time_s:
                pending[idx]    = False
                next_event[idx] = s.sim_time_s
                continue

            p_success = p_base
            if rng.random() < hidden_coll_prob:
                p_success *= sim_ker.HIDDEN_COLLISION_PENALTY

            channel_busy_until = client_time + airtime_s
            rtt_sample = max(5.0, rng.gauss(s.base_rtt_ms, s.jitter_ms))
            ack_ok     = rtt_sample <= local_rto

            if rng.random() < p_success and ack_ok:
                new_srtt    = 0.875 * pending_srtt[idx] + 0.125 * rtt_sample
                new_rttvar  = 0.75  * pending_rttv[idx] + \
                              0.25  * abs(new_srtt - rtt_sample)
                optimal_rto = new_srtt + 4.0 * new_rttvar
                attempt_cap = min(60000.0, 2000.0 * (2.0 ** attempt))
                optimal_rto = float(np.clip(optimal_rto, 100.0, attempt_cap))

                features.append({
                    'srtt_ms':             pending_srtt[idx],
                    'rttvar_ms':           pending_rttv[idx],
                    'attempt':             attempt,
                    'success_rate_window': pending_sr[idx],
                    'inter_arrival_ms':    pending_iat[idx],
                    'target_rto_ms':       optimal_rto,
                    'drop_label':          0,
                    'is_timeout':          0,
                })
                hist[idx] = (hist[idx] + [1])[-10:]
                update_srtt(idx, rtt_sample, attempt)
                pending[idx]      = False
                prev_tx_end[idx]  = current_time
                schedule_next(idx, client_time + rtt_sample / 1000.0)
            else:
                attempt_no[idx] += 1
                next_event[idx]  = min(client_time + local_rto / 1000.0,
                                       s.sim_time_s)

    return features


def generate_dataset() -> pd.DataFrame:
    scenarios = []

    for clients in [20, 50, 80, 100, 120, 150]:
        for pdr in [0.50, 0.60, 0.70, 0.80, 0.90]:
            for jitter in [50.0, 120.0, 180.0, 300.0, 500.0]:
                for burst in [False, True]:
                    scenarios.append(dict(
                        clients=clients, pdr=pdr,
                        jitter_ms=jitter, base_rtt_ms=200.0,
                        packet_size=512, sim_time_s=120.0,
                        burst_error_prob      = 0.20 if burst else 0.0,
                        burst_recovery_prob   = 0.10 if burst else 1.0,
                        burst_loss_multiplier = 0.40 if burst else 1.0,
                    ))

    for base_rtt in [500.0, 1000.0, 2000.0, 5000.0, 8000.0]:
        for jitter_frac in [0.3, 0.5, 0.8]:
            scenarios.append(dict(
                clients=80, pdr=0.65,
                jitter_ms=base_rtt * jitter_frac,
                base_rtt_ms=base_rtt,
                packet_size=512, sim_time_s=120.0,
                burst_error_prob=0.0,
                burst_recovery_prob=1.0,
                burst_loss_multiplier=1.0,
            ))

    for pdr in [0.30, 0.40]:
        for clients in [100, 150]:
            scenarios.append(dict(
                clients=clients, pdr=pdr,
                jitter_ms=200.0, base_rtt_ms=400.0,
                packet_size=512, sim_time_s=120.0,
                burst_error_prob=0.30,
                burst_recovery_prob=0.05,
                burst_loss_multiplier=0.30,
            ))

    for ps in [128, 512, 1024]:
        scenarios.append(dict(
            clients=100, pdr=0.65,
            jitter_ms=150.0, base_rtt_ms=200.0,
            packet_size=ps, sim_time_s=120.0,
            burst_error_prob=0.0,
            burst_recovery_prob=1.0,
            burst_loss_multiplier=1.0,
        ))

    for clients in [120, 150]:
        for pdr in [0.40, 0.50]:
            scenarios.append(dict(
                clients=clients, pdr=pdr,
                jitter_ms=200.0, base_rtt_ms=200.0,
                packet_size=512, sim_time_s=120.0,
                max_retx=6,
                burst_error_prob=0.25,
                burst_recovery_prob=0.08,
                burst_loss_multiplier=0.30,
            ))

    # Low PDR + high clients for timeout samples needed by the drop classifier
    for pdr in [0.30, 0.40, 0.50]:
        for clients in [120, 150]:
            for max_retx in [3, 4]:
                scenarios.append(dict(
                    clients=clients, pdr=pdr,
                    jitter_ms=300.0, base_rtt_ms=400.0,
                    packet_size=512, sim_time_s=120.0,
                    max_retx=max_retx,
                    burst_error_prob=0.35,
                    burst_recovery_prob=0.05,
                    burst_loss_multiplier=0.25,
                ))

    total = len(scenarios)
    print(f"Collecting data from {total} scenarios...")
    all_features = []

    for i, sc_kwargs in enumerate(scenarios):
        s = sim_ker.Scenario(**sc_kwargs)
        for seed in [1, 42]:
            for proto in ["CoAP", "CoCoA"]:
                feats = collect_from_scenario(s, seed=seed,
                                              collect_protocol=proto)
                all_features.extend(feats)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} scenarios done  "
                  f"({len(all_features)} samples so far)")

    return pd.DataFrame(all_features)


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()

    timeout_mask = df['is_timeout'].astype(int) == 1
    timeout_rows = df[timeout_mask].copy()
    success_rows = df[~timeout_mask].copy()

    success_rows = success_rows[
        (success_rows['target_rto_ms'] >= 100.0) &
        (success_rows['target_rto_ms'] <= 60000.0)
    ]
    if len(success_rows) > 100:
        q_low, q_hi = success_rows['target_rto_ms'].quantile([0.01, 0.99])
        success_rows = success_rows[
            (success_rows['target_rto_ms'] >= q_low) &
            (success_rows['target_rto_ms'] <= q_hi)
        ]

    max_timeout = len(success_rows) * 3
    if len(timeout_rows) > max_timeout:
        timeout_rows = timeout_rows.sample(n=max_timeout, random_state=42)
        print(f"Class balance cap: timeout -> {len(timeout_rows)} "
              f"(3x success={len(success_rows)})")

    combined = pd.concat([success_rows, timeout_rows], ignore_index=True)
    print(f"\nOutlier filter: {before} -> {len(combined)} samples "
          f"({len(success_rows)} success, {len(timeout_rows)} timeout)")
    return combined


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_dataset()
    df = postprocess(df)

    ML_FEATURES = ["srtt_ms", "rttvar_ms", "attempt",
                   "success_rate_window", "inter_arrival_ms"]

    print("\n--- Feature std ---")
    print(df[ML_FEATURES].std())
    print("\n--- Drop-label distribution ---")
    print(df['drop_label'].value_counts(normalize=True))
    print("\n--- target_rto_ms stats ---")
    print(df[df.is_timeout == 0]['target_rto_ms'].describe())
    print("\n--- Pearson correlation with target_rto_ms ---")
    print(df[ML_FEATURES + ['target_rto_ms']].corr()['target_rto_ms'])

    df.to_csv("data/training_data.csv", index=False)
    print(f"\nGenerated {len(df)} samples -> data/training_data.csv")
    print("Now run: python train_kernel.py && python train_linear.py && python train_drop.py")
