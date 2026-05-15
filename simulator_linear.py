from __future__ import annotations
import math
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

ML_FEATURES = [
    "srtt_ms", "rttvar_ms", "attempt",
    "success_rate_window", "inter_arrival_ms",
]
DROP_FEATURES = [
    "srtt_ms", "rttvar_ms", "success_rate_window", "inter_arrival_ms",
]

# Linear SVR hardware profile
INFERENCE_DELAY_MS       = 12.0
INFERENCE_DELAY_JITTER   = 0.25
INFERENCE_ENERGY_MJ      = 0.08
DROP_INFERENCE_DELAY_MS  = 4.0
DROP_INFERENCE_ENERGY_MJ = 0.02

ACK_MISS_DURING_INFERENCE = 0.04
COLD_START_IDLE_MS        = 500.0
COLD_START_MULTIPLIER     = 1.6
INFERENCE_QUEUE_DEPTH     = 12
ENERGY_BUDGET_MJ          = 5000.0
THROTTLE_SLEEP_MS         = 200.0
TEMP_DRIFT_PACKETS        = 80
TEMP_DRIFT_MULT           = 1.05
TEMP_DRIFT_DURATION       = 25

# 802.15.4-like radio model
PHY_RATE_BPS            = 250_000.0
PHY_PREAMBLE_AND_MAC_MS = 8.0
ACK_AND_TURNAROUND_MS   = 5.0
CCA_BACKOFF_MIN_MS      = 2.0
CCA_BACKOFF_MAX_MS      = 12.0
BUSY_CHANNEL_PENALTY    = 0.96
HIDDEN_COLLISION_PENALTY= 0.72
MCU_BUSY_ACK_PENALTY    = 0.99
MAC_PROCESSING_MS       = 3.0

# RTO bounds derived from 802.15.4 + linear SVR inference timing
ML_RTO_MIN_MS  = 75.0
ML_RTO_MAX_MS  = 35000.0
STD_RTO_MIN_MS = 1000.0
STD_RTO_MAX_MS = 60000.0

PREDICTION_NOISE = 0.12
BAD_RTO_PROB     = 0.01
BAD_RTO_FACTOR   = 3.0
FALSE_DROP_PROB  = 0.01

WARM_INFERENCE_FRACTION = 0.05

DROP_MIN_ATTEMPT        = 3
DROP_DECISION_THRESHOLD = 0.88

_TRAINING_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "training_data.csv"
)

def _load_priors() -> tuple:
    try:
        import pandas as pd
        df = pd.read_csv(_TRAINING_DATA_PATH, usecols=["srtt_ms", "rttvar_ms"])
        df = df[(df["srtt_ms"] > 0) & (df["rttvar_ms"] > 0)]
        if len(df) < 10:
            raise ValueError("too few rows")
        sp = float(df["srtt_ms"].median())
        rp = float(df["rttvar_ms"].median())
        print(f"[simulator_linear] Loaded priors: srtt={sp:.1f} ms, "
              f"rttvar={rp:.1f} ms (n={len(df)})")
        return sp, rp
    except Exception as e:
        print(f"[simulator_linear] WARNING: could not load priors ({e}), "
              "falling back to scenario defaults.")
        return None, None

_SRTT_PRIOR_MS, _RTTVAR_PRIOR_MS = _load_priors()


def _sample_inference_delay_s(rng, mean_ms, cold_start, drift_active):
    mult = 1.0
    if cold_start:
        mult *= COLD_START_MULTIPLIER
    if drift_active:
        mult *= TEMP_DRIFT_MULT
    sample = rng.gauss(mean_ms, mean_ms * INFERENCE_DELAY_JITTER)
    return max(mean_ms * 0.5, sample) * mult / 1000.0


def _packet_airtime_s(packet_size):
    payload_ms = (packet_size * 8.0 / PHY_RATE_BPS) * 1000.0
    return (PHY_PREAMBLE_AND_MAC_MS + payload_ms + ACK_AND_TURNAROUND_MS) / 1000.0


def _traffic_intensity(s):
    offered_rate = s.clients / max(1e-6, s.mean_interarrival_ms / 1000.0)
    return offered_rate * _packet_airtime_s(s.packet_size)


def _hidden_collision_probability(s):
    intensity = _traffic_intensity(s)
    return min(0.22, 1.0 - math.exp(-0.16 * intensity))


def percent_improvement(value, baseline, min_baseline=1e-9):
    if baseline <= min_baseline:
        return None
    return ((value - baseline) / baseline) * 100.0


def format_percent_label(value, baseline, min_baseline=1e-9):
    pct = percent_improvement(value, baseline, min_baseline=min_baseline)
    return "n/a" if pct is None else f"{pct:+.1f}%"


@dataclass(frozen=True)
class Scenario:
    clients: int
    packet_size: int
    pdr: float
    base_rtt_ms: float = 200.0
    jitter_ms: float = 50.0
    sim_time_s: float = 60.0
    max_retx: int = 4
    mean_interarrival_ms: float = 1200.0
    burst_error_prob: float = 0.0
    burst_recovery_prob: float = 1.0
    burst_loss_multiplier: float = 1.0
    queue_capacity: int = 25
    ack_loss_factor: float = 0.97


@dataclass
class Result:
    goodput_bytes_per_s: float
    pdr_percentage: float
    energy_mj: float
    avg_rto_ms: float
    acks: int
    retransmissions: int
    avg_retransmissions_per_transaction: float
    mean_latency_ms: float
    deadline_miss_rate: float
    carried_load_per_node_bps: List[float] = field(default_factory=list)


class ChannelModel:
    def __init__(self, scenario, rng):
        self.s   = scenario
        self.rng = rng
        self.in_bad_state = False

    def _burst_enabled(self):
        return self.s.burst_error_prob > 0.0 and self.s.burst_loss_multiplier < 1.0

    def step(self):
        if not self._burst_enabled():
            self.in_bad_state = False
            return
        if self.in_bad_state:
            if self.rng.random() < self.s.burst_recovery_prob:
                self.in_bad_state = False
        else:
            if self.rng.random() < self.s.burst_error_prob:
                self.in_bad_state = True

    def current_success_probability(self):
        base        = self.s.pdr * self.s.ack_loss_factor
        packet_bits = self.s.packet_size * 8
        bit_error_rate   = 0.00003
        ber_success_prob = math.pow(1 - bit_error_rate, packet_bits)

        intensity      = _traffic_intensity(self.s)
        capacity_scale = max(1.0, float(self.s.queue_capacity)) / 5.0
        norm_load      = intensity / capacity_scale
        overload       = max(0.0, norm_load - 1.0)
        cong_penalty   = math.exp(-0.45 * (overload ** 2.0))

        if norm_load > 2.5:
            qdrop = min(0.20, 0.04 * (norm_load - 2.5))
            base *= 1.0 - qdrop

        raw = base * ber_success_prob * cong_penalty
        if self.in_bad_state:
            raw *= self.s.burst_loss_multiplier
        return max(0.05, min(0.985, raw))


def _build_ml_features(scenario, srtt_ms, rttvar_ms, history,
                        attempt, last_app_time, current_time,
                        drop_model=None):
    success_rate = sum(history[-10:]) / len(history[-10:]) if history else 1.0
    arrival_gap  = max(0.0, (current_time - last_app_time) * 1000.0)
    iat          = float(min(arrival_gap, 35000.0))
    if drop_model is not None:
        _row  = np.array([[float(srtt_ms), float(rttvar_ms),
                           float(success_rate), iat]], dtype=np.float64)
        dp = float(drop_model.predict_proba(_row)[0, 1])
    else:
        dp = float(min(1.0, max(0.0, 1.0 - success_rate)))
    return {
        "srtt_ms":             float(srtt_ms),
        "rttvar_ms":           float(rttvar_ms),
        "attempt":             float(attempt),
        "success_rate_window": float(success_rate),
        "inter_arrival_ms":    iat,
        "drop_prob":           dp,
    }


def _prediction_to_ms(raw_pred, features):
    if _SRTT_PRIOR_MS is not None:
        fallback_ms = max(ML_RTO_MIN_MS, _SRTT_PRIOR_MS)
    else:
        fallback_ms = max(ML_RTO_MIN_MS,
                          features["srtt_ms"] + 4.0 * features["rttvar_ms"])
    pred_ms = float(np.expm1(np.clip(raw_pred, 0.0, 12.0)))
    return pred_ms if (np.isfinite(pred_ms) and pred_ms > 0) else fallback_ms


def _get_drop_threshold(features):
    sr = features["success_rate_window"]
    if sr < 0.3:
        return 0.75
    elif sr > 0.8:
        return 0.92
    return DROP_DECISION_THRESHOLD


def _predict_ml_rto(model, features, rng):
    attempt = int(features["attempt"])

    if isinstance(model, dict):
        sub = model.get(attempt)
        if sub is None:
            available = sorted(model.keys())
            sub = model[available[-1]]
        arr      = np.array([[features[n] for n in ML_FEATURES]], dtype=np.float64)
        raw_pred = float(sub.predict(arr)[0])
        attempt_cap = min(60000.0, 2000.0 * (2.0 ** attempt))
        raw_pred    = float(np.clip(raw_pred, 0.0, np.log1p(attempt_cap)))
    else:
        arr      = np.array([[features[n] for n in ML_FEATURES]], dtype=np.float64)
        raw_pred = float(model.predict(arr)[0]) if model is not None else 0.0

    pred_ms = _prediction_to_ms(raw_pred, features)
    pred_ms *= max(0.75, rng.gauss(1.0, PREDICTION_NOISE))
    if BAD_RTO_PROB > 0 and rng.random() < BAD_RTO_PROB:
        pred_ms *= BAD_RTO_FACTOR
    return max(ML_RTO_MIN_MS, min(ML_RTO_MAX_MS, pred_ms))


def simulate_protocol(s: Scenario, protocol: str,
                       model=None, drop_model=None, seed: int = 1) -> Result:
    rng     = random.Random(seed)
    channel = ChannelModel(s, rng)
    current_time = 0.0
    sent, acks, retx, completed, misses, generated = 0, 0, 0, 0, 0, 0
    total_latency_ms = 0.0
    total_ml_energy  = 0.0

    next_event    = [rng.uniform(0.0, s.mean_interarrival_ms / 1000.0)
                     for _ in range(s.clients)]
    last_app_time = list(next_event)
    pending       = [False] * s.clients
    tx_start_time = [0.0]   * s.clients
    attempt_no    = [0]     * s.clients

    mcu_busy_until         = [0.0]  * s.clients
    last_inference_end     = [-1e9] * s.clients
    energy_per_node        = [0.0]  * s.clients
    throttled_until        = [0.0]  * s.clients
    inferences_since_drift = [0]    * s.clients
    drift_remaining        = [0]    * s.clients

    if protocol in ("ML", "ML_BO") and _SRTT_PRIOR_MS is not None:
        srtt   = [_SRTT_PRIOR_MS   for _ in range(s.clients)]
        rttvar = [_RTTVAR_PRIOR_MS for _ in range(s.clients)]
        hist = [[1,0,1,0,1,0,1,0,1,0] for _ in range(s.clients)]
    else:
        srtt   = [s.base_rtt_ms       for _ in range(s.clients)]
        rttvar = [s.base_rtt_ms / 2.0 for _ in range(s.clients)]
        hist   = [[1] * 8             for _ in range(s.clients)]

    srtt_weak   = [s.base_rtt_ms       for _ in range(s.clients)]
    rttvar_weak = [s.base_rtt_ms / 4.0 for _ in range(s.clients)]
    rto = [rng.uniform(2000.0, 3000.0) for _ in range(s.clients)]
    cached_rto_val = [ML_RTO_MIN_MS] * s.clients

    channel_busy_until    = 0.0
    airtime_s             = _packet_airtime_s(s.packet_size)
    hidden_collision_prob = _hidden_collision_probability(s)

    def run_inference(idx, t_now, mean_ms):
        idle_ms      = (t_now - last_inference_end[idx]) * 1000.0
        cold         = idle_ms > COLD_START_IDLE_MS
        drift_active = drift_remaining[idx] > 0
        if drift_active:
            drift_remaining[idx] -= 1
        delay_s   = _sample_inference_delay_s(rng, mean_ms, cold, drift_active)
        energy_mj = (mean_ms / INFERENCE_DELAY_MS) * INFERENCE_ENERGY_MJ
        if cold:
            energy_mj *= 1.4
        last_inference_end[idx] = t_now + delay_s
        inferences_since_drift[idx] += 1
        if inferences_since_drift[idx] >= TEMP_DRIFT_PACKETS:
            inferences_since_drift[idx] = 0
            drift_remaining[idx]        = TEMP_DRIFT_DURATION
        return delay_s, energy_mj

    def schedule_next_application(idx, not_before):
        gap_s            = rng.expovariate(1.0 / s.mean_interarrival_ms) / 1000.0
        next_app_arrival = max(not_before, last_app_time[idx]) + gap_s
        if protocol == "ML":
            idle_s = next_app_arrival - last_app_time[idx]
            if idle_s > 2.0:
                decay = 0.90 ** int(idle_s / 2.0)
                hist[idx] = [round(0.5 + (h - 0.5) * decay) for h in hist[idx]]
        last_app_time[idx] = next_app_arrival
        next_event[idx]    = min(next_app_arrival, s.sim_time_s)

    while current_time < s.sim_time_s:
        next_t = min(next_event)
        if next_t >= s.sim_time_s:
            current_time = s.sim_time_s
            break
        current_time = next_t
        ready = [i for i, t in enumerate(next_event) if t <= current_time + 1e-9]
        channel.step()
        p_success_base = channel.current_success_probability()

        for idx in ready:
            if current_time >= s.sim_time_s:
                next_event[idx] = s.sim_time_s
                continue

            if protocol in ("ML", "ML_BO") and current_time < throttled_until[idx]:
                next_event[idx] = min(throttled_until[idx], s.sim_time_s)
                continue

            if not pending[idx]:
                pending[idx]       = True
                tx_start_time[idx] = current_time
                attempt_no[idx]    = 0
                generated         += 1

            _max_retx = 6 if protocol in ("ML", "ML_BO") else s.max_retx
            if attempt_no[idx] > _max_retx:
                pending[idx] = False
                misses      += 1
                schedule_next_application(idx, current_time)
                continue

            client_time = current_time
            p_success   = p_success_base
            attempt     = attempt_no[idx]
            sent       += 1

            if protocol in ("ML", "ML_BO"):
                if client_time < mcu_busy_until[idx]:
                    client_time = mcu_busy_until[idx]

                if protocol == "ML" or attempt == 0:
                    infer_cost_ms = (INFERENCE_DELAY_MS if attempt == 0
                                     else INFERENCE_DELAY_MS * WARM_INFERENCE_FRACTION)
                    delay_s, e_mj       = run_inference(idx, client_time, infer_cost_ms)
                    client_time        += delay_s
                    mcu_busy_until[idx] = client_time
                    total_ml_energy    += e_mj
                    energy_per_node[idx]+= e_mj
                    p_success          *= MCU_BUSY_ACK_PENALTY

                    ack_miss_prob = (ACK_MISS_DURING_INFERENCE if attempt == 0
                                     else ACK_MISS_DURING_INFERENCE * WARM_INFERENCE_FRACTION)
                    if rng.random() < ack_miss_prob:
                        p_success *= 0.55

                    if energy_per_node[idx] > ENERGY_BUDGET_MJ:
                        throttled_until[idx]  = client_time + THROTTLE_SLEEP_MS / 1000.0
                        energy_per_node[idx] *= 0.5

                    features  = _build_ml_features(
                        s, srtt[idx], rttvar[idx], hist[idx], attempt,
                        last_app_time[idx], current_time, drop_model=drop_model)
                    local_rto = _predict_ml_rto(model, features, rng)

                    if protocol == "ML_BO":
                        coap_equivalent = rng.uniform(STD_RTO_MIN_MS, 3000.0)
                        cached_rto_val[idx] = min(local_rto, coap_equivalent)
                else:
                    features     = _build_ml_features(
                        s, srtt[idx], rttvar[idx], hist[idx], attempt,
                        last_app_time[idx], current_time, drop_model=drop_model)
                    jitter_ratio = rttvar[idx] / max(1.0, srtt[idx])
                    vbf          = 1.5 + min(1.0, jitter_ratio)
                    rand_factor  = rng.uniform(1.0, 1.5)
                    local_rto    = max(ML_RTO_MIN_MS,
                                       min(ML_RTO_MAX_MS,
                                           cached_rto_val[idx] * (vbf ** attempt) * rand_factor))

                if drop_model is not None and attempt >= DROP_MIN_ATTEMPT:
                    d_delay, d_energy    = run_inference(idx, client_time,
                                                         DROP_INFERENCE_DELAY_MS)
                    client_time         += d_delay
                    mcu_busy_until[idx]  = client_time
                    total_ml_energy     += d_energy
                    energy_per_node[idx]+= d_energy
                    feat_row = np.array([[features[n] for n in DROP_FEATURES]],
                                        dtype=np.float64)
                    p_drop        = float(drop_model.predict_proba(feat_row)[0, 1])
                    dyn_threshold = _get_drop_threshold(features)
                    false_drop    = FALSE_DROP_PROB > 0 and rng.random() < FALSE_DROP_PROB
                    if p_drop >= dyn_threshold or false_drop:
                        pending[idx] = False
                        misses      += 1
                        schedule_next_application(idx, client_time)
                        continue

            elif protocol == "CoCoA":
                vbf = 3.0 if rto[idx] < 1500 else (1.5 if rto[idx] > 3000 else 2.0)
                local_rto = max(STD_RTO_MIN_MS,
                                min(STD_RTO_MAX_MS, rto[idx] * (vbf ** attempt)))
            elif protocol == "CoCoA+":
                idle_s = max(0.0, current_time - last_app_time[idx])
                if idle_s > 4.0:
                    rto[idx] = max(STD_RTO_MIN_MS, rto[idx] * 0.85)
                vbf = 2.5 if rto[idx] < 1500 else (1.4 if rto[idx] > 2500 else 1.8)
                local_rto = max(STD_RTO_MIN_MS,
                                min(STD_RTO_MAX_MS, rto[idx] * (vbf ** attempt)))
            else:  # CoAP RFC 7252
                local_rto = max(STD_RTO_MIN_MS,
                                min(STD_RTO_MAX_MS, rto[idx] * (2.0 ** attempt)))

            client_time += MAC_PROCESSING_MS / 1000.0

            if client_time < channel_busy_until:
                client_time = channel_busy_until + rng.uniform(
                    CCA_BACKOFF_MIN_MS, CCA_BACKOFF_MAX_MS) / 1000.0
                p_success  *= BUSY_CHANNEL_PENALTY
                if protocol in ("ML", "ML_BO"):
                    mcu_busy_until[idx] = max(mcu_busy_until[idx], client_time)

            if rng.random() < hidden_collision_prob:
                p_success *= HIDDEN_COLLISION_PENALTY

            if client_time >= s.sim_time_s:
                pending[idx]    = False
                misses         += 1
                next_event[idx] = s.sim_time_s
                continue

            channel_busy_until = client_time + airtime_s
            rtt_sample         = max(20.0, rng.gauss(s.base_rtt_ms, s.jitter_ms))
            ack_before_timeout = rtt_sample <= local_rto

            if rng.random() < p_success and ack_before_timeout:
                acks             += 1
                completed        += 1
                total_latency_ms += ((client_time - tx_start_time[idx]) * 1000.0
                                     + rtt_sample)

                if protocol in ("ML", "ML_BO"):
                    hist[idx]    = (hist[idx] + [1])[-10:]
                    delay_s, e_mj       = run_inference(idx, client_time, INFERENCE_DELAY_MS)
                    client_time        += delay_s
                    mcu_busy_until[idx] = client_time
                    total_ml_energy    += e_mj
                    energy_per_node[idx]+= e_mj
                    feat_next  = _build_ml_features(
                        s, rtt_sample, abs(srtt[idx] - rtt_sample),
                        hist[idx], 0, last_app_time[idx], current_time,
                        drop_model=drop_model)
                    rto[idx]    = _predict_ml_rto(model, feat_next, rng)
                    alpha_srtt  = 0.5 if len(hist[idx]) <= 5 else 0.125
                    srtt[idx]   = (1 - alpha_srtt) * srtt[idx] + alpha_srtt * rtt_sample
                    rttvar[idx] = 0.75 * rttvar[idx] + 0.25 * abs(srtt[idx] - rtt_sample)

                elif protocol == "CoCoA":
                    alpha = 0.125
                    if attempt == 0:
                        srtt[idx]   = (1-alpha)*srtt[idx]   + alpha*rtt_sample
                        rttvar[idx] = 0.75*rttvar[idx] + 0.25*abs(srtt[idx]-rtt_sample)
                        rto[idx]    = srtt[idx] + 4*rttvar[idx]
                    else:
                        srtt_weak[idx]   = (1-alpha)*srtt_weak[idx]   + alpha*rtt_sample
                        rttvar_weak[idx] = 0.75*rttvar_weak[idx] + 0.25*abs(srtt_weak[idx]-rtt_sample)
                        rto[idx] = (0.5*(srtt[idx]+4*rttvar[idx])
                                    + 0.5*(srtt_weak[idx]+4*rttvar_weak[idx]))
                    rto[idx] = max(STD_RTO_MIN_MS, min(STD_RTO_MAX_MS, rto[idx]))

                elif protocol == "CoCoA+":
                    alpha = 0.05
                    if attempt == 0:
                        srtt[idx]   = (1-alpha)*srtt[idx]   + alpha*rtt_sample
                        rttvar[idx] = 0.7*rttvar[idx] + 0.3*abs(srtt[idx]-rtt_sample)
                        rto[idx]    = srtt[idx] + 3*rttvar[idx]
                    else:
                        srtt_weak[idx]   = (1-alpha)*srtt_weak[idx]   + alpha*rtt_sample
                        rttvar_weak[idx] = 0.7*rttvar_weak[idx] + 0.3*abs(srtt_weak[idx]-rtt_sample)
                        rto[idx] = (0.25*(srtt[idx]+3*rttvar[idx])
                                    + 0.75*(srtt_weak[idx]+3*rttvar_weak[idx]))
                    rto[idx] = max(STD_RTO_MIN_MS, min(STD_RTO_MAX_MS, rto[idx]))

                elif protocol == "CoAP":
                    rto[idx] = rng.uniform(2000.0, 3000.0)
                    rto[idx] = max(STD_RTO_MIN_MS, min(STD_RTO_MAX_MS, rto[idx]))

                pending[idx] = False
                schedule_next_application(idx, client_time + rtt_sample / 1000.0)
                continue

            retx += 1
            if protocol in ("ML", "ML_BO"):
                hist[idx] = (hist[idx] + [0])[-10:]
            attempt_no[idx] += 1

            _max_retx = 6 if protocol in ("ML", "ML_BO") else s.max_retx
            if attempt_no[idx] > _max_retx:
                pending[idx] = False
                misses      += 1
                schedule_next_application(idx, current_time)
            else:
                next_event[idx] = min(client_time + local_rto / 1000.0, s.sim_time_s)

    goodput_bytes_per_s = (completed * s.packet_size) / max(1e-6, s.sim_time_s)

    return Result(
        goodput_bytes_per_s=goodput_bytes_per_s,
        pdr_percentage=(completed / max(1, generated)) * 100.0,
        energy_mj=(sent * 0.1) + total_ml_energy,
        avg_rto_ms=sum(rto) / len(rto),
        acks=acks,
        retransmissions=retx,
        avg_retransmissions_per_transaction=retx / completed if completed else 0.0,
        mean_latency_ms=total_latency_ms / completed if completed else 0.0,
        deadline_miss_rate=(misses / max(1, completed + misses)) * 100.0,
        carried_load_per_node_bps=[],
    )
