"""
SICEF-Bench Simulation and Evaluation (Optimized)
===================================================
n=5,000 cases — statistically robust, runs in reasonable time.
All randomness seeded for full reproducibility.
"""

import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

print("Starting SICEF experiment...", flush=True)

RNG = np.random.default_rng(42)

# ── 1. DOMAIN DEFINITIONS ────────────────────────────────────

DOMAINS = {
    "compute": {
        "dims": ["system_logs","resource_metrics","instance_state",
                 "hypervisor_logs","network_interface","storage_io","kernel_messages"],
        "healthy": [
            "collected system logs showing OOM killer triggered on process",
            "checked CPU steal time metrics indicating noisy neighbour contention",
            "reviewed hypervisor placement and confirmed underlying hardware fault",
            "inspected kernel messages for boot sequence errors at startup",
            "validated storage IOPS against provisioned throughput limits",
            "reviewed network interface statistics confirming packet loss",
            "collected memory profiling data showing application memory leak",
        ],
        "stagnant": [
            "asked customer to try restarting the instance again",
            "requested the same system logs already collected previously",
            "followed up to check if the issue is still occurring",
            "asked customer to describe the problem one more time",
            "suggested rebooting to see if that resolves the issue",
            "asked for update on current status of the problem",
            "requested confirmation that the instance is still affected",
        ],
        "anchor": "root cause confirmed hardware fault collected evidence resolved instance"
    },
    "networking": {
        "dims": ["route_trace","firewall_rules","packet_capture",
                 "dns_resolution","bgp_routes","nat_table","security_groups"],
        "healthy": [
            "ran traceroute confirming packet loss at third network hop",
            "reviewed security group rules and found missing inbound rule port 443",
            "captured packets showing TCP retransmissions indicating congestion",
            "checked BGP route table and identified missing network prefix",
            "reviewed NAT gateway logs showing connection table exhaustion",
            "validated DNS resolution returning stale cached record",
            "inspected firewall logs confirming traffic being explicitly dropped",
        ],
        "stagnant": [
            "asked customer to ping the endpoint and report results back",
            "suggested clearing DNS cache without checking the root cause",
            "asked if the issue happens with all traffic or just some",
            "requested customer to try connecting from a different client",
            "followed up asking if connectivity has improved at all",
            "asked customer to check if firewall is blocking anything",
            "suggested restarting the network interface without evidence",
        ],
        "anchor": "confirmed packet loss firewall rule root cause resolved network"
    },
    "storage": {
        "dims": ["disk_usage","io_stats","filesystem_check","snapshot_state",
                 "replication_status","volume_metadata","block_device_errors"],
        "healthy": [
            "reviewed disk usage showing var log consuming 87 percent of volume",
            "checked IO stats confirming queue depth exceeding provisioned limits",
            "ran filesystem check identifying orphaned inodes causing errors",
            "inspected snapshot state showing failure due to concurrent writes",
            "reviewed replication lag metrics showing four hour delay",
            "checked volume metadata confirming attachment state mismatch",
            "reviewed block device error log confirming bad sectors present",
        ],
        "stagnant": [
            "asked customer to check if disk usage has changed recently",
            "suggested deleting temporary files without checking what uses space",
            "asked if the IO errors are still occurring on the volume",
            "followed up to see if snapshot succeeded after retry attempt",
            "asked customer to run df command and share output again",
            "suggested waiting to see if replication lag catches up",
            "asked customer to describe the corruption symptoms once more",
        ],
        "anchor": "identified disk usage io error filesystem root cause confirmed resolved"
    },
    "identity": {
        "dims": ["policy_evaluation","role_bindings","auth_logs",
                 "trust_policy","permission_boundary","session_tokens","iam_conditions"],
        "healthy": [
            "ran IAM policy simulator confirming explicit deny in resource policy",
            "reviewed role trust policy and identified missing principal entry",
            "inspected CloudTrail auth logs showing MFA device mismatch",
            "checked permission boundary blocking action not included in boundary",
            "reviewed session token expiry confirming one hour maximum session",
            "inspected IAM conditions showing invalid IP address condition block",
            "reviewed role bindings confirming cross account trust misconfiguration",
        ],
        "stagnant": [
            "asked customer to try assuming the role again from console",
            "suggested checking if the user has the correct permissions assigned",
            "asked if the access denied error message is still appearing",
            "requested customer to share the exact error message again",
            "asked if the MFA device has been recently changed or reset",
            "followed up to see if the login issue has been resolved",
            "asked customer to try a different browser or clear cookies",
        ],
        "anchor": "policy evaluation role trust root cause access denied resolved confirmed"
    },
    "deployment": {
        "dims": ["deployment_logs","health_check_output","container_logs",
                 "artifact_registry","pipeline_state","config_diff","rollout_history"],
        "healthy": [
            "reviewed deployment logs showing image pull failure due to auth",
            "checked health check endpoint returning 503 due to missing env var",
            "inspected container logs showing OOM kill during application startup",
            "reviewed artifact registry confirming image tag does not exist",
            "checked pipeline state showing stuck waiting for approval gate",
            "compared config diff identifying missing ConfigMap reference",
            "reviewed rollout history showing three consecutive failed deployments",
        ],
        "stagnant": [
            "asked customer to retry the deployment and see what happens",
            "suggested checking if the configuration looks correct to them",
            "asked if the health check endpoint is passing now after change",
            "followed up to see if deployment succeeded after the retry",
            "asked customer to share the error message from console output",
            "suggested rolling back and trying the deployment again later",
            "asked if this deployment worked before the most recent change",
        ],
        "anchor": "container log health check config root cause deployment resolved confirmed"
    },
}

DOMAIN_NAMES = list(DOMAINS.keys())

# ── 2. DATASET GENERATION ────────────────────────────────────

def generate_case(case_id, domain_name, is_stagnating, rng):
    dom = DOMAINS[domain_name]
    n_updates = int(rng.integers(4, 14))
    stagnation_onset = int(rng.integers(1, max(2, n_updates // 2))) if is_stagnating else n_updates + 1

    updates = []
    covered = set()
    all_dims = dom["dims"]

    for i in range(n_updates):
        stagnant_update = is_stagnating and (i >= stagnation_onset)
        if stagnant_update:
            text = str(rng.choice(dom["stagnant"]))
            new_dims = []
        else:
            text = str(rng.choice(dom["healthy"]))
            avail = [d for d in all_dims if d not in covered]
            if avail:
                n_new = min(int(rng.integers(1, 3)), len(avail))
                new_dims = list(rng.choice(avail, n_new, replace=False))
                covered.update(new_dims)
            else:
                new_dims = []
        updates.append({
            "text": text,
            "dims": new_dims,
            "is_stagnant": bool(stagnant_update),
            "ts": float(i * rng.uniform(2, 12))
        })

    # Ground truth: stagnating if >=3 consecutive stagnant updates
    consec = max_consec = 0
    for u in updates:
        if u["is_stagnant"]:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    label = 1 if (is_stagnating and max_consec >= 3) else 0

    return {
        "id": case_id, "domain": domain_name, "label": label,
        "n": n_updates, "all_dims": all_dims, "updates": updates
    }


def make_dataset(n=5000, stag_rate=0.31, seed=42):
    rng = np.random.default_rng(seed)
    domains = (DOMAIN_NAMES * (n // len(DOMAIN_NAMES) + 1))[:n]
    rng.shuffle(domains)
    return [generate_case(i, domains[i], rng.random() < stag_rate, rng) for i in range(n)]


# ── 3. FEATURE FUNCTIONS ─────────────────────────────────────

def sem_novelty(texts, vec):
    if len(texts) < 2:
        return [0.0] * len(texts)
    M = vec.transform(texts).toarray()
    out = [0.0]
    for i in range(1, len(M)):
        sim = float(cosine_similarity(M[i:i+1], M[i-1:i])[0][0])
        out.append(1.0 - sim)
    return out


def coverage_exp(updates, all_dims):
    covered = set()
    total = len(all_dims)
    out = []
    for u in updates:
        new = set(u["dims"]) - covered
        out.append(len(new) / total if total else 0.0)
        covered.update(new)
    return out


def narrative_score(texts, vec, domain):
    anchor_vec = vec.transform([DOMAINS[domain]["anchor"]]).toarray()
    M = vec.transform(texts).toarray()
    return [float(cosine_similarity(M[i:i+1], anchor_vec)[0][0]) for i in range(len(texts))]


# ── 4. DETECTORS ─────────────────────────────────────────────

def detect_IT(case, thresh):
    ts = [u["ts"] for u in case["updates"]]
    return int(any(ts[i]-ts[i-1] > thresh for i in range(1, len(ts))))


def detect_tfidf(case, vec, tau, k=3):
    S = sem_novelty([u["text"] for u in case["updates"]], vec)
    return _rolling(S, tau, k)


def detect_sn(case, vec, tau, k=3):
    S = sem_novelty([u["text"] for u in case["updates"]], vec)
    return _rolling(S, tau, k)


def detect_th(case, vec, tau):
    ups = case["updates"]
    n = len(ups)
    if n < 2: return 0
    ts = [u["ts"] for u in ups]
    gaps = [ts[i]-ts[i-1] for i in range(1, n)]
    U = max(0.0, 1.0 - np.mean(gaps) / 48.0)
    covered = set(d for u in ups for d in u["dims"])
    D = len(covered) / len(case["all_dims"]) if case["all_dims"] else 0.0
    A = len(set(u["text"] for u in ups)) / n
    H = 0.25*U + 0.35*D + 0.25*A + 0.15*0.5
    return int(H < tau)


def detect_scpa(case, vec, tau, k=3, a=0.40, b=0.35, g=0.25):
    texts = [u["text"] for u in case["updates"]]
    S = sem_novelty(texts, vec)
    C = coverage_exp(case["updates"], case["all_dims"])
    N = narrative_score(texts, vec, case["domain"])
    scores = [a*S[i] + b*C[i] + g*N[i] for i in range(len(texts))]
    return _rolling(scores, tau, k)


def _rolling(scores, tau, k):
    if len(scores) < k: return 0
    for i in range(k-1, len(scores)):
        if np.mean(scores[i-k+1:i+1]) < tau:
            return 1
    return 0


def edr(cases, pred_fn, labels):
    stag = [c for c, l in zip(cases, labels) if l == 1]
    early = detected = 0
    for case in stag:
        n = case["n"]
        mid = n // 2
        ups = case["updates"]
        # find flag index
        flag = None
        for i in range(n):
            tmp = {**case, "updates": ups[:i+1]}
            if pred_fn(tmp):
                flag = i
                break
        if flag is not None:
            detected += 1
            if flag < mid:
                early += 1
    return round(early / detected, 3) if detected else 0.0


# ── 5. TUNING ────────────────────────────────────────────────

def tune(cases, fn_list, param_grids):
    best = {}
    labels = [c["label"] for c in cases]
    for name, fn, grid in fn_list:
        best_f1, best_p = 0, grid[0]
        for p in grid:
            preds = [fn(c, p) for c in cases]
            f = f1_score(labels, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_p = f, p
        best[name] = best_p
    return best


# ── 6. MAIN ──────────────────────────────────────────────────

print("Generating 5,000 cases...", flush=True)
all_cases = make_dataset(5000, 0.31, 42)

rng2 = np.random.default_rng(99)
idx = np.arange(len(all_cases)); rng2.shuffle(idx)
n_tr = int(0.70*len(all_cases)); n_va = int(0.15*len(all_cases))
train = [all_cases[i] for i in idx[:n_tr]]
val   = [all_cases[i] for i in idx[n_tr:n_tr+n_va]]
test  = [all_cases[i] for i in idx[n_tr+n_va:]]

test_labels = [c["label"] for c in test]
print(f"Train={len(train)} Val={len(val)} Test={len(test)}", flush=True)
print(f"Test stagnation rate: {sum(test_labels)/len(test_labels):.3f}", flush=True)

print("Fitting TF-IDF...", flush=True)
all_texts = [u["text"] for c in train for u in c["updates"]]
vec = TfidfVectorizer(max_features=300, ngram_range=(1,2))
vec.fit(all_texts)

print("Tuning thresholds on validation set...", flush=True)
val_labels = [c["label"] for c in val]
taus = [0.25, 0.30, 0.35, 0.40, 0.45]
it_hrs = [12.0, 18.0, 24.0, 36.0, 48.0]

def best_tau(cases, fn):
    labels = [c["label"] for c in cases]
    best_f1, best_t = 0, taus[0]
    for t in taus:
        preds = [fn(c, t) for c in cases]
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return best_t

def best_it(cases):
    labels = [c["label"] for c in cases]
    best_f1, best_t = 0, 24.0
    for t in it_hrs:
        preds = [detect_IT(c, t) for c in cases]
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return best_t

it_t    = best_it(val)
tfidf_t = best_tau(val, lambda c,t: detect_tfidf(c, vec, t))
sn_t    = best_tau(val, lambda c,t: detect_sn(c, vec, t))
th_t    = best_tau(val, lambda c,t: detect_th(c, vec, t))
scpa_t  = best_tau(val, lambda c,t: detect_scpa(c, vec, t))

print(f"Tuned: IT={it_t}h  TF-IDF tau={tfidf_t}  SN tau={sn_t}  TH tau={th_t}  SCP-A tau={scpa_t}", flush=True)

print("Evaluating on test set...", flush=True)
it_p    = [detect_IT(c, it_t) for c in test]
tfidf_p = [detect_tfidf(c, vec, tfidf_t) for c in test]
sn_p    = [detect_sn(c, vec, sn_t) for c in test]
th_p    = [detect_th(c, vec, th_t) for c in test]
scpa_p  = [detect_scpa(c, vec, scpa_t) for c in test]

def ev(name, preds):
    p = precision_score(test_labels, preds, zero_division=0)
    r = recall_score(test_labels, preds, zero_division=0)
    f = f1_score(test_labels, preds, zero_division=0)
    return {"method": name, "P": round(p,3), "R": round(r,3), "F1": round(f,3)}

rows = [
    ev("Inactivity Threshold (IT)", it_p),
    ev("TF-IDF Novelty",            tfidf_p),
    ev("SN-Only",                   sn_p),
    ev("Ticket Health Only",        th_p),
    ev("SCP-A (Proposed)",          scpa_p),
]

print("\n=== TABLE 3: OVERALL RESULTS ===")
print(f"{'Method':<32} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*54)
for r in rows:
    print(f"{r['method']:<32} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f}")

print("\nComputing EDR...", flush=True)
edr_it   = edr(test, lambda c: detect_IT(c, it_t),      test_labels)
edr_scpa = edr(test, lambda c: detect_scpa(c, vec, scpa_t), test_labels)
print(f"EDR — IT:    {edr_it:.3f}")
print(f"EDR — SCP-A: {edr_scpa:.3f}")

print("\n=== TABLE 4: PER-DOMAIN F1 ===")
print(f"{'Domain':<16} {'N':>5} {'Stag%':>6} {'IT F1':>7} {'SCP-A':>7} {'ΔF1':>6}")
print("-"*50)
for dom in DOMAIN_NAMES:
    dc = [c for c in test if c["domain"]==dom]
    dl = [c["label"] for c in dc]
    if sum(dl)==0: continue
    it_d  = [detect_IT(c, it_t) for c in dc]
    sc_d  = [detect_scpa(c, vec, scpa_t) for c in dc]
    fi    = round(f1_score(dl, it_d, zero_division=0), 3)
    fs    = round(f1_score(dl, sc_d, zero_division=0), 3)
    pct   = sum(dl)/len(dl)*100
    print(f"{dom:<16} {len(dc):>5} {pct:>5.1f}% {fi:>7.3f} {fs:>7.3f} {fs-fi:>+6.3f}")

print("\n=== SENSITIVITY ANALYSIS (val set) ===")
print(f"{'k':>3} {'tau':>5}  {'F1':>6}")
for k in [2,3,4,5]:
    for t in taus:
        preds = [detect_scpa(c, vec, t, k=k) for c in val]
        f = round(f1_score(val_labels, preds, zero_division=0), 3)
        print(f"{k:>3} {t:>5.2f}  {f:>6.3f}")

# Save
results = {
    "overall": rows,
    "edr": {"IT": edr_it, "SCP-A": edr_scpa},
    "params": {"it_thresh": it_t, "tfidf_tau": tfidf_t,
               "sn_tau": sn_t, "th_tau": th_t, "scpa_tau": scpa_t},
    "dataset": {"total": len(all_cases), "train": len(train),
                "val": len(val), "test": len(test),
                "test_stag_rate": round(sum(test_labels)/len(test_labels),3)}
}
with open("/home/claude/sicef_results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nDone. Results saved to sicef_results.json")
