# -*- coding: utf-8 -*-
"""[ADAPTIVE OFFLINE REPLAY]  저장된 trajectory 만으로 예산 정책을 비교한다.

engine 실행 0. production 파일 수정 0. 이 파일은 순수 offline 분석이다.

selection 규칙은 tk_r7.nmcs_pass(:542) 를 그대로 옮긴 것이다.
    bv  = max(probe_value)
    grp = 동점군
    len(grp)==1 -> 그대로, TIEHOLD 해제
    아니면 TIEHOLD(node) 가 동점군에 살아있고 잔여>0 이면 유지
    아니면 min(vis0, action) 을 고르고 TIEHOLD[node]=[a, K-1]
K = NMCS_K = 2.
"""
import collections
import io
import json
import math
import os
import sys

NMCS_K = 2
INF = 10 ** 9


# ===================== 1. 데이터 적재 / semantics =====================
def build(path):
    d = json.load(io.open(path, encoding="utf-8"))
    seed = d["seed"]
    inner = collections.defaultdict(list)
    for r in d["candidate_trace"]:
        inner[(r["node_id"], r["action"], r["probe_index"])].append(r)
    for k in inner:
        inner[k].sort(key=lambda x: x["inner_index"])

    probes = {}          # (node, action, j) -> probe dict
    for r in d["candidate_probe"]:
        k = (r["node_id"], r["action"], r["probe_index"])
        rows = inner.get(k, [])
        # nmcs_inner 는 첫 최대값을 유지한다 (if v > best)
        bi, bv = None, -1
        for x in rows:
            if x["iter_value"] > bv:
                bv, bi = x["iter_value"], x
        probes[k] = {
            "seed": seed, "node": r["node_id"], "a": r["action"],
            "j": r["probe_index"], "commit": r["commit_id"],
            "call": r["global_call_index"],
            "v": r["probe_value"],
            "vis0": (rows[0]["act_visits_before"] if rows else 0),
            "val_before": r["value_before"], "val_after": r["value_after"],
            "inner": len(rows),
            "forced": sum(x["forced_steps"] for x in rows),
            "roll": sum(x["rollout_steps"] for x in rows),
            "best_child": (bi["child_node"] if bi else None),
            "ncand": r["candidate_count"], "depth": r["node_depth"],
        }

    traj = collections.defaultdict(list)
    for k in sorted(probes):
        traj[(k[0], k[1])].append(probes[k])

    commits = {}
    for p in probes.values():
        c = commits.setdefault(p["commit"], {"commit": p["commit"],
                                             "call": p["call"],
                                             "node": p["node"],
                                             "ncand": p["ncand"],
                                             "depth": p["depth"], "cs": []})
        c["cs"].append(p)
    for c in commits.values():
        c["cs"].sort(key=lambda p: p["a"])
    order = [commits[i] for i in sorted(commits)]
    return {"seed": seed, "probes": probes, "traj": traj,
            "commits": order, "raw": d}


def finalval(traj, node, a):
    """평가 전용. 정책 판단에는 절대 쓰지 않는다."""
    t = traj.get((node, a))
    return t[-1]["val_after"] if t else 0


def maxfuture(traj, node, a, t):
    """probe t 이후의 max(B(t+1)..Bk) = 누적 value. 평가 전용.
    probe_value 만 보면 다른 iterate 의 backup 이 올린 부분을 놓친다."""
    tr = traj.get((node, a), [])
    fut = [p["val_after"] for p in tr[t:]]
    return max(fut) if fut else -1


def curval(traj, node, a, t):
    """probe t 까지 관측된 누적 value Bt. t<=0 이면 B0."""
    tr = traj.get((node, a), [])
    if not tr:
        return 0
    if t <= 0:
        return tr[0]["val_before"]
    return tr[min(t, len(tr)) - 1]["val_after"]


# ===================== 2. selection (nmcs_pass 그대로) =====================
def select(cands, tiehold, node):
    bv = max(c["v"] for c in cands)
    grp = [c for c in cands if c["v"] == bv]
    if len(grp) == 1:
        tiehold.pop(node, None)
        return grp[0], False
    gset = dict((c["a"], c) for c in grp)
    h = tiehold.get(node)
    if h is not None and h[0] in gset and h[1] > 0:
        h[1] -= 1
        return gset[h[0]], True
    p = min(grp, key=lambda c: (c["vis0"], c["a"]))
    tiehold[node] = [p["a"], NMCS_K - 1]
    return p, True


# ===================== 3. 정책 =====================
class Policy(object):
    """cap(rank, n, spent) -> 이 후보가 지금까지 쓸 수 있는 누적 probe 수."""

    def __init__(self, name, kind, arg=None):
        self.name = name
        self.kind = kind
        self.arg = arg

    def cap(self, rank, n, obs_first):
        if self.kind == "base":
            return INF
        if self.kind == "B":
            return self.arg
        if self.kind == "T":                      # value threshold
            if obs_first is None:
                return 1
            return 1 if obs_first <= self.arg else INF
        if self.kind == "C":                      # successive halving
            caps = self.arg                       # [(frac_div, cum_cap), ...]
            best = 0
            for div, cc in caps:
                m = max(1, int(math.ceil(n / float(div))))
                if rank < m:
                    best = max(best, cc)
            return best
        if self.kind == "E":                      # two-stage
            s1, frac = self.arg
            m = max(1, int(math.ceil(n * frac)))
            return INF if rank < m else s1
        return INF


def policies():
    P = [Policy("Baseline", "base")]
    for k in (1, 2, 3, 5, 8, 10, 15, 20):
        P.append(Policy("B K=%d" % k, "B", k))
    P.append(Policy("C 1/2/4/8", "C", [(1, 1), (2, 3), (4, 7), (8, 15)]))
    P.append(Policy("C 1/2/4/8/16", "C",
                    [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31)]))
    for s1 in (2, 3, 5):
        P.append(Policy("E %d->50%%" % s1, "E", (s1, 0.50)))
    for s1 in (2, 3, 5):
        P.append(Policy("E %d->25%%" % s1, "E", (s1, 0.25)))
    for t in (0, 1500, 1700, 3200, 4500):
        P.append(Policy("T B1<=%d" % t, "T", t))
    return P


# ===================== 4. replay =====================
def replay(ds, pol, watch=None):
    """commit 단위 counterfactual. 미래 정보는 판단에 쓰지 않는다."""
    traj = ds["traj"]
    spent = collections.Counter()          # (node,a) -> 정책이 준 probe 수
    obs = {}                               # (node,a) -> 최고 관측 probe_value
    obs1 = {}                              # (node,a) -> B1 (첫 probe 관측값)
    lastvis = {}                           # (node,a) -> 마지막 관측 vis0
    tiehold = {}
    out = {"probes": 0, "inner": 0, "forced": 0, "roll": 0,
           "sel": {}, "cuts": [], "watch": []}
    for c in ds["commits"]:
        node, n = c["node"], c["ncand"]
        cs = c["cs"]
        # 순위: 관측된 최고값 내림차순 -> 마지막 vis0 -> action
        def rk(p):
            k = (node, p["a"])
            return (-obs.get(k, -1), lastvis.get(k, 0), p["a"])
        rank = dict((p["a"], i) for i, p in enumerate(sorted(cs, key=rk)))
        view = []
        for p in cs:
            k = (node, p["a"])
            cap = pol.cap(rank[p["a"]], n, obs1.get(k))
            if spent[k] < cap:                       # probe 허용
                spent[k] += 1
                out["probes"] += 1
                out["inner"] += p["inner"]
                out["forced"] += p["forced"]
                out["roll"] += p["roll"]
                if k not in obs1:
                    obs1[k] = p["v"]
                if p["v"] > obs.get(k, -1):
                    obs[k] = p["v"]
                lastvis[k] = p["vis0"]
                view.append({"a": p["a"], "v": p["v"], "vis0": p["vis0"],
                             "p": p, "fresh": True})
            else:                                    # 동결
                fut = maxfuture(traj, node, p["a"], spent[k])
                cv = curval(traj, node, p["a"], spent[k])
                if fut > cv:
                    out["cuts"].append({
                        "node": node, "a": p["a"], "t": spent[k],
                        "obs": cv, "future": fut,
                        "commit": c["commit"], "call": c["call"],
                        "final": finalval(traj, node, p["a"])})
                view.append({"a": p["a"], "v": obs.get(k, -1),
                             "vis0": lastvis.get(k, 0), "p": p,
                             "fresh": False})
                if watch and (node, p["a"]) == watch:
                    out["watch"].append((c["commit"], c["call"], spent[k],
                                         obs.get(k, -1)))
        pick, _ = select(view, tiehold, node)
        out["sel"][c["commit"]] = pick["a"]
    return out


# ===================== 5. baseline 재현 검증 =====================
def verify(ds):
    traj = ds["traj"]
    tiehold = {}
    sel = {}
    ok = bad = 0
    detail = []
    byc = {}
    for c in ds["commits"]:
        byc.setdefault(c["call"], []).append(c)
    for c in ds["commits"]:
        view = [{"a": p["a"], "v": p["v"], "vis0": p["vis0"], "p": p}
                for p in c["cs"]]
        pick, _ = select(view, tiehold, c["node"])
        sel[c["commit"]] = pick["a"]
        nxt = [x for x in byc[c["call"]] if x["commit"] == c["commit"] + 1]
        if nxt:
            pred = pick["p"]["best_child"]
            act = nxt[0]["node"]
            if pred == act:
                ok += 1
            else:
                bad += 1
                if len(detail) < 6:
                    detail.append((c["commit"], c["node"], pick["a"],
                                   pred, act))
    return sel, ok, bad, detail


# ===================== 6. 평가 =====================
def evaluate(ds, base_sel, pol_out):
    traj = ds["traj"]
    same = 0
    samev = 0
    tot = 0
    reg = 0
    loss = 0
    gain = 0
    bbest = 0
    pbest = 0
    worst = None
    wcommit = None
    for c in ds["commits"]:
        ba = base_sel[c["commit"]]
        pa = pol_out["sel"][c["commit"]]
        bf = finalval(traj, c["node"], ba)
        pf = finalval(traj, c["node"], pa)
        tot += 1
        if ba == pa:
            same += 1
        else:
            if wcommit is None:
                wcommit = (c["commit"], c["node"], ba, pa, bf, pf)
        if bf == pf:
            samev += 1
        reg += (bf - pf)
        if bf > pf:
            loss += bf - pf
        else:
            gain += pf - bf
        bbest = max(bbest, bf)
        pbest = max(pbest, pf)
        if bf - pf > 0 and (worst is None or bf - pf > worst[4] - worst[5]):
            worst = (c["commit"], c["node"], ba, pa, bf, pf)
    return {"same": same, "samev": samev, "tot": tot, "reg": reg,
            "loss": loss, "gain": gain,
            "bbest": bbest, "pbest": pbest, "worst": worst,
            "wcommit": wcommit}


DELAY = [("B1<=0 -> >=9000", 0, 9000),
         ("B1<=1500 -> >=9000", 1500, 9000),
         ("B1<=1700 -> >=16000", 1700, 16000),
         ("B1<=1700 -> >=23000", 1700, 23000)]


def delayed_sets(ds):
    traj = ds["traj"]
    out = {}
    for nm, lo, hi in DELAY:
        s = set()
        for k, tr in traj.items():
            if tr[0]["v"] <= lo and finalval(traj, k[0], k[1]) >= hi:
                s.add(k)
        out[nm] = s
    return out


def main():
    paths = sys.argv[1:]
    DS = [build(p) for p in paths]

    print("=" * 104)
    print("[ADAPTIVE OFFLINE REPLAY]   engine 실행 0 / production UNCHANGED")
    print("=" * 104)

    # ---- 1. semantics ----
    print()
    print("  [1. TRAJECTORY SEMANTICS]")
    d0 = DS[0]
    tr = d0["traj"][(0, 1)]
    print("    probe j  = (node,action) 가 j 번째로 commit 후보가 되어")
    print("               nmcs_inner 가 호출된 회차. commit 당 후보별 정확히 1회.")
    print("    B0       = 첫 probe 직전 acts[a].value          (value_before)")
    print("    Bj       = j 번째 probe 직후 acts[a].value       (value_after)")
    print("    vj       = j 번째 probe 가 반환한 값             (probe_value)")
    print("               <- nmcs_pass 가 실제로 비교하는 값은 이것이다")
    print("    Bk       = 원래 탐색의 최종 best value")
    print("    seed %s node0/a1 : probe %d개, probe_index %d..%d (연속)"
          % (d0["seed"], len(tr), tr[0]["j"], tr[-1]["j"]))
    print("    B0=%d  v=[%s]" % (tr[0]["val_before"],
                                 ",".join(str(p["v"]) for p in tr[:12]) + ",..."))
    print("    B =[%s]" % (",".join(str(p["val_after"]) for p in tr[:12]) + ",..."))
    mism = sum(1 for k, t in d0["traj"].items()
               for i, p in enumerate(t) if p["j"] != i + 1)
    print("    probe_index <-> 호출 회차 불일치 : %d  (0 이어야 한다)" % mism)
    vg = sum(1 for k, t in d0["traj"].items() for p in t
             if p["v"] > p["val_after"])
    print("    v > B 인 probe : %d  (backup 이 max 이므로 0 이어야 한다)" % vg)

    # ---- 2. baseline 재현 ----
    print()
    print("  [2. BASELINE REPRODUCTION]")
    BASE = []
    allok = True
    for ds in DS:
        sel, ok, bad, det = verify(ds)
        BASE.append(sel)
        tp = sum(1 for _ in ds["probes"])
        print("    seed %-9s commit %-4d  후보 %-5d  probe %-5d"
              % (ds["seed"], len(ds["commits"]), len(ds["traj"]), tp))
        print("        다음 commit node 예측  일치 %d / 불일치 %d  -> %s"
              % (ok, bad, "PASS" if bad == 0 else "FAIL"))
        if bad:
            allok = False
            for x in det:
                print("          commit %d node %d a%d  pred %s != act %s" % x)
    print("    baseline reproduction : %s" % ("PASS" if allok else "FAIL"))
    if not allok:
        print("    FAIL -> adaptive 분석 중단")
        return

    # ---- 3~6. 정책 비교 ----
    POL = policies()
    DSETS = [delayed_sets(ds) for ds in DS]
    rows = []
    store = {}
    for pol in POL:
        agg = collections.Counter()
        dkeep = collections.Counter()
        dtot = collections.Counter()
        felim = collections.Counter()
        worsts = []
        for di, ds in enumerate(DS):
            out = replay(ds, pol)
            store[(pol.name, di)] = out
            ev = evaluate(ds, BASE[di], out)
            agg["same"] += ev["same"]
            agg["samev"] += ev["samev"]
            agg["tot"] += ev["tot"]
            agg["reg"] += ev["reg"]
            agg["loss"] += ev["loss"]
            agg["gain"] += ev["gain"]
            agg["bbest"] += ev["bbest"]
            agg["pbest"] += ev["pbest"]
            agg["probes"] += out["probes"]
            agg["inner"] += out["inner"]
            agg["forced"] += out["forced"]
            agg["roll"] += out["roll"]
            if ev["worst"]:
                worsts.append((ds["seed"],) + ev["worst"])
            cutkeys = set((c["node"], c["a"]) for c in out["cuts"])
            for nm, _lo, _hi in DELAY:
                s = DSETS[di][nm]
                dtot[nm] += len(s)
                dkeep[nm] += len(s - cutkeys)
            for c in out["cuts"]:
                felim["all"] += 1
                if c["future"] >= 9000:
                    felim["9k"] += 1
                if c["future"] >= 16000:
                    felim["16k"] += 1
                if c["future"] >= 23000:
                    felim["23k"] += 1
        rows.append((pol, agg, dkeep, dtot, felim, worsts))
        store[(pol.name, "agg")] = (agg, dkeep, dtot, felim, worsts)

    b_agg = rows[0][1]
    print()
    print("  [POLICY COMPARISON]   3 seed 합산, commit %d, baseline probe %d"
          % (b_agg["tot"], b_agg["probes"]))
    print()
    hdr = ("    %-14s %7s %7s %9s %9s %9s %8s %8s %8s" %
           ("Policy", "Probe↓", "Step↓", "SameCmt", "SameVal",
            "Regret", "≥9k", "≥16k", "≥23k"))
    print(hdr)
    print("    " + "-" * 96)
    for pol, agg, dkeep, dtot, felim, worsts in rows:
        pr = 100.0 * (b_agg["probes"] - agg["probes"]) / b_agg["probes"]
        bstep = b_agg["forced"] + b_agg["roll"]
        pstep = agg["forced"] + agg["roll"]
        sr = 100.0 * (bstep - pstep) / bstep
        d = []
        for nm, _l, _h in DELAY[1:]:
            d.append("%d/%d" % (dkeep[nm], dtot[nm]))
        nm0 = DELAY[0][0]
        print("    %-14s %6.1f%% %6.1f%% %8.1f%% %8.1f%% %9d %8s %8s %8s"
              % (pol.name, pr, sr,
                 100.0 * agg["same"] / agg["tot"],
                 100.0 * agg["samev"] / agg["tot"],
                 agg["reg"], d[0], d[1], d[2]))
    print()
    print("    Probe↓ = probe 감소율 / Step↓ = forced+rollout step 감소율")
    print("    SameCmt = baseline 과 같은 action 을 고른 commit 비율")
    print("    SameVal = 고른 후보의 최종 value 가 baseline 과 같은 commit 비율")
    print("    Regret  = sum(baseline 선택 후보 최종값 - 정책 선택 후보 최종값)")
    print("    ≥9k/≥16k/≥23k = delayed 후보 생존수/전체 (B1<=1500->9k,")
    print("                     B1<=1700->16k, B1<=1700->23k)")

    # ---- delayed B1<=0 ----
    print()
    print("  [DELAYED REWARD]  B1<=0 -> final>=9000")
    for pol, agg, dkeep, dtot, felim, worsts in rows:
        nm = DELAY[0][0]
        print("    %-14s 생존 %d / %d" % (pol.name, dkeep[nm], dtot[nm]))

    # ---- global best ----
    print()
    print("  [FINAL VALUE REGRET]  (seed 별 최고 선택값의 합)")
    for pol, agg, dkeep, dtot, felim, worsts in rows:
        print("    %-14s baseline %d -> policy %d   regret %d"
              % (pol.name, agg["bbest"], agg["pbest"],
                 agg["bbest"] - agg["pbest"]))

    # ---- false elimination ----
    print()
    print("  [FALSE ELIMINATION]  (Bt 관측 후 잘렸지만 실제 미래 gain > 0)")
    for pol, agg, dkeep, dtot, felim, worsts in rows:
        print("    %-14s 총 %-5d  future>=9k %-4d  >=16k %-4d  >=23k %-4d"
              % (pol.name, felim["all"], felim["9k"], felim["16k"],
                 felim["23k"]))

    # ---- compute saving ----
    print()
    print("  [COMPUTE SAVING]")
    print("    %-14s %8s %8s %9s %9s %9s"
          % ("Policy", "probe", "inner", "forced", "rollout", "total"))
    for pol, agg, dkeep, dtot, felim, worsts in rows:
        bt = b_agg["forced"] + b_agg["roll"]
        pt = agg["forced"] + agg["roll"]
        print("    %-14s %8d %8d %9d %9d %9d  (-%.1f%%)"
              % (pol.name, agg["probes"], agg["inner"], agg["forced"],
                 agg["roll"], pt, 100.0 * (bt - pt) / bt))

    # ---- 14. 정책별 대표 반례 ----
    print()
    print("  [WORST CASES]  정책별 최대 value loss commit")
    for pol, agg, dkeep, dtot, felim, worsts in rows:
        if pol.kind == "base":
            continue
        ws = sorted(worsts, key=lambda x: -(x[5] - x[6]))
        if not ws or ws[0][5] - ws[0][6] <= 0:
            print("    %-14s (value loss 있는 commit 없음)" % pol.name)
            continue
        sd, cid, node, ba, pa, bf, pf = ws[0]
        print("    %-14s seed %s commit %d node %d : baseline a%d(final %d)"
              " -> policy a%d(final %d)  loss %d"
              % (pol.name, sd, cid, node, ba, bf, pa, pf, bf - pf))
        dsx = [x for x in DS if x["seed"] == sd][0]
        tr = dsx["traj"].get((node, ba), [])
        if tr:
            print("        버려진 후보 B0=%d -> %s"
                  % (tr[0]["val_before"],
                     " -> ".join(str(q["val_after"]) for q in tr)))

    # ---- 37300 관련 라인 추적 ----
    print()
    print("  [37300-RELATED DELAY]  seed 20260917 / node 0 / a1")
    ds = [x for x in DS if x["seed"] == "20260917"]
    if ds:
        ds = ds[0]
        tr = ds["traj"][(0, 1)]
        print("    B0=%d  ->  %s" % (tr[0]["val_before"],
                                     " -> ".join(str(p["val_after"]) for p in tr)))
        print("    v   =  %s" % " ".join(str(p["v"]) for p in tr))
        print("    최종 %d, probe %d, 첫 nonzero probe %s"
              % (tr[-1]["val_after"], len(tr),
                 next((p["j"] for p in tr if p["v"] > 0), None)))
        print()
        for pol in POL:
            out = replay(ds, pol, watch=(0, 1))
            cuts = [c for c in out["cuts"] if (c["node"], c["a"]) == (0, 1)]
            if not cuts:
                st = "생존 (끝까지 probe)"
            else:
                c0 = cuts[0]
                st = ("commit %d / call %d 에서 제거  (그때 관측 %d, "
                      "이후 실제 %d, 최종 %d)"
                      % (c0["commit"], c0["call"], c0["obs"], c0["future"],
                         c0["final"]))
            print("    %-14s %s" % (pol.name, st))


if __name__ == "__main__":
    main()
