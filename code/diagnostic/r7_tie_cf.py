# -*- coding: utf-8 -*-
"""[NMCS TIE -> DEEP ALLOCATION]  terminal-ending tie commit 과 offline counterfactual.

engine 0. 기존 trajectory JSON (Sangen 20260917 / 12345 / 777) 만 쓴다.
selection 재현은 tk_r7.nmcs_pass(:596-626) 를 그대로 옮긴 것이다
(adaptive_replay.verify 로 baseline 재현 PASS 확인).

후보의 '다음 commit 노드' = 그 후보 probe 의 best inner iterate 의 child
  (nmcs_inner 는 첫 최대값 iterate 의 nodes 를 돌려주고, nmcs_pass 는
   cur = bn[len(pfx)] 로 이동한다 -> child_node 와 같다)
기록된 continuation 길이 = 그 iterate 가 실제로 terminal 까지 간 결정 수
  line_length(= forced + tree) + rollout_steps  -  commit depth
  (ROLLNODE=1 이므로 rollout 구간도 그래프에 기록된 경로다)
새 state 는 만들지 않는다. 기록된 iterate 가 간 곳까지만 센다.
"""
import collections
import io
import json
import sys

sys.path.insert(0, "/opt/ygo/_diag/turnkill_trial/src")
import adaptive_replay as AR    # noqa: E402

OUT = "/opt/ygo/_diag/turnkill_trial/out"
NMCS_K = 2
KS = (1, 2, 5, 10, 20)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def mean(v):
    return sum(v) / float(len(v)) if v else 0.0


def select_trace(cands, th, node):
    """nmcs_pass 와 같은 선택. 어느 경로로 골랐는지도 돌려준다."""
    bv = max(c["v"] for c in cands)
    grp = [c for c in cands if c["v"] == bv]
    if len(grp) == 1:
        th.pop(node, None)
        return grp[0], "single"
    gset = dict((c["a"], c) for c in grp)
    h = th.get(node)
    if h is not None and h[0] in gset and h[1] > 0:
        h[1] -= 1
        return gset[h[0]], "hold"
    p = min(grp, key=lambda c: (c["vis0"], c["a"]))
    th[node] = [p["a"], NMCS_K - 1]
    return p, "min_vis0"


def analyse(seed):
    path = "%s/traj_big_%s.json" % (OUT, seed)
    ds = AR.build(path)
    raw = json.load(io.open(path, encoding="utf-8"))["candidate_trace"]
    rows = collections.defaultdict(list)
    for r in raw:
        rows[(r["node_id"], r["action"], r["probe_index"])].append(r)
    for k in rows:
        rows[k].sort(key=lambda r: r["inner_index"])

    def best_row(p):
        bi, bv = None, -1
        for r in rows[(p["node"], p["a"], p["j"])]:
            if r["iter_value"] > bv:
                bv, bi = r["iter_value"], r
        return bi

    def cdepth(p):
        rs = rows[(p["node"], p["a"], p["j"])]
        return rs[0]["forced_steps"] - 1 if rs else None

    def info(p, d):
        bi = best_row(p)
        rs = rows[(p["node"], p["a"], p["j"])]
        if bi is None:
            return None
        return {"a": p["a"], "v": p["v"], "vis0": p["vis0"],
                "term": bool(bi["terminal"]), "child": bi["child_node"],
                "cont": bi["line_length"] + bi["rollout_steps"] - d,
                "cont_max": max(r["line_length"] + r["rollout_steps"]
                                for r in rs) - d,
                "atk": bi["attack_count"], "final": ds["traj"][(p["node"],
                                                                p["a"])][-1]
                ["val_after"]}

    th = {}
    calls = collections.defaultdict(list)
    recs = []
    for c in ds["commits"]:
        cands = [{"a": p["a"], "v": p["v"], "vis0": p["vis0"], "p": p}
                 for p in c["cs"]]
        pick, how = select_trace(cands, th, c["node"])
        calls[c["call"]].append((c, pick, how))
    for cl, lst in calls.items():
        c, pick, how = max(lst, key=lambda z: z[0]["commit"])
        d = cdepth(pick["p"])
        sel = info(pick["p"], d)
        if sel is None or not sel["term"]:
            continue
        bv = max(p["v"] for p in c["cs"])
        tied = [p for p in c["cs"] if p["v"] == bv and p["a"] != pick["a"]]
        alts = [x for x in (info(p, d) for p in tied)
                if x is not None and not x["term"] and x["child"] is not None]
        alt = min(alts, key=lambda x: (x["vis0"], x["a"])) if alts else None
        recs.append({"seed": seed, "call": cl, "commit": c["commit"],
                     "node": c["node"], "depth": d, "ncand": c["ncand"],
                     "values": [p["v"] for p in c["cs"]],
                     "vis0s": [p["vis0"] for p in c["cs"]],
                     "sel": sel, "how": how, "bv": bv,
                     "n_alt": len(alts), "alt": alt})
    return ds, recs


def main():
    ALL = []
    DS = {}
    for seed in ("20260917", "12345", "777"):
        ds, recs = analyse(seed)
        DS[seed] = ds
        ALL += recs

    print("=" * 108)
    print("[106 TERMINAL COMMITS]  Sangen trajectory 3 seed")
    print("=" * 108)
    n = len(ALL)
    z = [r for r in ALL if r["bv"] == 0]
    pz = [r for r in ALL if r["bv"] > 0]
    wa = [r for r in ALL if r["n_alt"] > 0]
    how = collections.Counter(r["how"] for r in ALL)
    print("  count %d | all-zero %d (%.1f%%) | value>0 %d (%.1f%%)"
          % (n, len(z), pct(len(z), n), len(pz), pct(len(pz), n)))
    print("  nonterminal tied candidate 있음 %d (%.1f%%) | 없음 %d"
          % (len(wa), pct(len(wa), n), n - len(wa)))
    print("    all-zero 중 %d / value>0 중 %d"
          % (sum(1 for r in z if r["n_alt"]), sum(1 for r in pz if r["n_alt"])))
    print("  선택 경로 : %s  (hold = TIEHOLD 유지, min_vis0 = 새로 min(vis0,action))"
          % dict(how))
    print("  선택된 terminal 후보의 기록 continuation (sanity, 1 이어야 함) : %s"
          % dict(collections.Counter(r["sel"]["cont"] for r in ALL)))
    print("  commit depth 분포 : %s"
          % dict(sorted(collections.Counter(
              "<5" if r["depth"] < 5 else "5-19" if r["depth"] < 20
              else "20-39" if r["depth"] < 40 else "40+"
              for r in ALL).items())))
    print("  commit 당 후보 수 평균 %.1f, 동점 비terminal 후보 수 평균 %.2f"
          % (mean([r["ncand"] for r in ALL]), mean([r["n_alt"] for r in ALL])))
    sv = [r for r in ALL if r["alt"]]
    print("  선택된 후보 vis0 vs 대안 vis0 : 선택 평균 %.1f / 대안 평균 %.1f "
          "(선택 < 대안 %d, 같음 %d, 선택 > 대안 %d)"
          % (mean([r["sel"]["vis0"] for r in sv]),
             mean([r["alt"]["vis0"] for r in sv]),
             sum(1 for r in sv if r["sel"]["vis0"] < r["alt"]["vis0"]),
             sum(1 for r in sv if r["sel"]["vis0"] == r["alt"]["vis0"]),
             sum(1 for r in sv if r["sel"]["vis0"] > r["alt"]["vis0"])))

    print()
    print("  [COUNTERFACTUAL]  동점 + 비terminal + 최소 vis0 + action index 로 바꿨다면")
    print("    기록된 continuation (commit 노드 이후 terminal 까지 결정 수)")
    for lab, pop in (("전체 106 (대안 없으면 +0)", ALL),
                     ("대안이 있는 commit", wa)):
        conts = [r["alt"]["cont"] if r["alt"] else 0 for r in pop]
        cmx = [r["alt"]["cont_max"] if r["alt"] else 0 for r in pop]
        print("    %-26s n=%3d | %s | 평균 %.1f 중앙 %s 최대 %d"
              % (lab, len(pop), "  ".join("P(+%d) %5.1f%%" %
                                          (k, pct(sum(1 for x in conts
                                                      if x >= k), len(pop)))
                                          for k in KS),
                 mean(conts), sorted(conts)[len(conts) // 2] if conts else 0,
                 max(conts) if conts else 0))
        print("    %-26s       | (probe 내 최장 iterate 기준) 평균 %.1f, "
              "P(+10) %.1f%%, P(+20) %.1f%%"
              % ("", mean(cmx), pct(sum(1 for x in cmx if x >= 10), len(pop)),
                 pct(sum(1 for x in cmx if x >= 20), len(pop))))
    absd = [r["depth"] + r["alt"]["cont"] for r in wa]
    print("    대안 continuation 이 닿는 절대 depth : 평균 %.1f | >=30 %d | >=35 %d "
          "| >=40 %d (대안 %d 개 중)"
          % (mean(absd), sum(1 for x in absd if x >= 30),
             sum(1 for x in absd if x >= 35), sum(1 for x in absd if x >= 40),
             len(absd)))
    print("    대안의 best iterate value : >0 %d, 평균 %.0f | 그 (node,action) 의 "
          "최종 B : >=9000 %d, >=16000 %d, >=23000 %d"
          % (sum(1 for r in wa if r["alt"]["v"] > 0), mean([r["alt"]["v"]
                                                          for r in wa]),
             sum(1 for r in wa if r["alt"]["final"] >= 9000),
             sum(1 for r in wa if r["alt"]["final"] >= 16000),
             sum(1 for r in wa if r["alt"]["final"] >= 23000)))
    print("    비교: 기존 P(fp>=40) Sangen 1.1%% / 기존 선택(terminal) continuation 은 전부 +1")

    print()
    print("  [20260917 SANGEN]  root(node 0) 에서 끝난 call")
    ds = DS["20260917"]
    tr = ds["traj"][(0, 1)]
    fnz = next((p for p in tr if p["v"] > 0), None)
    print("    root a1 첫 nonzero probe: probe %d, call %d (v=%d)"
          % (fnz["j"], fnz["call"], fnz["v"]))
    for r in [x for x in ALL if x["seed"] == "20260917" and x["node"] == 0]:
        a1 = [p for p in ds["commits"] if p["commit"] == r["commit"]][0]
        p1 = [p for p in a1["cs"] if p["a"] == 1]
        a1v = p1[0]["v"] if p1 else None
        alt_s = ("a%d cont +%d v=%d 최종B %d" %
                 (r["alt"]["a"], r["alt"]["cont"], r["alt"]["v"],
                  r["alt"]["final"]) if r["alt"] else "없음")
        print("    call %-3d commit %-4d depth %d | 값 %s vis0 %s | 선택 a%d(terminal, %s)"
              " | a1 v=%s | 대안 %s"
              % (r["call"], r["commit"], r["depth"], r["values"], r["vis0s"],
                 r["sel"]["a"], r["how"], a1v, alt_s))
    others = [x for x in ALL if x["seed"] == "20260917" and x["node"] != 0]
    print("    root 이외에서 끝난 call %d 개 (commit depth %s)"
          % (len(others), sorted(r["depth"] for r in others)))

    print()
    print("  [상세]  대안 continuation 상위 10")
    for r in sorted(wa, key=lambda r: -r["alt"]["cont"])[:10]:
        print("    seed %-9s call %-3d depth %-3d node %-6d 값 %s | 선택 a%d(+1) -> 대안 "
              "a%d +%d (abs %d) v=%d 최종B %d atk %d"
              % (r["seed"], r["call"], r["depth"], r["node"], r["values"],
                 r["sel"]["a"], r["alt"]["a"], r["alt"]["cont"],
                 r["depth"] + r["alt"]["cont"], r["alt"]["v"],
                 r["alt"]["final"], r["alt"]["atk"]))

    json.dump(ALL, io.open("%s/deep/tie_cf.json" % OUT, "w", encoding="utf-8"),
              ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
