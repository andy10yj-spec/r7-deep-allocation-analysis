# -*- coding: utf-8 -*-
"""NMCS call 을 끝낸 commit 은 무엇을 골랐는가. offline 전용 (trajectory JSON)."""
import collections
import io
import json
import sys

sys.path.insert(0, "/opt/ygo/_diag/turnkill_trial/src")
import adaptive_replay as AR    # noqa: E402

OUT = "/opt/ygo/_diag/turnkill_trial/out"
DEEPF = {"20260917": "promo/off_sangen_20260917.json",
         "12345": "promo/off_sangen_12345.json",
         "777": "promo/off_sangen_777.json"}


def mean(v):
    return sum(v) / float(len(v)) if v else 0.0


def pct(a, b):
    return 100.0 * a / b if b else 0.0


print("=" * 104)
print("[NMCS CALL 종료 commit 분석]  Sangen trajectory 3 seed")
print("=" * 104)
tot = collections.Counter()
for seed in ("20260917", "12345", "777"):
    path = "%s/traj_big_%s.json" % (OUT, seed)
    ds = AR.build(path)
    raw = json.load(io.open(path, encoding="utf-8"))["candidate_trace"]
    rows = collections.defaultdict(list)
    for r in raw:
        rows[(r["node_id"], r["action"], r["probe_index"])].append(r)
    sel, ok, bad, _ = AR.verify(ds)

    def best_row(p):
        rs = sorted(rows[(p["node"], p["a"], p["j"])],
                    key=lambda r: r["inner_index"])
        bi, bv = None, -1
        for r in rs:
            if r["iter_value"] > bv:
                bv, bi = r["iter_value"], r
        return bi

    calls = collections.defaultdict(list)
    for c in ds["commits"]:
        calls[c["call"]].append(c)
    ends = []
    allterm = 0
    for c in ds["commits"]:
        p = [x for x in c["cs"] if x["a"] == sel[c["commit"]]][0]
        bi = best_row(p)
        if bi is not None and bi["terminal"]:
            allterm += 1
    for cl, cs in calls.items():
        c = max(cs, key=lambda z: z["commit"])
        a = sel[c["commit"]]
        p = [x for x in c["cs"] if x["a"] == a][0]
        bi = best_row(p)
        if bi is None or not bi["terminal"]:
            continue
        vs = [x["v"] for x in c["cs"]]
        bv = max(vs)
        grp = [x for x in c["cs"] if x["v"] == bv]
        # 다른 후보 중 terminal 이 아닌 child 로 이어진 것
        nonterm = 0
        for x in c["cs"]:
            if x["a"] == a:
                continue
            b2 = best_row(x)
            if b2 is not None and not b2["terminal"]:
                nonterm += 1
        dep = (rows[(p["node"], p["a"], p["j"])][0]["forced_steps"] - 1
               if rows[(p["node"], p["a"], p["j"])] else -1)
        ends.append({"dep": dep, "n": len(c["cs"]), "tie": len(grp) >= 2,
                     "v": p["v"], "bv": bv, "zero": bv == 0,
                     "nonterm": nonterm, "line": bi["line_length"]})
    n = len(ends)
    print("  seed %-9s 재현 %s | commit %d 중 고른 action 의 다음 노드가 terminal: %d (%.1f%%)"
          % (seed, "PASS" if bad == 0 else "FAIL", len(ds["commits"]),
             allterm, pct(allterm, len(ds["commits"]))))
    print("    terminal 로 끝난 call %d 개의 마지막 commit:" % n)
    print("      commit depth 평균 %.1f (분포 %s)"
          % (mean([e["dep"] for e in ends]),
             dict(sorted(collections.Counter(
                 ("<5" if e["dep"] < 5 else "5-19" if e["dep"] < 20
                  else "20-39" if e["dep"] < 40 else "40+")
                 for e in ends).items()))))
    print("      후보 값 전부 0 : %d (%.1f%%) | 최고값 동점 : %d (%.1f%%) | 고른 후보가 "
          "단독 최고 : %d"
          % (sum(e["zero"] for e in ends), pct(sum(e["zero"] for e in ends), n),
             sum(e["tie"] for e in ends), pct(sum(e["tie"] for e in ends), n),
             sum(1 for e in ends if not e["tie"])))
    print("      다른 후보 중 terminal 이 아닌 곳으로 이어진 것이 있었던 경우 : %d (%.1f%%)"
          % (sum(1 for e in ends if e["nonterm"] > 0),
             pct(sum(1 for e in ends if e["nonterm"] > 0), n)))
    print("      고른 후보 v 평균 %.0f, 그 commit 최고 v 평균 %.0f"
          % (mean([e["v"] for e in ends]), mean([e["bv"] for e in ends])))
    for e in ends:
        tot["ends"] += 1
        tot["zero"] += e["zero"]
        tot["tie"] += e["tie"]
        tot["nonterm"] += 1 if e["nonterm"] > 0 else 0
        tot["shallow"] += 1 if e["dep"] < 20 else 0

    # line>=40 iterate 의 fp 분포 (deep JSON)
    r = json.load(io.open("%s/%s" % (OUT, DEEPF[seed]), encoding="utf-8"))
    le = [x for x in r["it"] if x["picks"] + x["roll"] >= 40]
    fpd = collections.Counter("<20" if x["fp"] < 20 else "20-29" if x["fp"] < 30
                              else "30-39" if x["fp"] < 40 else "40+"
                              for x in le)
    print("    line end>=40 인 iterate %d 개의 forced prefix : %s"
          % (len(le), dict(sorted(fpd.items()))))

print()
print("  합계: terminal 로 끝난 call %d | 마지막 commit depth<20 %d (%.1f%%) | 후보 값 "
      "전부 0 %d (%.1f%%) | 동점 %d (%.1f%%) | 다른 후보는 비terminal %d (%.1f%%)"
      % (tot["ends"], tot["shallow"], pct(tot["shallow"], tot["ends"]),
         tot["zero"], pct(tot["zero"], tot["ends"]), tot["tie"],
         pct(tot["tie"], tot["ends"]), tot["nonterm"],
         pct(tot["nonterm"], tot["ends"])))
