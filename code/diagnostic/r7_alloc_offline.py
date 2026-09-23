# -*- coding: utf-8 -*-
"""[DEEP ALLOCATION BOTTLENECK]  기존 JSON 만으로 세 축을 분리한다. engine 0.

입력 (모두 production tk_r7 baseline 과 같은 탐색)
  out/promo/off_*.json      6 run  (graph_hash 로 production 과 동일 확인됨)
  out/deep/sangen_8888.json 1 run
  out/traj_big_*.json       Sangen 3 seed  (commit 단위 기록)

iterate 행: fp(forced prefix 길이), picks(= fp + tree), tree, roll, why
nmcs_inner 행: start(= commit depth + 1), n, why, ...
node 행: depth, visits, ncand, untried, untried_edge, pick, picknew, treepass, src
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, "/opt/ygo/_diag/turnkill_trial/src")
import adaptive_replay as AR    # noqa: E402

OUT = "/opt/ygo/_diag/turnkill_trial/out"
DEEP = [("sangen", "20260917", "promo/off_sangen_20260917.json"),
        ("sangen", "12345", "promo/off_sangen_12345.json"),
        ("sangen", "777", "promo/off_sangen_777.json"),
        ("sangen", "8888", "deep/sangen_8888.json"),
        ("chundra", "20260917", "promo/off_chundra_20260917.json"),
        ("chundra", "12345", "promo/off_chundra_12345.json"),
        ("chundra", "777", "promo/off_chundra_777.json")]
TRAJ = ["20260917", "12345", "777"]


def b3(d):
    return "<20" if d < 20 else ("20-39" if d < 40 else "40+")


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def mean(v):
    return sum(v) / float(len(v)) if v else 0.0


def q(v, p):
    if not v:
        return 0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p))]


R = {}
for tag, seed, f in DEEP:
    R[(tag, seed)] = json.load(io.open(os.path.join(OUT, f), encoding="utf-8"))

print("=" * 110)
print("[DEEP ALLOCATION BOTTLENECK]   offline, engine 0")
print("=" * 110)

# ------------------------------------------------------------ 1. forced prefix
print()
print("  [FORCED PREFIX]  iterate 의 forced prefix 길이 (fp)")
KS = (5, 10, 20, 30, 35, 40)
for deck in ("sangen", "chundra"):
    fps = []
    pk = []
    end = []
    for (tag, seed), r in R.items():
        if tag != deck:
            continue
        fps += [x["fp"] for x in r["it"]]
        pk += [x["picks"] for x in r["it"]]
        end += [x["picks"] + x["roll"] for x in r["it"]]
    n = float(len(fps))
    hist = collections.Counter()
    for x in fps:
        if x <= 10:
            hist[str(x)] += 1
        elif x < 20:
            hist["11-19"] += 1
        elif x < 30:
            hist["20-29"] += 1
        elif x < 40:
            hist["30-39"] += 1
        else:
            hist["40+"] += 1
    keys = [str(i) for i in range(11)] + ["11-19", "20-29", "30-39", "40+"]
    print("    %s (%d iterate)" % (deck, len(fps)))
    print("      분포 : " + " ".join("%s:%.1f%%" % (k, pct(hist[k], n))
                                   for k in keys))
    print("      P(fp>=k)        " + "  ".join("k=%d %.2f%%" %
                                            (k, pct(sum(1 for x in fps
                                                        if x >= k), n))
                                            for k in KS))
    print("      P(frontier>=k)  " + "  ".join("k=%d %.2f%%" %
                                            (k, pct(sum(1 for x in pk
                                                        if x >= k), n))
                                            for k in KS))
    print("      P(line end>=k)  " + "  ".join("k=%d %.2f%%" %
                                            (k, pct(sum(1 for x in end
                                                        if x >= k), n))
                                            for k in KS))
    print("      fp 중앙 %d / 90%% %d / 최대 %d" % (q(fps, .5), q(fps, .9),
                                               max(fps)))

# ------------------------------------------------------------ 2. NMCS commit
print()
print("  [NMCS COMMIT]  nmcs_inner 시작 depth = commit depth (start-1)")
for deck in ("sangen", "chundra"):
    c = collections.Counter()
    it = collections.Counter()
    for (tag, seed), r in R.items():
        if tag != deck:
            continue
        for x in r["inner"]:
            b = b3(max(0, x["start"] - 1))
            c[b] += 1
            it[b] += x["n"]
    tot = sum(c.values())
    print("    %-8s probe  <20 %d (%.1f%%) / 20-39 %d (%.1f%%) / 40+ %d (%.1f%%)"
          "   iterate  %d / %d / %d"
          % (deck, c["<20"], pct(c["<20"], tot), c["20-39"],
             pct(c["20-39"], tot), c["40+"], pct(c["40+"], tot),
             it["<20"], it["20-39"], it["40+"]))

print()
print("  [NMCS CALL]  Sangen trajectory: call 당 commit 수와 도달한 최대 commit depth")
for seed in TRAJ:
    path = "%s/traj_big_%s.json" % (OUT, seed)
    ds = AR.build(path)
    raw = json.load(io.open(path, encoding="utf-8"))["candidate_trace"]
    first = {}
    term = {}
    for r in raw:
        k = (r["node_id"], r["action"], r["probe_index"])
        if r["inner_index"] == 0:
            first[k] = r["forced_steps"]
    sel, ok, bad, _ = AR.verify(ds)
    calls = collections.defaultdict(list)
    cand = []
    for c in ds["commits"]:
        p0 = c["cs"][0]
        dep = first.get((p0["node"], p0["a"], p0["j"]), 0) - 1
        calls[c["call"]].append((c["commit"], dep, c))
        cand.append((dep, len(c["cs"]), c["ncand"]))
    mx = [max(d for _, d, _ in v) for v in calls.values()]
    ncm = [len(v) for v in calls.values()]
    # call 의 마지막 commit 다음이 끊긴 이유
    endwhy = collections.Counter()
    byc = dict((c["commit"], c) for c in ds["commits"])
    for cl, v in calls.items():
        cm, dep, c = max(v, key=lambda z: z[0])
        a = sel[cm]
        p = [x for x in c["cs"] if x["a"] == a][0]
        rows = [r for r in raw if r["node_id"] == p["node"]
                and r["action"] == a and r["probe_index"] == p["j"]]
        bi = None
        bv = -1
        for r in sorted(rows, key=lambda r: r["inner_index"]):
            if r["iter_value"] > bv:
                bv, bi = r["iter_value"], r
        if bi is None:
            endwhy["기록 없음"] += 1
        elif bi["child_node"] is None:
            endwhy["best line 이 prefix 에서 끊김"] += 1
        elif bi["terminal"]:
            endwhy["다음 노드 terminal"] += 1
        else:
            endwhy["예산 소진/기타"] += 1
    db = collections.Counter(b3(d) for d, _, _ in cand)
    print("    seed %-9s baseline 재현 %s | call %d, call 당 commit 평균 %.1f "
          "(중앙 %d, 최대 %d)"
          % (seed, "PASS" if bad == 0 else "FAIL", len(calls), mean(ncm),
             q(ncm, .5), max(ncm)))
    print("      call 당 최대 commit depth : 평균 %.1f 중앙 %d 90%% %d 최대 %d"
          % (mean(mx), q(mx, .5), q(mx, .9), max(mx)))
    print("      commit depth : <20 %d / 20-39 %d / 40+ %d   "
          "commit 당 후보 평균 %.1f, probe 평균 %.1f"
          % (db["<20"], db["20-39"], db["40+"],
             mean([n for _, _, n in cand]), mean([p for _, p, _ in cand])))
    print("      call 종료 : %s" % dict(endwhy))

# ------------------------------------------------------------ 3. node table
print()
print("  [DEEP NODE BOTTLENECK]  비종단 노드, 후보>=2")
for scope in ("sangen", "chundra", ("sangen", "20260917")):
    agg = collections.OrderedDict((b, collections.Counter())
                                  for b in ("<20", "20-39", "40+"))
    vis = collections.defaultdict(list)
    fe = collections.Counter()
    fpb = collections.defaultdict(list)
    for key, r in R.items():
        if isinstance(scope, tuple):
            if key != scope:
                continue
        elif key[0] != scope:
            continue
        for n in r["nodes"]:
            if n["terminal"] or n["ncand"] < 2:
                continue
            a = agg[b3(n["depth"])]
            a["nodes"] += 1
            a["arrive"] += n["treepass"]
            a["pick"] += n["pick"]
            a["picknodes"] += 1 if n["pick"] > 0 else 0
            a["expand"] += n["picknew"]
            a["roll"] += max(0, n["visits"] - n["treepass"])
            a["untried"] += n["untried"]
            a["cand"] += n["ncand"]
            a["unt_pos"] += 1 if n["untried"] > 0 else 0
            a["inc2"] += 1 if n.get("incoming", 0) >= 2 else 0
            vis[b3(n["depth"])].append(n["visits"])
        for x in r["it"]:
            if x["why"] == "new expansion" and x["picks"] >= 1:
                fe[b3(x["picks"] - 1)] += 1
            fpb[b3(max(0, x["picks"] - 1))].append(x["fp"])
    lab = scope if isinstance(scope, str) else "%s %s" % scope
    print("    %s" % lab)
    rows = [("node count", "nodes"), ("iterate arrival (forced+tree 경로)",
                                      "arrive"),
            ("tree pick (pick_action 호출)", "pick"),
            ("tree pick 받은 노드", "picknodes"),
            ("tree expansion (untried pop)", "expand"),
            ("rollout arrival (visits-arrival, 추정)", "roll"),
            ("untried remaining", "untried"), ("untried>0 노드", "unt_pos"),
            ("incoming>=2 노드", "inc2")]
    print("      %-40s %12s %12s %12s" % ("metric", "depth<20", "depth20-39",
                                        "depth40+"))
    for name, k in rows:
        print("      %-40s %12d %12d %12d"
              % (name, agg["<20"][k], agg["20-39"][k], agg["40+"][k]))
    print("      %-40s %12.1f %12.1f %12.1f"
          % ("mean visits", mean(vis["<20"]), mean(vis["20-39"]),
             mean(vis["40+"])))
    print("      %-40s %12d %12d %12d"
          % ("first-new-expansion (그 depth 에서 break)", fe["<20"],
             fe["20-39"], fe["40+"]))
    print("      %-40s %12.1f %12.1f %12.1f"
          % ("forced prefix (frontier 가 그 band 인 iterate)", mean(fpb["<20"]),
             mean(fpb["20-39"]), mean(fpb["40+"])))

# ------------------------------------------------------------ 4. A/B/C/D @40
print()
print("  [DEPTH>=40 도달 단계]  run 별, iterate 5,000 기준")
print("    %-17s %8s | %8s %8s %8s | %-28s | %s"
      % ("run", "node>=40", "line>=40", "front>=40", "fp>=40",
         "fp>=40 iterate: tree 결정/종료", "front>=40 인 iterate 의 fp"))
for key, r in R.items():
    ns = [n for n in r["nodes"] if n["depth"] >= 40 and not n["terminal"]]
    its = r["it"]
    le = [x for x in its if x["picks"] + x["roll"] >= 40]
    fr = [x for x in its if x["picks"] >= 40]
    fp = [x for x in its if x["fp"] >= 40]
    why = collections.Counter(x["why"] for x in fp)
    tr = mean([x["tree"] for x in fp])
    fr_fp = [x["fp"] for x in fr]
    print("    %-17s %8d | %8d %8d %8d | tree %.2f %-18s | fp<35 %d / 35-39 %d "
          "/ 40+ %d"
          % ("%s_%s" % key, len(ns), len(le), len(fr), len(fp), tr,
             dict(why.most_common(2)), sum(1 for x in fr_fp if x < 35),
             sum(1 for x in fr_fp if 35 <= x < 40),
             sum(1 for x in fr_fp if x >= 40)))

# ------------------------------------------------------------ 5. selection
print()
print("  [SELECTION REASON]  pick_action 호출 분해")
for scope in ("sangen", "chundra"):
    agg = collections.OrderedDict((b, collections.Counter())
                                  for b in ("<20", "20-39", "40+"))
    ex = eo = 0
    for key, r in R.items():
        if key[0] != scope:
            continue
        ex += r["stat"].get("sel_explore", 0)
        eo += r["stat"].get("sel_exploit", 0)
        for n in r["nodes"]:
            a = agg[b3(n["depth"])]
            a["untried"] += n["picknew"]
            a["other"] += n["pick"] - n["picknew"]
    print("    %s" % scope)
    for b, a in agg.items():
        t = a["untried"] + a["other"]
        print("      %-6s pick %6d | UNTRIED %6d (%5.1f%%) | 소진 후 선택 %6d (%5.1f%%)"
              % (b, t, a["untried"], pct(a["untried"], t), a["other"],
                 pct(a["other"], t)))
    print("      소진 후 선택의 전체 분해(STAT, depth 구분 불가): LEAST_VISITED %d "
          "(%.1f%%) / MAX_VALUE %d (%.1f%%)"
          % (ex, pct(ex, ex + eo), eo, pct(eo, ex + eo)))

# ------------------------------------------------------------ 6. first expansion
print()
print("  [FIRST NEW EXPANSION]  fp band 별: prefix 뒤 몇 결정 만에 break 했는가")
for scope in ("sangen", "chundra"):
    print("    %s" % scope)
    grp = collections.defaultdict(list)
    for key, r in R.items():
        if key[0] != scope:
            continue
        for x in r["it"]:
            grp[b3(x["fp"])].append(x)
    for b in ("<20", "20-39", "40+"):
        xs = grp[b]
        ne = [x for x in xs if x["why"] == "new expansion"]
        te = [x for x in xs if x["why"] == "terminal"]
        print("      fp %-6s iterate %6d | new expansion %5.1f%% (prefix 뒤 평균 "
              "%.2f 결정, 절대 depth 평균 %.1f) | terminal %5.1f%%"
              % (b, len(xs), pct(len(ne), len(xs)),
                 mean([x["tree"] for x in ne]),
                 mean([x["picks"] for x in ne]), pct(len(te), len(xs))))

# ------------------------------------------------------------ 7. counterfactual
print()
print("  [COUNTERFACTUAL A]  first-new-expansion break 제거 (prefix 그대로)")
print("    제어 흐름상: break 가 없으면 pick_action 이 terminal 까지 계속 불린다.")
print("    엔진이 가는 depth 는 그대로(라인은 원래 terminal 까지 간다), 바뀌는 것은")
print("    rollout 구간이 tree 규칙(untried pop / 교대 선택)으로 바뀌는 것뿐이다.")
for key, r in R.items():
    its = r["it"]
    tr0 = sum(x["tree"] for x in its)
    ro = sum(x["roll"] for x in its)
    fr = sum(1 for x in its if x["picks"] >= 40)
    le = sum(1 for x in its if x["picks"] + x["roll"] >= 40)
    fr20 = sum(1 for x in its if x["picks"] >= 20)
    le20 = sum(1 for x in its if x["picks"] + x["roll"] >= 20)
    print("    %-17s tree 결정 %6d -> %6d | tree 가 depth>=20 닿는 iterate %5d -> "
          "%5d | depth>=40 %4d -> %4d"
          % ("%s_%s" % key, tr0, tr0 + ro, fr20, le20, fr, le))
print()
print("    기존 기록 간선만 따라갈 수 있는 양 (추정, 최종 그래프 기준)")
print("    break 뒤 child 에서 untried 를 pop 할 때 그 action 에 기록된 간선이 있을 확률")
print("    = sum(untried 중 간선 있음) / sum(untried).  경로별 정확 재현은 NOT RECOVERABLE")
print("    (iterate 행에 경로 노드 id 가 없고 JSON 에 간선 목록이 저장되지 않았다)")
for scope in ("sangen", "chundra"):
    for b in ("<20", "20-39", "40+"):
        ut = ue = 0
        for key, r in R.items():
            if key[0] != scope:
                continue
            for n in r["nodes"]:
                if n["terminal"] or n["ncand"] < 2 or b3(n["depth"]) != b:
                    continue
                ut += n["untried"]
                ue += n["untried_edge"]
        p = ue / float(ut) if ut else 0.0
        print("      %-8s %-6s p = %.3f  -> 기록 간선으로 이어질 기대 추가 결정 "
              "p/(1-p) = %.2f" % (scope, b, p, p / (1 - p) if p < 1 else 0.0))

print()
print("  [COUNTERFACTUAL B]  tree 규칙 그대로, prefix 만 더 깊게 준다면")
for key, r in R.items():
    ns = [n for n in r["nodes"] if n["depth"] >= 40 and not n["terminal"]
          and n["ncand"] >= 2]
    its = [x for x in r["it"] if x["fp"] >= 40]
    per = mean([x["tree"] for x in its]) if its else 0.0
    hv = sum(1 for n in ns if n["value"] >= 9000)
    print("    %-17s depth>=40 결정노드 %5d, 후보 %5d, untried %5d, value>=9000 노드 %4d "
          "| 현재 fp>=40 iterate 당 tree 결정 %.2f"
          % ("%s_%s" % key, len(ns), sum(n["ncand"] for n in ns),
             sum(n["untried"] for n in ns), hv, per))
