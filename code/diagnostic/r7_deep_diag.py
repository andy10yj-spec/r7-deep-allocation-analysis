# -*- coding: utf-8 -*-
"""[DEEP EXPANSION DIAGNOSTIC]  깊은 노드에서 무엇이 일어나는지 계측만 한다.

tk_r7.py 는 수정하지 않는다. 프로세스 안에서 읽기 전용 wrapper 만 단다.
래퍼는 RNG 를 소비하지 않고, 인자/반환값을 바꾸지 않고, 분기를 바꾸지 않는다.
탐색은 tk_r7.main() 의 NMCS 루프와 같은 production nmcs_pass 를 그대로 쓴다.

감싸는 함수 (모두 module global 로 호출되므로 교체가 반영된다)
  iterate      : iterate 단위 종료 사유, forced/tree/rollout step
  pick_action  : tree phase 선택 (확장 is_new / None)
  rollout      : rollout 구간 표시, 시작 depth, step 수
  get_node     : 새 노드 생성과 생성 출처(forced/tree/rollout)
  get_edge     : 새 간선 vs 기존 간선 재방문
  backup       : node.value 상승
  nmcs_inner   : continuation 길이와 종료 사유, probe 당 새 노드/간선
"""
import collections
import hashlib
import io
import json
import os
import sys
import time

os.environ.setdefault("TK_ITERS", "0")
os.environ.setdefault("TK_FROLL", "0")
os.environ.setdefault("TK_PROP", "0")
os.environ.setdefault("TK_OPTSEL", "0")
os.environ.setdefault("TK7_ORDER", "0")
os.environ.setdefault("TK7_NMCS", "2")
os.environ.setdefault("TK7_NMCS_K", "2")
sys.path.insert(0, "/opt/ygo/_diag/turnkill_trial/src")
import importlib                                      # noqa: E402
# 계측 대상 모듈. 기본은 production tk_r7. prototype 비교 시 DD_MOD 로 바꾼다.
R7 = importlib.import_module(os.environ.get("DD_MOD", "tk_r7"))

SEED = os.environ.get("TK_RSEED", "12345")
HAND = os.environ.get("TK_HAND", "91810826")
CLK = time.perf_counter

_o = {k: getattr(R7, k) for k in ("iterate", "pick_action", "rollout",
                                  "get_node", "get_edge", "backup",
                                  "nmcs_inner")}

PH = ["forced"]            # 현재 단계: forced / tree / rollout
IT = {}                    # 현재 iterate 컨텍스트
INNER = []                 # nmcs_inner 컨텍스트 스택
ITROWS = []                # iterate 행
INROWS = []                # nmcs_inner 행
NODE_SRC = {}              # node -> 생성 출처
PICKCALL = collections.Counter()   # node -> pick_action 호출 수
PICKNEW = collections.Counter()    # node -> 그중 확장(is_new)
TREEPASS = collections.Counter()   # node -> tree/forced 경로에 등장한 수
EDGE_NEW = collections.Counter()   # (출처, band) -> 새 간선
EDGE_OLD = collections.Counter()   # (출처, band) -> 기존 간선 재방문
RISE = collections.Counter()       # node.depth band -> value 상승 횟수
ROLLSTART = collections.Counter()  # band -> rollout 시작


def band(d):
    if d is None:
        return "?"
    if d < 20:
        return "00-19"
    if d >= 70:
        return "70+"
    lo = (d // 10) * 10
    return "%02d-%02d" % (lo, lo + 9)


def band4(d):
    if d < 20:
        return "00-19"
    if d < 40:
        return "20-39"
    if d < 60:
        return "40-59"
    return "60+"


# ---------------------------------------------------------------- wrappers
def w_get_node(obs, prompt, labels, depth, wit, info=None, raw=0):
    n0 = len(R7.NODES)
    i = _o["get_node"](obs, prompt, labels, depth, wit, info, raw)
    if len(R7.NODES) > n0:
        NODE_SRC[i] = PH[0]
        IT["newnode"] = IT.get("newnode", 0) + 1
        for c in INNER:
            c["newnode"] += 1
    return i


def w_get_edge(parent, action, child, ctx):
    existed = (parent, action, child) in R7.EIDX
    e = _o["get_edge"](parent, action, child, ctx)
    b = band(R7.NODES[parent].depth)
    if existed:
        EDGE_OLD[(PH[0], b)] += 1
        for c in INNER:
            c["oldedge"] += 1
    else:
        EDGE_NEW[(PH[0], b)] += 1
        IT["newedge"] = IT.get("newedge", 0) + 1
        for c in INNER:
            c["newedge"] += 1
    return e


def w_pick_action(nd):
    PH[0] = "tree"
    a, is_new = _o["pick_action"](nd)
    PICKCALL[nd.id] += 1
    if a is None:
        IT["none"] = True
    elif is_new:
        PICKNEW[nd.id] += 1
        IT["new"] = True
    return a, is_new


def w_rollout(g, info, depth, node, path):
    prev = PH[0]
    PH[0] = "rollout"
    ROLLSTART[band(depth)] += 1
    s0 = R7.STAT["rollout_step"]
    try:
        r = _o["rollout"](g, info, depth, node, path)
    finally:
        PH[0] = prev
    IT["roll"] = R7.STAT["rollout_step"] - s0
    IT["rolled"] = True
    return r


def w_backup(path, val):
    for eid in path:
        nd = R7.NODES[R7.EDGES[eid].parent]
        if val > nd.value:
            RISE[band(nd.depth)] += 1
    return _o["backup"](path, val)


def w_iterate(fpath=None):
    fp = tuple(fpath) if fpath else ()
    PH[0] = "forced"
    IT.clear()
    se0 = R7.STAT.get("send_error", 0)
    oo0 = R7.STAT.get("oor_skip", 0)
    r = _o["iterate"](fpath=fpath) if fpath is not None \
        else _o["iterate"]()
    best, dep, picks, pnodes, natk = r
    for nid in pnodes:
        TREEPASS[nid] += 1
    last = R7.NODES[pnodes[-1]] if pnodes else None
    if not pnodes:
        why = "exception"
    elif R7.STAT.get("send_error", 0) > se0:
        why = "exception"
    elif R7.STAT.get("oor_skip", 0) > oo0:
        why = "oor"
    elif last.terminal:
        why = "terminal"
    elif IT.get("new"):
        why = "new expansion"
    elif IT.get("none"):
        why = "pick_action=None"
    elif len(picks) >= R7.MAXDEC:
        why = "MAXDEC"
    elif len(picks) < len(fp):
        why = "prefix exhausted"
    else:
        why = "other"
    roll = IT.get("roll", 0)
    ITROWS.append({
        "fp": len(fp), "picks": len(picks),
        "tree": max(0, len(picks) - len(fp)),
        "forced": min(len(picks), len(fp)),
        "roll": roll, "why": why, "rolled": bool(IT.get("rolled")),
        "best": best, "natk": natk,
        "newnode": IT.get("newnode", 0), "newedge": IT.get("newedge", 0),
        "inner": bool(INNER),
    })
    if INNER:
        c = INNER[-1]
        c["iters"].append((len(fp), len(pnodes), len(picks), roll, why,
                           (R7.NODES[pnodes[len(fp)]].terminal
                            or not R7.NODES[pnodes[len(fp)]].acts)
                           if len(pnodes) > len(fp) else None))
    return r


def w_nmcs_inner(pfx, cap):
    c = {"start": len(pfx), "iters": [], "newnode": 0, "newedge": 0,
         "oldedge": 0, "s0": R7.STAT["engine_session"]}
    INNER.append(c)
    try:
        v, nodes = _o["nmcs_inner"](pfx, cap)
    finally:
        INNER.pop()
    its = c["iters"]
    if not its:
        why = "cap"
    else:
        fpl, pl, _pk, _rl, _w, nxt_term = its[-1]
        if pl <= fpl:
            why = "prefix exhausted"
        elif nxt_term:
            why = "terminal"
        elif len(its) >= R7.NMCS_INNER:
            why = "inner budget"
        elif R7.STAT["engine_session"] >= cap:
            why = "cap"
        else:
            why = "other"
    # continuation = 후보 prefix 너머로 라인이 실제로 몇 결정 갔는가
    cont = [max(0, pk + rl - c["start"]) for (_f, _p, pk, rl, _w, _t) in its]
    INROWS.append({
        "start": c["start"], "n": len(its), "why": why,
        "p2grow": (its[-1][0] - c["start"]) if its else 0,
        "cont_max": max(cont) if cont else 0,
        "cont_first": cont[0] if cont else 0,
        "tree_max": max([max(0, pk - f) for (f, _p, pk, _r, _w, _t) in its]
                        or [0]),
        "inner_why": [w for (_f, _p, _pk, _rl, w, _t) in its],
        "newnode": c["newnode"], "newedge": c["newedge"],
        "oldedge": c["oldedge"], "v": v,
    })
    return v, nodes


def install():
    R7.iterate = w_iterate
    R7.pick_action = w_pick_action
    R7.rollout = w_rollout
    R7.get_node = w_get_node
    R7.get_edge = w_get_edge
    R7.backup = w_backup
    R7.nmcs_inner = w_nmcs_inner


# ---------------------------------------------------------------- analysis
def node_row(n):
    ncand = n.prompt[1] if n.prompt else 0
    exp = sum(1 for a in n.acts if n.acts[a][2])
    unt = len(n.untried)
    unt_edge = sum(1 for a in n.untried if n.acts.get(a, [0, 0, []])[2])
    kids = set()
    multi = 0
    for a in n.acts:
        ks = set()
        for e in n.acts[a][2]:
            kids.add(R7.EDGES[e].child)
            ks.add(R7.EDGES[e].child)
        if len(ks) >= 2:
            multi += 1
    return {"id": n.id, "depth": n.depth, "visits": n.visits,
            # 같은 state 를 run 사이에서 찾기 위한 observation signature 해시.
            # obs 는 이미 masking 된 signature 이므로 새 정보가 들어가지 않는다.
            "oh": hashlib.md5(repr(n.obs).encode()).hexdigest()[:16],
            "incoming": len(n.incoming), "multi": multi,
            "value": n.value, "raw": n.raw, "terminal": n.terminal,
            "ncand": ncand, "untried": unt, "untried_edge": unt_edge,
            "expanded": exp, "children": len(kids),
            "pick": PICKCALL[n.id], "picknew": PICKNEW[n.id],
            "treepass": TREEPASS[n.id], "src": NODE_SRC.get(n.id, "?")}


def best_line():
    """root 에서 edge.value 가 가장 큰 간선을 따라간다 (동률은 visits)."""
    line = []
    cur = 0
    seen = set()
    atk = 0
    while cur not in seen:
        seen.add(cur)
        n = R7.NODES[cur]
        row = node_row(n)
        row["atk_so_far"] = atk
        line.append(row)
        es = [R7.EDGES[e] for a in n.acts for e in n.acts[a][2]]
        if not es:
            break
        e = max(es, key=lambda x: (x.value, x.visits, -x.id))
        lab = n.labels[e.action] if e.action < len(n.labels) else ""
        if lab.startswith("공격"):
            atk += 1
        row["next_action"] = e.action
        row["next_label_is_attack"] = lab.startswith("공격")
        row["edge_value"] = e.value
        row["edge_visits"] = e.visits
        cur = e.child
    return line


def graph_hash():
    nr = [[n.depth, n.visits, n.value, 1 if n.terminal else 0,
           (n.prompt[1] if n.prompt else 0), len(n.incoming), n.raw]
          for n in R7.NODES]
    er = [[e.parent, e.action, e.child, e.visits, e.value] for e in R7.EDGES]
    return hashlib.md5(json.dumps({"n": nr, "e": er},
                                  sort_keys=True).encode()).hexdigest()


def main():
    iters = int(os.environ.get("DD_ITERS", "5000"))
    install()
    t0 = CLK()
    while R7.STAT["engine_session"] < iters:
        b0 = R7.STAT["engine_session"]
        R7.nmcs_pass(R7.NMCS_LEVEL, iters)
        if R7.STAT["engine_session"] == b0:
            R7.iterate()
    wall = CLK() - t0
    rows = [node_row(n) for n in R7.NODES]
    line = best_line()
    st = R7.STAT
    out = {
        "seed": SEED, "hand": HAND, "iters": iters, "wall": wall,
        "best": max([n.value for n in R7.NODES], default=0),
        "maxatk": st.get("maxatk", 0),
        "maxdepth": max([n.depth for n in R7.NODES], default=0),
        "graph_hash": graph_hash(),
        "nodes": rows, "line": line,
        "it": ITROWS, "inner": INROWS,
        "edge_new": {"%s|%s" % k: v for k, v in EDGE_NEW.items()},
        "edge_old": {"%s|%s" % k: v for k, v in EDGE_OLD.items()},
        "rise": dict(RISE), "rollstart": dict(ROLLSTART),
        "stat": dict(st),
        "mod": R7.__name__, "unique": len(R7.NIDX),
        "root": ([[a, R7.NODES[0].acts[a][0], R7.NODES[0].acts[a][1]]
                  for a in sorted(R7.NODES[0].acts)] if R7.NODES else []),
        "promo_on": bool(getattr(R7, "PROMO", False)),
        "promo_d": {str(k): v for k, v in
                    getattr(R7, "PROMO_D", {}).items()},
        "promo_per": {str(k): v for k, v in
                      getattr(R7, "PROMO_PER", {}).items()},
    }
    p = os.environ.get("DD_OUT", "")
    if p:
        json.dump(out, io.open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print("seed=%s hand=%s best=%d maxatk=%d maxdepth=%d nodes=%d edges=%d "
          "hash=%s wall=%.0fs" % (SEED, HAND, out["best"], out["maxatk"],
                                  out["maxdepth"], len(R7.NODES),
                                  len(R7.EDGES), out["graph_hash"], wall))


if __name__ == "__main__":
    main()
