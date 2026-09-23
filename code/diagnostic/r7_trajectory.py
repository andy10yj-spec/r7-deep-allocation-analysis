# -*- coding: utf-8 -*-
"""[TRAJECTORY INSTRUMENTATION]  후보별 value trajectory 계측.

production 파일은 디스크에서 고치지 않는다. 이 프로세스 안에서만 runtime wrap.
래퍼는 읽기만 한다 - RNG 를 소비하지도, 그래프를 바꾸지도, 분기를 바꾸지도 않는다.

호출 구조(코드에서 확인, 추측 아님)
  nmcs_pass(level=2)  - commit 마다 - for a in range(n):
                          nmcs_inner(pfx+[a], cap)      <- 후보 1개의 probe 1회
                            +- for _ in range(NMCS_INNER):
                                 iterate(fpath=p2)      <- probe 안의 inner step
  => 후보 probe 완료 지점 = nmcs_inner 반환 시점.
  => nmcs_pass 의 commit node cur 는 pnodes 로 복원된다.
     nmcs_inner 에 들어온 pfx 는 (commit pfx + [a]) 이므로
     cur == pnodes[len(c_pfx) - 1], child == pnodes[len(c_pfx)].

TRACE_OFF=1 로 돌리면 래퍼를 달지 않는다(동일성 회귀 비교용).
"""
import collections
import hashlib
import io
import json
import os
import sys
import time

os.environ.setdefault("TK_ITERS", "0")
os.environ.setdefault("TK_HAND", "30336082,73642296")
os.environ.setdefault("TK_FROLL", "0")
os.environ.setdefault("TK_PROP", "0")
os.environ.setdefault("TK_OPTSEL", "0")
os.environ.setdefault("TK7_ORDER", "0")
os.environ.setdefault("TK7_NMCS", "2")
os.environ.setdefault("TK7_NMCS_K", "2")
sys.path.insert(0, "/opt/ygo/_diag/turnkill_trial/src")
import tk_r7 as R7                                    # noqa: E402

OFF = os.environ.get("TRACE_OFF", "0") == "1"
SEED = os.environ.get("TK_RSEED", "12345")
CLK = time.perf_counter
_o_inner = R7.nmcs_inner
_o_iter = R7.iterate
_o_roll = R7.rollout

CTX = []                       # nmcs_inner 중첩 컨텍스트
ROWS = []                      # iterate(=inner step) 단위 행
PROBE = []                     # probe(=nmcs_inner) 단위 행
PCNT = collections.Counter()   # (node, action) -> 지금까지의 probe 수
FZ = {}                        # (node, action) -> [value, call]
BEST = {}                      # (node, action) -> best value so far
LIC = {}                       # (node, action) -> last improvement call
ROLLSTEP = [0]
# acts 값/방문은 backup(path,...) 에서만 바뀐다. 그런데 rollout 이 ROLLNODE 일 때
# path 에 간선을 덧붙이므로(tk_r7.py:420) path 는 zip(pnodes,picks) 보다 길다.
# 그래서 backup 을 감싸 실제로 건드린 쌍만 모으고, 다음 iterate 진입 시점에
# 미러에 반영한다. 그러면 nmcs_pass 가 vis0 를 읽는 순간의 값과 정확히 같다.
MVAL = {}                      # (node, action) -> value  (직전 iterate 종료 시점)
MVIS = {}                      # (node, action) -> visits
MNV = {}                       # node -> visits
PEND = []                      # backup 이 건드린 (node, action) / node


def w_rollout(g, info, depth, node, path):
    b0 = R7.STAT["rollout_step"]
    r = _o_roll(g, info, depth, node, path)
    ROLLSTEP[0] = R7.STAT["rollout_step"] - b0
    return r


def _aval(nid, a):
    nd = R7.NODES[nid]
    return nd.acts[a][1] if a in nd.acts else 0


def _avis(nid, a):
    nd = R7.NODES[nid]
    return nd.acts[a][0] if a in nd.acts else 0


_o_backup = R7.backup


def w_backup(path, val):
    _o_backup(path, val)
    for eid in path:
        e = R7.EDGES[eid]
        PEND.append((e.parent, e.action))
    if path:
        PEND.append(R7.EDGES[path[-1]].child)


def _flush():
    for x in PEND:
        if isinstance(x, tuple):
            MVAL[x] = _aval(x[0], x[1])
            MVIS[x] = _avis(x[0], x[1])
            MNV[x[0]] = R7.NODES[x[0]].visits
        else:
            MNV[x] = R7.NODES[x].visits
    del PEND[:]


def w_iterate(fpath=None):
    fp = tuple(fpath) if fpath else ()
    ROLLSTEP[0] = 0
    _flush()                 # 직전 iterate 의 backup 결과를 여기서 반영한다
    c = CTX[-1] if CTX else None
    t0 = CLK()
    r = _o_iter(fpath=fpath) if fpath is not None else _o_iter()
    dt = CLK() - t0
    best, dep, picks, pnodes, natk = r
    if c is None:

        return r
    a = c["a"]
    # 첫 inner step 에서 commit node 를 확정한다
    if c["node"] is None:
        if len(pnodes) < len(c["pfx"]):

            return r                       # 라인이 끊겨 부모를 못 정한다
        nid = pnodes[len(c["pfx"]) - 1]
        c["node"] = nid
        key = (nid, a)
        PCNT[key] += 1
        c["probe_index"] = PCNT[key]
        c["key"] = key
    nid = c["node"]
    # 미러에는 이번 iterate 직전 값이 들어 있다 (아직 갱신 전)
    vb = MVAL.get((nid, a), 0)
    vsb = MVIS.get((nid, a), 0)
    nvb = MNV.get(nid, 0)
    nd = R7.NODES[nid]
    va = _aval(nid, a)
    child = pnodes[len(c["pfx"])] if len(pnodes) > len(c["pfx"]) else None
    ROWS.append({
        "seed": SEED, "commit_id": c["commit"], "global_call_index": c["call"],
        "node_id": nid, "node_depth": nd.depth,
        "action": a, "candidate_index": a,
        "candidate_count": (nd.prompt[1] if nd.prompt else 0),
        "probe_index": c["probe_index"], "inner_index": c["inner"],
        "value_before": vb, "value_after": va, "delta_value": va - vb,
        "iter_value": best,
        "act_visits_before": vsb, "act_visits_after": _avis(nid, a),
        "node_visits_before": nvb, "node_visits_after": nd.visits,
        "forced_steps": len(fp), "rollout_steps": ROLLSTEP[0],
        "child_node": child,
        "child_depth": (R7.NODES[child].depth if child is not None else None),
        "terminal": bool(child is not None and R7.NODES[child].terminal),
        "line_length": len(pnodes) - 1, "attack_count": natk,
        "ms": 1000.0 * dt,
    })
    c["inner"] += 1
    if best > c["best"]:
        c["best"] = best

    return r


def w_inner(pfx, cap):
    c = {"pfx": tuple(pfx), "a": (pfx[-1] if pfx else None), "node": None,
         "key": None, "probe_index": 0, "inner": 0, "best": -1,
         "commit": R7.STAT["nmcs_step"], "call": R7.STAT["nmcs_calls"]}
    CTX.append(c)
    try:
        v, nodes = _o_inner(pfx, cap)
    finally:
        CTX.pop()
    nid = c["node"]
    if nid is None:
        return v, nodes
    key = c["key"]
    a = c["a"]
    nd = R7.NODES[nid]
    val = _aval(nid, a)
    if val > 0 and key not in FZ:
        FZ[key] = [val, c["call"]]
    pb = BEST.get(key, 0)
    improved = val > pb
    if improved:
        BEST[key] = val
        LIC[key] = c["call"]
    # 이 probe 의 첫 inner step 직전 값 = trajectory 의 이전 점
    v_in = None
    for row in reversed(ROWS):
        if row["node_id"] == nid and row["action"] == a \
                and row["probe_index"] == c["probe_index"] \
                and row["inner_index"] == 0:
            v_in = row["value_before"]
            break
    PROBE.append({
        "seed": SEED, "node_id": nid, "action": a, "node_depth": nd.depth,
        "candidate_count": (nd.prompt[1] if nd.prompt else 0),
        "probe_index": c["probe_index"], "commit_id": c["commit"],
        "global_call_index": c["call"], "inner_steps": c["inner"],
        "probe_value": v, "value_before": v_in, "value_after": val,
        "delta_value": (val - v_in) if v_in is not None else None,
        "best_value": BEST.get(key, 0), "improved": improved,
        "act_visits_after": _avis(nid, a), "node_visits_after": nd.visits,
        "first_nonzero": FZ.get(key, [0, -1])[0],
        "first_nonzero_call": FZ.get(key, [0, -1])[1],
        "last_improvement_call": LIC.get(key, -1),
    })
    return v, nodes


def install():
    R7.backup = w_backup
    R7.nmcs_inner = w_inner
    R7.iterate = w_iterate
    R7.rollout = w_rollout


def run(iters):
    t0 = CLK()
    while R7.STAT["engine_session"] < iters:
        b0 = R7.STAT["engine_session"]
        R7.nmcs_pass(R7.NMCS_LEVEL, iters)
        if R7.STAT["engine_session"] == b0:
            R7.iterate()
    return CLK() - t0


def graph_hash():
    nr = [[n.depth, n.visits, n.value, 1 if n.terminal else 0,
           (n.prompt[1] if n.prompt else 0), len(n.incoming), n.raw,
           sorted([a, s[0], s[1], list(s[2])] for a, s in n.acts.items()),
           list(n.untried)]
          for n in R7.NODES]
    er = [[e.parent, e.action, e.child, e.visits, e.value] for e in R7.EDGES]
    blob = json.dumps({"n": nr, "e": er}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def main():
    iters = int(os.environ.get("TR_ITERS", "200"))
    if not OFF:
        install()
    wall = run(iters)
    st = R7.STAT
    out = {
        "seed": SEED, "iters": iters, "off": OFF,
        "graph_hash": graph_hash(),
        "nodes": len(R7.NODES), "edges": len(R7.EDGES),
        "best": max([n.value for n in R7.NODES], default=0),
        "maxdepth": max([n.depth for n in R7.NODES], default=0),
        "maxatk": st.get("maxatk", 0),
        "commits": st.get("nmcs_step", 0), "calls": st.get("nmcs_calls", 0),
        "sessions": st.get("engine_session", 0),
        "rng_roll": list(R7.ROLLRNG.getstate()[1][:6]),
        "rng_exp": list(R7.RNG.getstate()[1][:6]),
        "stat": dict(st), "wall": wall,
        # 기존 slog / stag_*.json 형식을 건드리지 않고 새 키로만 넣는다
        "candidate_trace": ROWS,
        "candidate_probe": PROBE,
    }
    p = os.environ.get("TR_OUT", "")
    if p:
        json.dump(out, io.open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print("seed=%s off=%s  nodes=%d edges=%d best=%d maxdepth=%d maxatk=%d"
          % (SEED, OFF, out["nodes"], out["edges"], out["best"],
             out["maxdepth"], out["maxatk"]))
    print("  commits=%d calls=%d sessions=%d iters=%d forced=%d rollstep=%d"
          % (out["commits"], out["calls"], out["sessions"],
             st.get("iters", 0), st.get("forced_step", 0),
             st.get("rollout_step", 0)))
    print("  hold_new=%d hold_keep=%d tie_switch=%d allzero=%d"
          % (st.get("nmcs_hold_new", 0), st.get("nmcs_hold_keep", 0),
             st.get("nmcs_tie_switch", 0), st.get("nmcs_all_zero_step", 0)))
    print("  integrity: " + " ".join(
        "%s=%d" % (k, st.get(k, 0)) for k in
        ("edge_overwrite", "edge_loss", "edge_rebind", "edge_dup_child",
         "arity_mix", "oor_skip", "send_error", "sig_collide")))
    print("  graph_hash=%s  wall=%.1fs" % (out["graph_hash"], wall))
    print("  rng_roll=%s" % out["rng_roll"])
    print("  rng_exp=%s" % out["rng_exp"])
    print("  trace rows=%d  probe rows=%d" % (len(ROWS), len(PROBE)))


if __name__ == "__main__":
    main()
