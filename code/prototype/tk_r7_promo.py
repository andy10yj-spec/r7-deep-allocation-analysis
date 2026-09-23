# -*- coding: utf-8 -*-
"""[R7] Node / Edge 를 분리한 새 search graph prototype.

기존 tk_r6d.py 의 전제
    children[parent][action] = child   (행동 하나당 자식 하나)
는 이 엔진에서 성립하지 않는다. 전수 census 에서 확인된 사실:
관측 가능한 state 가 완전히 같고 같은 action 을 보내도, 엔진 내부의
continuation 에 따라 서로 다른 후속 상태로 갈라지는 경우가 396건 중
234건(59.1%)이었다.

그래서 이 파일은 그 현상을 '고쳐야 할 오류' 가 아니라 '표현해야 할 사실' 로
다룬다. 같은 (parent, action) 아래에 여러 transition 이 공존할 수 있고,
서로 다른 transition 이 같은 관측 state 로 수렴할 수도 있다.

tk_r6d.py / solver.py 는 읽기만 한다. 수정하지 않는다.
reward / policy / MAST / 카드별 규칙은 쓰지 않는다.
"""
import io
import json
import os
import random
import sys
import time
from collections import Counter

HERE = "/opt/ygo/code_next"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_pol"))
os.chdir(HERE)

HAND = os.environ.get("TK_HAND", "91810826")
LP0 = int(os.environ.get("TK_LP", "40000"))
ITERS = int(os.environ.get("TK_ITERS", "1000"))
MAXDEC = int(os.environ.get("TK_MAXDEC", "140"))
RSEED = int(os.environ.get("TK_RSEED", "12345"))
TAG = os.environ.get("TK_TAG", "R7")
OUT = os.environ.get("TK_OUT", "")
# selection: 짝수 번째 방문은 '가장 덜 가본 행동' 을 고른다.
# 값이 낮다는 이유만으로 어떤 가지가 영구히 봉쇄되지 않게 하는 유일한 장치다.
EXPLORE_EVERY = int(os.environ.get("TK7_EXPLORE", "2"))
ROLLNODE = os.environ.get("TK7_ROLLNODE", "1") == "1"
# [1단계 이식] candidate ordering 만. 확장 순서만 정하고 그 외엔 관여하지 않는다.
ORDER_ON = os.environ.get("TK7_ORDER", "0") == "1"
# [2단계 이식] NMCS. 예산을 한 continuation 에 몰아주는 장치.
NMCS_LEVEL = int(os.environ.get("TK7_NMCS", "0"))   # 0=끔, 1, 2
NMCS_INNER = int(os.environ.get("TK7_NMCS_INNER", "6"))
# [tie-fix] value 동점일 때 방문이 적은 쪽을 고른다. 0 이면 예전 동작.
NMCS_TIE = os.environ.get("TK7_NMCS_TIE", "1") == "1"
# [persistence] 동점 후보 하나를 연속 몇 번(NMCS step)까지 유지할 것인가.
#   k=1  : 매번 다시 고른다 = 현재 least-visited 와 동일
#   k=큰값: 한 번 고르면 계속 유지 = 기존 first-action 쪽
NMCS_K = int(os.environ.get("TK7_NMCS_K", "1"))
TIEHOLD = {}        # node -> [잡고 있는 action, 남은 횟수]
# [계측 전용] (node, action) -> stagnation 기록. 탐색에 영향을 주지 않는다.
STAG_ON = os.environ.get("TK7_STAG", "0") == "1"
SLOG = {}
AUDIT_ON = os.environ.get("TK_AUDIT", "1") == "1"

for k, v in (("DECK_YDK", "/opt/ygo/_diag/turnkill_trial/src/Tenpai.ydk"),
             ("OPP_YDK", "/opt/ygo/decks_meta/Sky_Striker.ydk"),
             ("HAND", HAND), ("OPP_HAND", "RANDOM:0"), ("OPP_SEED", "7"),
             ("OPP_POLICY", "pass"), ("REPLAY_SETUP", "0"),
             ("BATTLE", "1"), ("PLAY_SETS", "1"),
             ("POLICY_BATTLE", "1"), ("GAME_MAXTURNS", "2"),
             ("SELMAX_OPTS", "1")):
    os.environ.setdefault(k, v)

import solver as S                      # noqa: E402
from _pol import rawstate as R          # noqa: E402
from ocg import OCG_QueryInfo as QI     # noqa: E402

_FLAGS = S.DUEL_FLAGS | 0x2
_mk = S.make_duel
S.make_duel = (lambda draw=0, flags=None, lp=8000, per_turn=1,
               seed=(1, 2, 3, 4):
               _mk(draw=draw, flags=_FLAGS, lp=LP0, per_turn=0, seed=seed))

# ---- policy (읽기 전용). ordering 에만 쓴다. ----
POL = None
try:
    import polfeat
    import dump_boards as db
    import build_label_queue as blq
    from _pol import evalfn
    from _pol.duelgame import board_context
    _f, _p, _vw = evalfn.load("script")
    polfeat.set_leaf_eval(_f, _p)

    def _lw(pth, d=None):
        try:
            x = json.load(io.open(pth, encoding="utf-8"))
            return dict(x.get("weights", x))
        except Exception:
            return dict(d or {})
    POL = (_lw("/opt/ygo/code_next/_pol/policy_base2.json"),
           _lw("/opt/ygo/_diag/turnkill_trial/src/scriptheur.json", _vw))
except Exception as _e:
    print("policy 로드 실패, 원래 순서로 진행: %r" % _e)

SELS = []           # 이번 iteration 에서 고른 라벨 (policy feature 용)
FIRSTEXP = {}       # depth -> Counter(첫 확장 action)

RNG = random.Random(RSEED)                # 확장 순서
ROLLRNG = random.Random(RSEED ^ 0x5EED)   # rollout 전용 (스트림 분리)
STAT = Counter()
# [P1] rollout 이 실제로 고른 action 을 그 노드의 untried 에서도 뺀다.
# forced 경로(iterate)와 같은 bookkeeping. 노드/간선/backup 은 건드리지 않는다.
PROMO = os.environ.get("TK7_PROMO", "0") == "1"
PROMO_D = Counter()     # 제거가 일어난 노드의 depth
PROMO_PER = Counter()   # rollout 1회당 제거 수


def close(g):
    try:
        g.close()
    except Exception:
        pass


# ============================ OBSERVATION ============================
# 공개 정보만 쓴다. 상대 뒷면 카드의 코드/상태는 읽지 않는다(기존 masking 유지).
# history / PENDTAG / 누적 카운터는 절대 넣지 않는다. 그건 state 가 아니다.
def snap():
    d = S.LIVE[0]
    if not d:
        return None
    try:
        f = R.query_field(S.core, d)
    except Exception:
        return None
    b, atks, wit = [], [], []
    for con in (0, 1):
        for loc in (S.LOC_MZONE, S.LOC_SZONE):
            try:
                cs = list(R.scan_location(S.core, QI, d, con, loc))
            except Exception:
                cs = []
            for c in cs:
                if not c:
                    continue
                pos = c.get("position", 0)
                fu = bool(pos & R.POS_FACEUP)
                hid = (con == 1 and not fu)     # 상대 뒷면 = 비공개
                b.append((con, loc, c.get("seq"), pos,
                          (c.get("code") or 0) if fu else 0,
                          0 if hid else (c.get("status") or 0)))
                if fu and loc == S.LOC_MZONE:
                    atks.append((con, c.get("seq"), c.get("atk"),
                                 c.get("level"), c.get("type")))
                    wit.append((con, c.get("seq"), c.get("atk"),
                                c.get("level"), c.get("type")))

    def loc(l):
        try:
            return tuple(sorted(S.query_loc_con(d, l, 0)))
        except Exception:
            return ()

    def rawloc(l):
        try:
            return tuple(S.query_loc_con(d, l, 0))
        except Exception:
            return ()

    olp = f["players"][1]["lp"]
    if olp >= (1 << 31):
        olp -= (1 << 32)
    cx = S.CTX[0] or {}
    dk = rawloc(S.LOC_DECK)
    return {"olp": olp, "mlp": f["players"][0]["lp"],
            "board": tuple(sorted(b)), "atk": tuple(sorted(atks)),
            "hand": loc(S.LOC_HAND), "gy": loc(S.LOC_GRAVE),
            "rm": loc(S.LOC_REMOVED), "dk": dk, "ex": loc(S.LOC_EXTRA),
            "turn": cx.get("turn"), "tp": cx.get("turn_player"),
            "ph": cx.get("phase"),
            "wit": ((tuple(sorted(wit)), cx.get("phase"), len(dk))
                    if AUDIT_ON else None)}


def observe(info):
    """(관측 signature, 프롬프트, 후보라벨, damage, witness) 를 만든다.

    signature 는 '지금 보이는 것' 만 담는다. 어떻게 여기 왔는지는 담지 않는다.
    같은 signature 가 서로 다른 continuation 을 가질 수 있다는 것이
    이 프로토타입의 출발점이므로, 그걸 history 로 덮어 감추지 않는다.
    """
    st = snap()
    if st is None:
        # 듀얼이 끝나 엔진 상태를 더 읽을 수 없다. 그래도 종단 노드는 만든다.
        # 만들지 않으면 그 간선이 영원히 visits=0 이 되어 selection 이
        # 그 행동에 고착된다(초기 구현에서 실제로 발생).
        cx = S.CTX[0] or {}
        return ("__END__", cx.get("winner")), None, (), None, None
    live = (info is not None and isinstance(info[1], int) and info[1] > 0)
    if live:
        mt = (info[6], info[1])
        lab = tuple(str(x) for x in (info[7] or [])[:info[1]])
    else:
        mt, lab = None, ()
    cx = S.CTX[0] or {}
    sig = (st["olp"], st["mlp"], st["board"], st["atk"], st["hand"],
           st["gy"], st["rm"], st["dk"], st["ex"], st["turn"], st["tp"],
           st["ph"], mt, lab,
           tuple(c.get("code") for c in (cx.get("chain") or [])),
           cx.get("chain_req") if mt and mt[0] == 16 else None,
           cx.get("attacker"))
    return sig, mt, lab, (LP0 - st["olp"]), st["wit"]


def update_best(best, obs, dmg):
    """damage 갱신. 듀얼이 끝난 경우 상대 LP 는 0 이므로 최대 damage 다."""
    if dmg is not None:
        return dmg if dmg > best else best
    if obs and obs[0] == "__END__" and obs[1] == 0:
        return LP0
    return best


def order_cands(info, cands):
    """policy_base2 점수 내림차순 인덱스. 실패하면 원래 순서.

    후보를 제거하거나 합치지 않는다. 순서만 바꾼다.
    동점은 파이썬 정렬이 안정적이라 원래 인덱스 순서가 유지된다(결정적).
    """
    n = len(cands)
    base = list(range(n))
    if not ORDER_ON or POL is None or n < 2:
        return base
    try:
        rec = db.zones(info[3], set())
        rec["used"] = sorted(blq.derive_used(tuple(SELS)))
        board_context(S, rec)
        sc = list(polfeat.score_options(POL[0], POL[1], rec, cands, play=True))
        out = sorted(base, key=lambda i: -sc[i])
    except Exception:
        STAT["policy_fail"] += 1
        return base
    if out != base:
        STAT["policy_reorder"] += 1
    else:
        STAT["policy_same"] += 1
    return out


# ============================ GRAPH ============================
# Node = 관측 state.            Edge = 하나의 transition.
# 같은 (parent, action) 아래 여러 Edge 가 공존할 수 있다  <- 핵심
# 서로 다른 Edge 가 같은 Node 로 들어올 수 있다            <- 핵심
class Node(object):
    __slots__ = ("id", "obs", "prompt", "labels", "terminal", "visits",
                 "value", "depth", "untried", "acts", "incoming", "raw")

    def __init__(self, nid, obs, prompt, labels, depth, order=None, raw=0):
        self.id = nid
        self.obs = obs
        self.prompt = prompt
        self.labels = labels
        self.terminal = prompt is None
        self.visits = 0
        self.value = 0
        self.depth = depth
        self.raw = raw              # 생성 시점의 실제 누적 damage (backup 아님)
        n = prompt[1] if prompt else 0
        if order is not None and len(order) == n:
            # pop() 은 뒤에서 꺼내므로 뒤집어 둔다 -> 점수 높은 것부터 확장
            self.untried = list(reversed(order))
        else:
            self.untried = list(range(n))
            RNG.shuffle(self.untried)
        # action -> [visits, value, [edge_id, ...]]
        self.acts = dict((a, [0, 0, []]) for a in range(n))
        self.incoming = []          # edge_id 목록 (여러 부모 허용)


class Edge(object):
    __slots__ = ("id", "parent", "action", "child", "ctx", "visits", "value")

    def __init__(self, eid, parent, action, child, ctx):
        self.id = eid
        self.parent = parent
        self.action = action
        self.child = child
        self.ctx = ctx          # transition context (식별자가 아니라 기록)
        self.visits = 0
        self.value = 0


NODES = []
EDGES = []
NIDX = {}               # obs signature -> node id
EIDX = {}               # (parent, action, child) -> edge id
AUDIT = {}
COLLIDE = []


def get_node(obs, prompt, labels, depth, wit, info=None, raw=0):
    i = NIDX.get(obs)
    if i is None:
        # ORDER_ON 이 꺼져 있으면 order=None 을 넘겨 기존 RNG.shuffle 경로를
        # 그대로 타게 한다. 그래야 baseline 이 이전 측정과 같은 탐색이 된다.
        order = None
        if ORDER_ON and prompt is not None:
            order = order_cands(info, list(labels))
        i = len(NODES)
        NODES.append(Node(i, obs, prompt, labels, depth, order, raw))
        if prompt is not None:
            # 실제로 가장 먼저 확장될 action (pop 은 뒤에서 꺼낸다)
            FIRSTEXP.setdefault(depth, Counter())[NODES[i].untried[-1]] += 1
        NIDX[obs] = i
        STAT["node_new"] += 1
        if AUDIT_ON and wit is not None:
            AUDIT[obs] = wit
    else:
        STAT["node_reuse"] += 1
        nd = NODES[i]
        if depth < nd.depth:
            nd.depth = depth
        # 무결성: 같은 signature 인데 arity 가 다르면 즉시 드러난다
        if nd.prompt != prompt:
            STAT["arity_mix"] += 1
        if AUDIT_ON and wit is not None:
            old = AUDIT.get(obs)
            if old is None:
                AUDIT[obs] = wit
            elif old != wit:
                STAT["sig_collide"] += 1
                if len(COLLIDE) < 5:
                    COLLIDE.append((old, wit))
    return i


def get_edge(parent, action, child, ctx):
    """(parent, action) 이 이미 있어도 child 가 다르면 새 edge 를 만든다."""
    k = (parent, action, child)
    e = EIDX.get(k)
    if e is None:
        before = list(NODES[parent].acts[action][2])
        e = len(EDGES)
        EDGES.append(Edge(e, parent, action, child, ctx))
        EIDX[k] = e
        NODES[parent].acts[action][2].append(e)
        NODES[child].incoming.append(e)
        STAT["edge_new"] += 1
        # [무결성] 새 transition 이 들어와도 기존 edge 는 하나도 사라지면 안 된다.
        after = NODES[parent].acts[action][2]
        if before != after[:len(before)] or len(after) != len(before) + 1:
            STAT["edge_overwrite"] += 1
        if len(after) >= 2:
            STAT["multi_transition_created"] += 1
    else:
        if EDGES[e].parent != parent or EDGES[e].action != action                 or EDGES[e].child != child:
            STAT["edge_rebind"] += 1
        STAT["edge_reuse"] += 1
    return e


def trans_ctx(info):
    """transition 을 설명하는 관측 맥락. 식별자로 쓰지 않는다."""
    cx = S.CTX[0] or {}
    return (cx.get("phase"), cx.get("turn"),
            None if info is None or not isinstance(info[1], int)
            else info[1])


# ============================ SEARCH ============================
def advance(g, info):
    """후보가 1개뿐인 지점은 결정이 아니므로 그냥 통과시킨다."""
    while info is not None and isinstance(info[1], int) and info[1] == 1:
        try:
            info = g.send(0)
        except StopIteration:
            return None
        except Exception:
            return None
    return info


def pick_action(nd):
    """가장 단순한 graph selection.

    1. 아직 시도하지 않은 행동이 있으면 그것 (expansion)
    2. 없으면 - 짝수 번째 방문은 가장 덜 가본 행동, 그 외는 값이 가장 큰 행동
    damage weighting / policy / MAST / 카드별 보너스는 쓰지 않는다.
    """
    if nd.untried:
        return nd.untried.pop(), True
    if not nd.acts:
        return None, False
    if EXPLORE_EVERY > 0 and (nd.visits % EXPLORE_EVERY) == 0:
        a = min(nd.acts, key=lambda x: (nd.acts[x][0], x))
        STAT["sel_explore"] += 1
    else:
        a = max(nd.acts, key=lambda x: (nd.acts[x][1], -nd.acts[x][0], -x))
        STAT["sel_exploit"] += 1
    return a, False


def rollout(g, info, depth, node, path):
    """균등 무작위 playout. 정책도 가중치도 쓰지 않는다."""
    best = 0
    natk = 0
    _np = 0
    while depth < MAXDEC:
        if info is None or not isinstance(info[1], int) or info[1] <= 0:
            break
        if (S.CTX[0] or {}).get("winner") is not None:
            break
        n = info[1]
        a = ROLLRNG.randrange(n)
        if node is not None and a < len(NODES[node].labels):
            _l = NODES[node].labels[a]
            SELS.append(_l)
            if _l.startswith("공격"):
                natk += 1
        try:
            info = g.send(a)
        except StopIteration:
            info = None
        except Exception:
            break
        info = advance(g, info)
        depth += 1
        STAT["rollout_step"] += 1
        obs, prompt, lab, dmg, wit = observe(info)
        best = update_best(best, obs, dmg)
        if ROLLNODE and node is not None:
            c = get_node(obs, prompt, lab, depth, wit, info, dmg or 0)
            e = get_edge(node, a, c, trans_ctx(info))
            if PROMO and a in NODES[node].untried:
                NODES[node].untried.remove(a)
                STAT["promo"] += 1
                PROMO_D[NODES[node].depth] += 1
                _np += 1
            path.append(e)
            node = c
            if NODES[c].terminal:
                break
        elif prompt is None:
            break
    if PROMO:
        PROMO_PER[_np] += 1
    return best, depth, natk


def backup(path, val):
    """max-backup. edge 와 node 를 각각 독립적으로 올린다."""
    for eid in path:
        e = EDGES[eid]
        e.visits += 1
        if val > e.value:
            e.value = val
        nd = NODES[e.parent]
        st = nd.acts.get(e.action)
        if st is not None:
            st[0] += 1
            if val > st[1]:
                st[1] = val
        nd.visits += 1
        if val > nd.value:
            nd.value = val
    if path:
        c = NODES[EDGES[path[-1]].child]
        c.visits += 1
        if val > c.value:
            c.value = val


def iterate(fpath=None):
    """selection -> expansion -> rollout -> max-backup.

    fpath 가 주어지면 그 길이만큼 행동을 강제한 뒤 평소대로 이어간다.
    엔진에 snapshot API 가 없어서 중간 상태에서 다시 시작하려면 root 부터
    재생하는 수밖에 없다. 강제 구간에서도 노드/간선 생성과 backup 은
    평소와 똑같은 경로를 탄다(구조를 우회하지 않는다).
    """
    _fp = tuple(fpath) if fpath else ()
    STAT["engine_session"] += 1
    g = S.duel_session(opp_turn=True, want_mt=True, opp_play=True, obs=None,
                       max_turns=2)
    try:
        info = next(g)
    except Exception:
        return 0, 0, [], [], 0
    info = advance(g, info)
    obs, prompt, lab, dmg, wit = observe(info)
    if obs is None or dmg is None:
        close(g)
        return 0, 0, [], [], 0
    del SELS[:]
    cur = get_node(obs, prompt, lab, 0, wit, info, dmg)
    path = []
    picks = []
    pnodes = [cur]
    natk = 0
    depth = 0
    best = dmg
    expanded = False
    while depth < MAXDEC:
        nd = NODES[cur]
        if nd.terminal or (S.CTX[0] or {}).get("winner") is not None:
            break
        if depth < len(_fp):
            a = _fp[depth]
            if a >= (nd.prompt[1] if nd.prompt else 0):
                break
            if a in nd.untried:
                nd.untried.remove(a)      # 강제로 써도 미시도 목록에서 뺀다
            is_new = False
            STAT["forced_step"] += 1
        else:
            a, is_new = pick_action(nd)
        if a is None:
            break
        if a >= (nd.prompt[1] if nd.prompt else 0):
            STAT["oor_skip"] += 1
            break
        picks.append(a)
        _lb = nd.labels[a] if a < len(nd.labels) else "?"
        SELS.append(_lb)
        if _lb.startswith("공격"):
            natk += 1
        try:
            info = g.send(a)
        except StopIteration:
            info = None
        except Exception:
            STAT["send_error"] += 1
            break
        info = advance(g, info)
        depth += 1
        obs, prompt, lab, dmg, wit = observe(info)
        best = update_best(best, obs, dmg)
        child = get_node(obs, prompt, lab, depth, wit, info, dmg or 0)
        e = get_edge(cur, a, child, trans_ctx(info))
        path.append(e)
        cur = child
        pnodes.append(cur)
        if is_new:
            expanded = True
            break
    if (expanded or depth >= len(_fp)) and not NODES[cur].terminal:
        rb, depth, ra = rollout(g, info, depth, cur if ROLLNODE else None,
                                path)
        if rb > best:
            best = rb
        natk += ra
    close(g)
    backup(path, best)
    STAT["iters"] += 1
    if natk > STAT["maxatk"]:
        STAT["maxatk"] = natk
    return best, depth, picks, pnodes, natk


def nmcs_pass(level, cap):
    """NMCS. 한 라인을 따라가며 각 후보의 continuation 을 실제로 재생해 보고
    가장 좋은 것을 확정한 뒤 한 칸 전진한다. 그 다음 결정점에서 다시 반복한다.

    카드 이름도 알려진 combo 도 쓰지 않는다. 엔진이 주는 후보만 쓴다.
    결과는 전부 기존 get_node / get_edge / backup 경로로 기록되므로
    다른 transition 을 지우거나 덮어쓰지 않는다.
    """
    pfx = []
    cur = 0 if NODES else None
    STAT["nmcs_calls"] += 1
    while cur is not None and STAT["engine_session"] < cap \
            and len(pfx) < MAXDEC:
        nd = NODES[cur]
        if nd.terminal or not nd.acts:
            break
        n = nd.prompt[1] if nd.prompt else 0
        res = []
        for a in range(n):
            if STAT["engine_session"] >= cap:
                break
            # 탐색 전 방문 수를 쓴다. 탐색하면 그 action 의 방문이 늘어나서
            # 나중에 읽으면 순서에 따라 불공평해진다.
            vis0 = nd.acts[a][0] if a in nd.acts else 0
            if level >= 2:
                v, nodes = nmcs_inner(pfx + [a], cap)
            else:
                v, _d, _p, nodes, _k = iterate(fpath=pfx + [a])
                STAT["nmcs_inner"] += 1
            res.append((a, v, vis0, nodes))
        if not res:
            break
        if STAG_ON:
            _call = STAT["nmcs_calls"]
            for _a, _v, _vis0, _nodes in res:
                _k2 = (cur, _a)
                _r = SLOG.get(_k2)
                if _r is None:
                    # [first_seen, first_nz_call, first_nz_value, best_value,
                    #  best_call, last_improve_call, vis_at_last_improve,
                    #  last_seen_call, probe_cnt]
                    _r = [_call, -1, 0, 0, -1, -1, 0, _call, 0]
                    SLOG[_k2] = _r
                _r[7] = _call
                _r[8] += 1
                _cv = nd.acts[_a][1] if _a in nd.acts else 0
                if _cv > 0 and _r[1] < 0:
                    _r[1] = _call
                    _r[2] = _cv
                if _cv > _r[3]:
                    _r[3] = _cv
                    _r[4] = _call
                    _r[5] = _call
                    _r[6] = nd.acts[_a][0] if _a in nd.acts else 0
        bv = max(z[1] for z in res)
        grp = [z for z in res if z[1] == bv]        # 최고 value 동점군
        if len(grp) >= 2:
            STAT["nmcs_tie_switch"] += len(grp) - 1
        if bv == 0:
            STAT["nmcs_all_zero_step"] += 1
        if not NMCS_TIE:
            pick = min(grp, key=lambda z: z[0])     # 기존 first-action
        elif len(grp) == 1:
            pick = grp[0]
            TIEHOLD.pop(cur, None)                  # 동점이 아니면 hold 해제
        else:
            gset = dict((z[0], z) for z in grp)
            h = TIEHOLD.get(cur)
            # value 가 더 높은 후보가 나타나면 hold 는 즉시 무효다.
            # (h[0] 가 최고 value 동점군에 없으면 그 상황이다)
            if h is not None and h[0] in gset and h[1] > 0:
                pick = gset[h[0]]
                h[1] -= 1
                STAT["nmcs_hold_keep"] += 1
            else:
                pick = min(grp, key=lambda z: (z[2], z[0]))
                TIEHOLD[cur] = [pick[0], NMCS_K - 1]
                STAT["nmcs_hold_new"] += 1
        ba, bn = pick[0], pick[3]
        pfx.append(ba)
        STAT["nmcs_step"] += 1
        if bn is not None and len(bn) > len(pfx):
            cur = bn[len(pfx)]
        else:
            break


def nmcs_inner(pfx, cap):
    """level 2 용. pfx 에서 시작해 짧은 level-1 탐색을 돌린다."""
    best, bnodes = -1, None
    p2 = list(pfx)
    for _ in range(NMCS_INNER):
        if STAT["engine_session"] >= cap:
            break
        v, _d, _p, nodes, _k = iterate(fpath=p2)
        STAT["nmcs_inner"] += 1
        if v > best:
            best, bnodes = v, nodes
        if len(nodes) <= len(p2):
            break
        nxt = NODES[nodes[len(p2)]]
        if nxt.terminal or not nxt.acts:
            break
        if NMCS_TIE:
            # value 내림차순 -> 방문 오름차순 -> index 오름차순
            _na = min(nxt.acts,
                      key=lambda x: (-nxt.acts[x][1], nxt.acts[x][0], x))
        else:
            _na = max(nxt.acts, key=lambda x: (nxt.acts[x][1], -x))
        p2 = p2 + [_na]
    return best, bnodes


# ============================ RUN ============================
def main():
    t0 = time.time()
    best = 0
    bestpicks = []
    bestdepth = 0
    if NMCS_LEVEL > 0:
        # 총 engine session 예산을 baseline 과 같게 맞춘다.
        while STAT["engine_session"] < ITERS:
            b0 = STAT["engine_session"]
            nmcs_pass(NMCS_LEVEL, ITERS)
            if STAT["engine_session"] == b0:
                v, d, p, _n, _k = iterate()
                if v > best:
                    best, bestpicks, bestdepth = v, list(p), d
        for _n in NODES:
            if _n.value > best:
                best = _n.value
    else:
        for _ in range(ITERS):
            v, d, p, _n2, _k = iterate()
            if v > best:
                best = v
                bestpicks = list(p)
                bestdepth = d
    el = time.time() - t0

    # [무결성 전수검사] edge 하나라도 부모/자식 목록에서 빠지면 edge_loss.
    for e in EDGES:
        if e.id != EDGES[e.id].id:
            STAT["edge_loss"] += 1
            continue
        if e.id not in NODES[e.parent].acts.get(e.action, [0, 0, []])[2]:
            STAT["edge_loss"] += 1
        if e.id not in NODES[e.child].incoming:
            STAT["edge_loss"] += 1
        if EIDX.get((e.parent, e.action, e.child)) != e.id:
            STAT["edge_loss"] += 1
    for nd in NODES:
        for a, st in nd.acts.items():
            if len(st[2]) != len(set(EDGES[x].child for x in st[2])):
                STAT["edge_dup_child"] += 1

    multi = [(n.id, a) for n in NODES for a, st in n.acts.items()
             if len(st[2]) >= 2]
    multi_par = set(x[0] for x in multi)
    shared = [n for n in NODES
              if len(set(EDGES[e].parent for e in n.incoming)) >= 2]
    maxdepth = max((n.depth for n in NODES), default=0)
    rt = NODES[0] if NODES else None
    root = []
    if rt:
        for a in sorted(rt.acts):
            root.append([a, rt.acts[a][0], rt.acts[a][1]])

    print("=" * 78)
    print("[TK_R7] tag=%s iters=%d seed=%d" % (TAG, ITERS, RSEED))
    print("=" * 78)
    print("  노드                 : %d" % len(NODES))
    print("  고유 관측 state      : %d" % len(NIDX))
    print("  간선(transition)     : %d" % len(EDGES))
    print("  고유 간선            : %d" % len(EIDX))
    print("  multi-transition     : %d 개 (부모,action) 쌍, 부모 %d개"
          % (len(multi), len(multi_par)))
    print("  여러 부모로 수렴 노드 : %d" % len(shared))
    print("  최대 depth           : %d" % maxdepth)
    print("  best damage          : %d  (depth %d)" % (best, bestdepth))
    print("  소요                 : %.0fs (%.0f ms/iter)"
          % (el, 1000.0 * el / max(ITERS, 1)))
    print()
    print("  NMCS                  : level=%d calls=%d step=%d inner=%d"
          % (NMCS_LEVEL, STAT["nmcs_calls"], STAT["nmcs_step"],
             STAT["nmcs_inner"]))
    print("  NMCS tie              : %s  k=%d  (동점군 %d / 전부0 스텝 %d)"
          % ("least-visited" if NMCS_TIE else "first-wins(기존)", NMCS_K,
             STAT["nmcs_tie_switch"], STAT["nmcs_all_zero_step"]))
    print("  NMCS persistence      : 유지 %d / 새로선택 %d"
          % (STAT["nmcs_hold_keep"], STAT["nmcs_hold_new"]))
    if STAG_ON:
        print("  stagnation 계측        : (node,action) %d건 기록" % len(SLOG))
    print("  engine sessions       : %d   (forced step %d)"
          % (STAT["engine_session"], STAT["forced_step"]))
    print("  max attack count      : %d" % STAT["maxatk"])
    print("  ordering              : %s  (reorder %d / same %d / fail %d)"
          % ("ON" if ORDER_ON else "OFF", STAT["policy_reorder"],
             STAT["policy_same"], STAT["policy_fail"]))
    print()
    print("  [INTEGRITY]")
    for k in ("arity_mix", "oor_skip", "sig_collide", "send_error",
              "edge_overwrite", "edge_rebind", "edge_loss", "edge_dup_child"):
        print("    %-18s %d" % (k, STAT[k]))
    print("    %-18s %s" % ("signone_reuse", "N/A (private 노드 구조 없음)"))
    print("    %-18s %s" % ("stale", "N/A (edge 덮어쓰기 없음)"))
    print()
    print("  [ROOT]")
    for a, v, val in root:
        print("    a%-3d visits=%-8d value=%d" % (a, v, val))
    print()
    print("  [multi-transition 표본]")
    for nid, a in multi[:6]:
        es = NODES[nid].acts[a][2]
        print("    node %-7d action %-3d -> 자식 %s  (프롬프트 %s)"
              % (nid, a, [EDGES[e].child for e in es], NODES[nid].prompt))
        for e in es:
            E = EDGES[e]
            print("        edge %-7d child %-7d ctx=%s visits=%d value=%d"
                  % (E.id, E.child, E.ctx, E.visits, E.value))

    if OUT:
        json.dump({"tag": TAG, "iters": ITERS, "seed": RSEED,
                   "nodes": len(NODES), "unique_obs": len(NIDX),
                   "edges": len(EDGES), "unique_edges": len(EIDX),
                   "multi_transition": len(multi),
                   "multi_parents": len(multi_par),
                   "shared_nodes": len(shared),
                   "maxdepth": maxdepth, "best": best,
                   "best_depth": bestdepth, "best_picks": bestpicks,
                   "root": root, "sec": el,
                   "stat": dict(STAT),
                   "maxatk": STAT["maxatk"],
                   "total_calls": STAT["nmcs_calls"],
                   "slog": ([[n_, a_, r_[0], r_[1], r_[2], r_[3], r_[4],
                              r_[5], r_[6], r_[7], r_[8],
                              NODES[n_].acts[a_][0] if a_ in NODES[n_].acts
                              else 0,
                              NODES[n_].acts[a_][1] if a_ in NODES[n_].acts
                              else 0]
                             for (n_, a_), r_ in SLOG.items()]
                            if STAG_ON else []),
                   "act_rows": [[n.id, a, st[0], st[1]] for n in NODES
                                for a, st in n.acts.items()],
                   "firstexp": dict((str(d), dict(c))
                                    for d, c in FIRSTEXP.items()),
                   "groups": [{"parent": nid, "action": a,
                               "mt": (NODES[nid].prompt or (-1, 0))[0],
                               "ncand": (NODES[nid].prompt or (-1, 0))[1],
                               "pdepth": NODES[nid].depth,
                               "pvisits": NODES[nid].visits,
                               "act_visits": NODES[nid].acts[a][0],
                               "act_value": NODES[nid].acts[a][1],
                               "edges": [[e, EDGES[e].child, EDGES[e].visits,
                                          EDGES[e].value,
                                          NODES[EDGES[e].child].visits,
                                          NODES[EDGES[e].child].value,
                                          NODES[EDGES[e].child].depth,
                                          1 if NODES[EDGES[e].child].terminal
                                          else 0,
                                          NODES[EDGES[e].child].raw]
                                         for e in NODES[nid].acts[a][2]]}
                              for nid, a in multi],
                   "node_rows": [[n.depth, n.visits, n.value,
                                  1 if n.terminal else 0,
                                  n.prompt[1] if n.prompt else 0,
                                  len(n.incoming),
                                  n.prompt[0] if n.prompt else -1,
                                  n.raw]
                                 for n in NODES],
                   "edge_rows": [[e.parent, e.action, e.child, e.visits,
                                  e.value] for e in EDGES]},
                  io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        print("\n  저장: %s" % OUT)


if __name__ == "__main__":
    main()
