"""ARC-1 inference.

    from arc1 import ARC1Predictor
    model = ARC1Predictor("realslapout/ARC-1")          # Hugging Face repo id or a local checkpoint folder
    model.predict(state, questions)

`predict` / `predict_batch` take the typed-decision wire format (choice / score / noul questions) and return
calibrated probabilities; `logits` is the lower-level interface used by the evaluation harness.
"""
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from arc1.models.branched import BranchedBuilder, assemble_tree, collate_branched, collate_tree
from arc1.models.checkpoint import load_checkpoint
from arc1.models.formats import QTYPES, render_options, serialize_state


def _bucket(qtype: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11-32" if k <= 32 else "33+"
    return "%s:%s" % (qtype, size)


def option_keys(q: Dict) -> List[str]:
    """Answer keys in option order: choice labels, score level indices, or ['false', 'true']."""
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    crit = q["criteria"]
    return [str(k) for k in (crit if isinstance(crit, list) else crit.keys())]


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def instructions_text(q: Dict) -> str:
    ins = q["instructions"]
    return ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)


def serve_flags() -> Dict:
    """Serving switches from the environment (all off by default):
    ARC1_SERVE_GRAPH=1  CUDA graphs for small batches (lowest single-request latency on a GPU)
    ARC1_SERVE_HALF=1   keep the weights in bf16 instead of casting under autocast on every call
    ARC1_SERVE_EXIT=L:C answer at layer L when the top option's probability is >= C (experimental)
    ARC1_SERVE_MAXLEN=N serving window in tokens (default: the checkpoint's max_len)"""
    out = {}
    for k, env in (("half", "ARC1_SERVE_HALF"), ("graph", "ARC1_SERVE_GRAPH")):
        out[k] = os.environ.get(env) == "1"
    v = os.environ.get("ARC1_SERVE_EXIT")
    ex = v.split(":") if v and v != "0" else None
    out["exit"] = (int(ex[0]), float(ex[1])) if ex else None
    v = os.environ.get("ARC1_SERVE_MAXLEN")
    out["maxlen"] = int(v) if v else 0
    return out


def resolve(spec: str) -> str:
    """A local checkpoint folder as is; anything else is treated as a Hugging Face repo id and downloaded once."""
    if os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download
    return snapshot_download(spec)


class ARC1Predictor:
    """ARC-1 decision model.

    spec         Hugging Face repo id (e.g. "realslapout/ARC-1") or a local checkpoint folder
    device       "cuda" / "cpu" (default: cuda when available)
    cuda_graphs  capture CUDA graphs for small batches: lowest latency for single requests on a GPU
                 (the first call of each new input shape is slower while it is captured)
    graph_cache  how many padded input shapes keep a captured graph (longer inputs beyond it run eagerly)
    """

    _GKEYS = ("input_ids", "position_ids", "branch", "score_pos", "score_mask", "qtype")

    def __init__(self, spec: str, device: Optional[str] = None, max_len: Optional[int] = None, dtype: str = "bf16",
                 max_tokens_per_batch: int = 32768, max_rows: int = 64, optfit: Optional[int] = None,
                 cuda_graphs: Optional[bool] = None, graph_cache: int = 64):
        t = time.time()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device).startswith("cpu") and dtype == "bf16" and not os.environ.get("ARC1_CPU_BF16"):
            dtype = "fp32"            # bf16 autocast on CPU is slow on most machines
        path = resolve(spec)
        self.model, tok, cfg = load_checkpoint(path, device)
        self.name = cfg.get("name") or path.replace("\\", "/").rstrip("/").split("/")[-1]
        self.load_s = time.time() - t
        self.tok, self.cfg, self.device = tok, cfg, torch.device(device)
        flags = serve_flags()
        if cuda_graphs is not None:
            flags["graph"] = cuda_graphs
        ml = max_len or max(cfg.get("max_len", 1024), flags["maxlen"])
        self.builder = BranchedBuilder(tok, max_len=ml, opt_cap=cfg.get("opt_cap", 48), min_state=cfg.get("min_state", 160))
        optfit = optfit or cfg.get("optfit") or None
        self.max_tokens_per_batch, self.max_rows = max_tokens_per_batch, max_rows
        if optfit:
            self.builder.dyn_cap = int(optfit)
            self.max_tokens_per_batch = max(max_tokens_per_batch, int(optfit))
        self.optfit = optfit
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        on_gpu = str(device).startswith("cuda")
        if flags["half"] and self.dtype != torch.float32 and on_gpu:
            self.model.to(self.dtype)
        # CUDA graphs: each padded batch shape is captured once and replayed, which removes the per-kernel launch
        # overhead of a batch-1 forward
        self._graphs = {} if (flags["graph"] and on_gpu) else None
        self.graph_cache = graph_cache
        self.exit = flags["exit"]
        self.exit_stats = [0, 0]           # rows answered at the exit layer, rows scored
        if self.exit:
            self.name += "+exit%d@%.2f" % self.exit
        self.temps = cfg.get("temperature", {})       # calibration: {"choice": t, "score": t, "noul": t}
        self.pad_id = tok.pad_token_id

    def info(self) -> Dict:
        return {"name": self.name, "max_len": self.builder.max_len, "optfit": self.optfit, "dtype": str(self.dtype),
                "params_M": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
                "load_s": round(self.load_s, 2), "cuda_graphs": self._graphs is not None}

    def temperature(self, item: Dict, k: int) -> float:
        qt = item["question"]["type"]
        return float(self.temps.get(_bucket(qt, k), self.temps.get(qt, 1.0)))

    # ------------------------------------------------------------------ rows
    def _rows_for(self, state, question) -> List[Dict]:
        """One row normally; several when the options cannot fit one window (chunked scoring)."""
        opts = render_options(question)
        pre, st, head, o_ids = self.builder.parts(question["type"], instructions_text(question), opts,
                                                  serialize_state(state))
        left = isinstance(state, list)
        n_chunks = 1
        while True:
            size = -(-len(opts) // n_chunks)
            rows, ok = [], True
            for c in range(n_chunks):
                idx = list(range(c * size, min(len(opts), (c + 1) * size)))
                if not idx:
                    continue
                r = self.builder.assemble(pre, st, head, [o_ids[i] for i in idx], truncate_left=left)
                if r[4]["overflow"]:
                    ok = False
                    break
                rows.append({"row": r, "opt_index": idx})
            if ok:
                return rows
            n_chunks *= 2

    @torch.no_grad()
    def _forward_rows(self, rows: List[Dict]) -> List[np.ndarray]:
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["row"][0]))
        out: List[Optional[np.ndarray]] = [None] * len(rows)
        s = 0
        while s < len(order):
            e = s
            while e < len(order) and (e - s) < self.max_rows and \
                    len(rows[order[e]]["row"][0]) * (e - s + 1) <= self.max_tokens_per_batch // 2:
                e += 1
            e = max(e, s + 1)
            chunk = [rows[i] for i in order[s:e]]
            b = collate_branched([r["row"] for r in chunk], self.pad_id, qtypes=[r["qtype"] for r in chunk])
            # graphs only for short serving rows: every captured shape keeps its own buffers
            lg = self._graph_forward(b) if self._graphs is not None and len(chunk) <= 4 and b["input_ids"].shape[1] <= 2048 else None
            if lg is None and self.exit:
                with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                    logits, sure = self.model.forward_exit(*(b[k].to(self.device) for k in self._GKEYS),
                                                           exit_layer=self.exit[0], exit_conf=self.exit[1])
                lg = logits.float().cpu().numpy()
                self.exit_stats[0] += int(sure.sum())
                self.exit_stats[1] += len(chunk)
            if lg is None:
                with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                    logits, _ = self.model(*(b[k].to(self.device) for k in self._GKEYS))
                lg = logits.float().cpu().numpy()
            for j, i in enumerate(order[s:e]):
                out[i] = lg[j, :len(rows[i]["opt_index"])]
            s = e
        return out

    # ------------------------------------------------------------------ CUDA graphs
    @torch.no_grad()
    def _graph_forward(self, b) -> Optional[np.ndarray]:
        """Replay a captured CUDA graph for this batch's padded shape (captured on first use). Padding adds only
        branch -2 tokens (never attended to) and masked score slots, so the scores of the real options are the
        same computation. Returns None when the shape is not cached and the cache is full."""
        n, L = b["input_ids"].shape
        K = b["score_pos"].shape[1]
        Lp = -(-L // 64) * 64
        Kp = -(-K // 8) * 8
        key = (n, Lp, Kp)
        ent = self._graphs.get(key)
        if ent is None:
            if len(self._graphs) >= self.graph_cache:
                return None
            st = {"input_ids": torch.full((n, Lp), self.pad_id, dtype=torch.long, device=self.device),
                  "position_ids": torch.zeros((n, Lp), dtype=torch.long, device=self.device),
                  "branch": torch.full((n, Lp), -2, dtype=torch.long, device=self.device),
                  "score_pos": torch.zeros((n, Kp), dtype=torch.long, device=self.device),
                  "score_mask": torch.zeros((n, Kp), dtype=torch.bool, device=self.device),
                  "qtype": torch.zeros((n,), dtype=torch.long, device=self.device)}
            self._fill(st, b, L, K)
            args = [st[k] for k in self._GKEYS]
            try:
                if self.exit:
                    XL = self.exit[0]

                    def run_a():
                        hs, cm, pe, mid = self.model.exit_stage_a(*args, XL)
                        mid = mid.float()
                        return hs, cm, pe, mid, torch.softmax(mid, -1).max(-1).values
                    ga, oa = self._capture(run_a)
                    hs, cm, pe = oa[0], oa[1], oa[2]
                    gb, ob = self._capture(lambda: self.model.exit_stage_b(hs, cm, pe, st["position_ids"], st["score_pos"],
                                                                         st["score_mask"], st["qtype"], XL).float())
                    ent = (st, (ga, oa), (gb, ob))
                else:
                    ent = (st, self._capture(lambda: self.model(*args)[0].float()), None)
            except Exception as e:      # a forward that cannot be captured: serve without graphs from now on
                print("CUDA graph capture failed (%s); serving without graphs" % str(e)[:200], flush=True)
                self._graphs = None
                torch.cuda.synchronize()
                return None
            self._graphs[key] = ent
        st, (ga, oa), stage_b = ent
        self._fill(st, b, L, K)
        ga.replay()
        if stage_b is None:
            return oa[:, :K].cpu().numpy()
        mid, conf = oa[3], oa[4]
        sure = (conf >= self.exit[1]).cpu()
        self.exit_stats[0] += int(sure.sum())
        self.exit_stats[1] += n
        if bool(sure.all()):
            return mid[:, :K].cpu().numpy()
        gb, final = stage_b
        gb.replay()
        return torch.where(sure.to(mid.device)[:, None], mid, final)[:, :K].cpu().numpy()

    def _capture(self, fn):
        """Warm up on a side stream, then capture fn into a CUDA graph (shared memory pool). Returns (graph, outputs)."""
        ac = dict(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.autocast(**ac):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        if not hasattr(self, "_gpool"):
            self._gpool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._gpool), torch.autocast(**ac):
            out = fn()
        return g, out

    def _fill(self, st, b, L, K):
        st["input_ids"].fill_(self.pad_id)
        st["position_ids"].zero_()
        st["branch"].fill_(-2)
        st["score_pos"].zero_()
        st["score_mask"].zero_()
        st["input_ids"][:, :L].copy_(b["input_ids"], non_blocking=True)
        st["position_ids"][:, :L].copy_(b["position_ids"], non_blocking=True)
        st["branch"][:, :L].copy_(b["branch"], non_blocking=True)
        st["score_pos"][:, :K].copy_(b["score_pos"], non_blocking=True)
        st["score_mask"][:, :K].copy_(b["score_mask"], non_blocking=True)
        st["qtype"].copy_(b["qtype"], non_blocking=True)

    # ------------------------------------------------------------------ public API
    def logits(self, items: List[Dict]) -> List[np.ndarray]:
        """Raw (uncalibrated) logits per item; an item is {"state", "question", "keys"}."""
        self.builder.prefetch_states([serialize_state(it["state"]) for it in items])
        rows, owner = [], []
        for i, it in enumerate(items):
            for r in self._rows_for(it["state"], it["question"]):
                r["qtype"] = QTYPES[it["question"]["type"]]
                rows.append(r)
                owner.append(i)
        lg = self._forward_rows(rows)
        out = [np.zeros(len(it["keys"]), dtype=np.float64) for it in items]
        for r, i, z in zip(rows, owner, lg):
            out[i][r["opt_index"]] = z
        return [z.astype(np.float32) for z in out]

    @torch.no_grad()
    def _tree_logits(self, states: List, questions: Dict) -> Optional[List[List[np.ndarray]]]:
        """All of a state's questions in one tree row (the state is encoded once). None if any state's tree
        overflows the window (the caller then scores one row per question)."""
        qlist = list(questions.values())
        trees = []
        for st in states:
            parts = [self.builder.parts(q["type"], instructions_text(q), render_options(q), serialize_state(st))
                     for q in qlist]
            t = assemble_tree(self.builder, parts[0][0], parts[0][1], [p[2] for p in parts], [p[3] for p in parts],
                              truncate_left=isinstance(st, list))
            if t is None:
                return None
            trees.append(t)
        out = []
        for s in range(0, len(trees), self.max_rows):
            chunk = trees[s:s + self.max_rows]
            b = collate_tree(chunk, self.pad_id, [[QTYPES[qlist[g]["type"]] for g in t["group"]] for t in chunk])
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                lg, _ = self.model(*(b[k].to(self.device) for k in ("input_ids", "position_ids", "branch", "score_pos",
                                                                     "score_mask", "qtype", "qid", "score_group")))
            lg = lg.float().cpu().numpy()
            for j, t in enumerate(chunk):
                g = np.array(t["group"])
                out.append([lg[j, :len(g)][g == qi] for qi in range(len(qlist))])
        return out

    def predict_batch(self, states: List, questions: Dict) -> List[Dict]:
        """Answers for every state: {"model", "answers": {question_id: answer}}."""
        items = [{"state": s, "question": q, "keys": option_keys(q), "qid": qid}
                 for s in states for qid, q in questions.items()]
        self.builder.prefetch_states([serialize_state(st) for st in states])
        tree = self._tree_logits(states, questions) if len(questions) > 1 else None
        lg = [z for per_state in tree for z in per_state] if tree is not None else self.logits(items)
        results, j = [], 0
        for _ in states:
            answers = {}
            for qid, q in questions.items():
                it, z = items[j], lg[j]
                j += 1
                p = softmax(z, self.temperature(it, len(z)))
                if q["type"] == "choice":
                    answers[qid] = {"type": "choice", "choice": it["keys"][int(p.argmax())],
                                    "probabilities": dict(zip(it["keys"], p.round(6).tolist())),
                                    "answer_confidence": float(p.max())}
                elif q["type"] == "score":
                    answers[qid] = {"type": "score", "score": float((np.arange(len(p)) * p).sum()),
                                    "probabilities": {str(i): float(v) for i, v in enumerate(p)},
                                    "answer_confidence": float(p.max())}
                else:
                    answers[qid] = {"type": "noul", "noul": float(p[1]), "answer_confidence": float(p.max())}
            results.append({"model": self.name, "answers": answers})
        return results

    def predict(self, state, questions: Dict) -> Dict:
        """Answer one or more questions about one state."""
        return self.predict_batch([state], questions)[0]
