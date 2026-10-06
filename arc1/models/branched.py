"""Branched-option decision model on a causal decoder backbone (e.g. Qwen3-0.6B).

Layout of one question row:

    prefix  = "State:\n<state>\n\n<Type> question: <instructions>\nAnswer:"
    branch j = " <option j text><|im_end|>"          for every option j

Attention (a 4-D mask, built on device from per-token branch ids):
    * prefix tokens: ordinary causal attention within the prefix;
    * branch j tokens: the whole prefix + earlier tokens of branch j only (never other branches);
    * padding: attends to the prefix only (keeps softmax finite), never attended to.
Position ids: every branch starts at the same position (len(prefix)), so the model sees each option
as "the continuation right after the prefix" -- scores are invariant to option order by construction,
and the prefix is computed once no matter how many options there are.

Each option is scored at its final `<|im_end|>` token: h_j -> [optional position-free set layers over
all options of the question] -> MLP -> logit. Softmax over the question's options.
"""
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from arc1.models.formats import render_options, serialize_state, waterfill_cap

PREFIX_TMPL = "State:\n%s\n\n%s question: %s\nAnswer:"


class SetLayer(nn.Module):
    """Pre-norm self-attention + MLP over option vectors, no positional information."""

    def __init__(self, d: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask, group=None):
        h = self.n1(x)
        if group is None:
            a, _ = self.att(h, h, h, key_padding_mask=~mask, need_weights=False)
        else:
            # options interact only with options of the same question (multi-question rows)
            blocked = (group[:, :, None] != group[:, None, :]) | ~mask[:, None, :]
            eye = torch.eye(group.size(1), dtype=torch.bool, device=group.device)[None]
            blocked = blocked & ~eye                      # padding slots attend to themselves: finite softmax
            a, _ = self.att(h, h, h, attn_mask=blocked.repeat_interleave(self.att.num_heads, 0), need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class BranchedDecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, set_layers: int = 1, type_emb: bool = True):
        super().__init__()
        self.backbone = backbone
        d = backbone.config.hidden_size
        self.set_layers = nn.ModuleList([SetLayer(d, heads=max(1, d // 128)) for _ in range(set_layers)])
        self.type_emb = nn.Embedding(3, d) if type_emb else None
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    @staticmethod
    def build_mask(branch: torch.Tensor, dtype, qid: torch.Tensor = None) -> torch.Tensor:
        """branch: [n, L] long (-1 prefix, j >= 0 option branch, -2 padding) -> additive mask [n, 1, L, L].

        With `qid` ([n, L]: -1 shared prefix, q >= 0 question q, -2 padding) the row is a tree: the shared
        prefix (state), one sub-prefix per question (its header, branch -1) and one branch per option.
        """
        n, L = branch.shape
        idx = torch.arange(L, device=branch.device)
        causal = idx[None, :] <= idx[:, None]                       # [L, L] s <= t
        bs, bt = branch[:, None, :], branch[:, :, None]             # [n, 1, L] (key), [n, L, 1] (query)
        if qid is None:
            allowed = causal[None] & ((bs == -1) | (bs == bt)) & (bs != -2)
        else:
            qs, qt = qid[:, None, :], qid[:, :, None]
            allowed = causal[None] & ((qs == -1) | ((qs == qt) & ((bs == -1) | (bs == bt)))) & (bs != -2)
        mask = torch.zeros((n, 1, L, L), dtype=dtype, device=branch.device)
        mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        return mask

    def forward_split(self, input_ids, position_ids, branch, score_pos, score_mask, qtype, qid=None, score_group=None):
        """Prefix-split training forward: the shared prefix (state + question header, branch == -1) is encoded once
        without gradient and kept as a KV cache; only the option branches run with gradient on top of it.
        Supervision sits only at the branch ends, and the prefix is ~70% of the tokens, so the backward pass and
        the activation memory shrink to the branch tokens. Scores equal the full forward (same weights, same
        attention pattern); only the gradient path through the prefix encoding is dropped.
        Tree rows (`qid` given): the shared prefix is the state (qid == -1); question headers run with the branches.
        Works with gradient checkpointing: after the prefix pass the cache is frozen (each layer's update returns
        prefix + new keys without storing them), so a recomputed layer reads exactly the same states."""
        from transformers import DynamicCache
        dev, n = input_ids.device, input_ids.size(0)
        pre = (qid == -1) if qid is not None else (branch == -1)
        P = pre.sum(1)                                                   # prefix length per row (contiguous at start)
        Lb = ((branch != -2) & ~pre).sum(1)
        Pm, Lm = int(P.max()), int(Lb.max())
        ar_p = torch.arange(Pm, device=dev)
        ar_b = torch.arange(Lm, device=dev)
        mdt = torch.bfloat16 if torch.is_autocast_enabled() else self.scorer[1].weight.dtype
        neg = torch.finfo(mdt).min
        # 1) prefix, no gradient (eval mode so the backbone keeps the cache)
        p_ids, p_pos = input_ids[:, :Pm], position_ids[:, :Pm]
        p_ok = ar_p[None, :] < P[:, None]
        p_allow = (ar_p[None, :] <= ar_p[:, None])[None] & p_ok[:, None, :]
        p_mask = torch.zeros((n, 1, Pm, Pm), dtype=mdt, device=dev)
        p_mask.masked_fill_(~p_allow[:, None], neg)
        cache = DynamicCache()
        was = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            self.backbone(input_ids=p_ids, position_ids=p_pos, attention_mask=p_mask, past_key_values=cache, use_cache=True)
        self.backbone.train(was)
        for lay in cache.layers:     # read-only from here: update() returns prefix + new keys and stores nothing
            lay.update = (lambda k, v, *a, _pk=lay.keys, _pv=lay.values, **kw: (torch.cat([_pk, k], -2), torch.cat([_pv, v], -2)))
        # 2) branches with gradient, attending to their row's prefix KV and to earlier tokens of their own branch
        gather = (P[:, None] + ar_b[None, :]).clamp(max=input_ids.size(1) - 1)
        b_ok = ar_b[None, :] < Lb[:, None]
        b_ids = torch.gather(input_ids, 1, gather)
        b_pos = torch.gather(position_ids, 1, gather)
        b_br = torch.where(b_ok, torch.gather(branch, 1, gather), torch.full_like(gather, -2))
        if qid is None:
            same = (b_br[:, :, None] == b_br[:, None, :])
        else:   # same question, and the key is that question's header or the query's own branch
            b_q = torch.gather(qid, 1, gather)
            same = (b_q[:, :, None] == b_q[:, None, :]) & ((b_br[:, None, :] == -1) | (b_br[:, :, None] == b_br[:, None, :]))
        same = same & (ar_b[None, :] <= ar_b[:, None])[None] & b_ok[:, None, :]
        allow = torch.cat([p_ok[:, None, :].expand(n, Lm, Pm), same], 2)
        allow = allow | (~b_ok)[:, :, None] & (torch.arange(Pm + Lm, device=dev) == 0)[None, None, :]   # padded queries: one key
        b_mask = torch.zeros((n, 1, Lm, Pm + Lm), dtype=p_mask.dtype, device=dev)
        b_mask.masked_fill_(~allow[:, None], neg)
        from transformers.modeling_layers import GradientCheckpointingLayer
        ckl = [m for m in self.backbone.modules() if isinstance(m, GradientCheckpointingLayer)]
        for m in ckl:
            m._can_checkpoint_with_cache = True      # the frozen cache is only read, so the recompute sees the same states
        try:
            out = self.backbone(input_ids=b_ids, position_ids=b_pos, attention_mask=b_mask, past_key_values=cache, use_cache=True)
        finally:
            for m in ckl:
                m.__dict__.pop("_can_checkpoint_with_cache", None)
        h = out.last_hidden_state
        rel = (score_pos - P[:, None]).clamp(min=0, max=Lm - 1)
        v = torch.gather(h, 1, rel[:, :, None].expand(-1, -1, h.size(-1))).float()
        if self.type_emb is not None:
            te = self.type_emb(qtype).float()
            v = v + (te[:, None, :] if qtype.dim() == 1 else te)
        for layer in self.set_layers:
            v = layer(v, score_mask, score_group)
        logits = self.scorer(v).squeeze(-1)
        return logits.masked_fill(~score_mask, -1e4), None

    def _head(self, h, score_pos, score_mask, qtype, score_group=None):
        idx = score_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        v = torch.gather(h, 1, idx).float()
        if self.type_emb is not None:
            te = self.type_emb(qtype).float()
            v = v + (te[:, None, :] if qtype.dim() == 1 else te)
        for layer in self.set_layers:
            v = layer(v, score_mask, score_group)
        return self.scorer(v).squeeze(-1).masked_fill(~score_mask, -1e4)

    def exit_stage_a(self, input_ids, position_ids, branch, score_pos, score_mask, qtype, exit_layer: int):
        """Layers [0, exit_layer) + the head on their output. Returns (hidden, mask, rotary, logits at the exit layer)."""
        from transformers.masking_utils import create_causal_mask
        bb = self.backbone
        dt = torch.bfloat16 if torch.is_autocast_enabled() else self.scorer[1].weight.dtype
        hs = bb.embed_tokens(input_ids)
        cm = create_causal_mask(config=bb.config, inputs_embeds=hs, attention_mask=self.build_mask(branch, dt),
                                past_key_values=None, position_ids=position_ids)
        pe = bb.rotary_emb(hs, position_ids)
        for layer in bb.layers[:exit_layer]:
            hs = layer(hs, attention_mask=cm, position_embeddings=pe, position_ids=position_ids, past_key_values=None, use_cache=False)
        return hs, cm, pe, self._head(bb.norm(hs), score_pos, score_mask, qtype)

    def exit_stage_b(self, hs, cm, pe, position_ids, score_pos, score_mask, qtype, exit_layer: int):
        """The remaining layers and the final head."""
        bb = self.backbone
        for layer in bb.layers[exit_layer: bb.config.num_hidden_layers]:
            hs = layer(hs, attention_mask=cm, position_embeddings=pe, position_ids=position_ids, past_key_values=None, use_cache=False)
        return self._head(bb.norm(hs), score_pos, score_mask, qtype)

    @torch.no_grad()
    def forward_exit(self, input_ids, position_ids, branch, score_pos, score_mask, qtype, exit_layer: int, exit_conf: float):
        """Calibrated early exit (inference): score the options after `exit_layer` layers with the final norm + head
        ("logit lens"). Rows whose top option already has probability >= exit_conf keep those scores; if every row is
        that sure, the remaining layers are skipped. Returns (logits, exited rows mask)."""
        n = self.backbone.config.num_hidden_layers
        L = min(exit_layer, n)
        hs, cm, pe, mid = self.exit_stage_a(input_ids, position_ids, branch, score_pos, score_mask, qtype, L)
        if L >= n:
            return mid, torch.zeros(mid.size(0), dtype=torch.bool, device=mid.device)
        sure = torch.softmax(mid, -1).max(-1).values >= exit_conf
        if bool(sure.all()):
            return mid, sure
        final = self.exit_stage_b(hs, cm, pe, position_ids, score_pos, score_mask, qtype, L)
        return torch.where(sure[:, None], mid, final), sure

    def forward(self, input_ids, position_ids, branch, score_pos, score_mask, qtype, qid=None, score_group=None):
        mask = self.build_mask(branch, torch.bfloat16 if torch.is_autocast_enabled() else self.scorer[1].weight.dtype, qid)
        out = self.backbone(input_ids=input_ids, position_ids=position_ids, attention_mask=mask, use_cache=False)
        h = out.last_hidden_state
        idx = score_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        v = torch.gather(h, 1, idx).float()
        if self.type_emb is not None:
            te = self.type_emb(qtype).float()                          # [n, d] or [n, K, d] (per option slot)
            v = v + (te[:, None, :] if qtype.dim() == 1 else te)
        for layer in self.set_layers:
            v = layer(v, score_mask, score_group)
        logits = self.scorer(v).squeeze(-1)
        return logits.masked_fill(~score_mask, -1e4), None


class BranchedBuilder:
    """Tokenises and assembles branched rows (ids, position ids, branch ids, score positions)."""

    def __init__(self, tok, max_len: int = 1024, opt_cap: int = 48, min_state: int = 160):
        self.tok = tok
        self.max_len, self.opt_cap, self.min_state = max_len, opt_cap, min_state
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")
        assert self.end_id is not None and self.end_id != tok.unk_token_id

    def encode_many(self, texts: List[str]) -> List[List[int]]:
        return self.tok(texts, add_special_tokens=False)["input_ids"] if texts else []

    def prefetch_states(self, texts: List[str]):
        """Tokenise all distinct, not-yet-cached states in one batched call (throughput path)."""
        cache = self.__dict__.setdefault("_scache", {})
        todo = list(dict.fromkeys(t for t in texts if t not in cache))
        if len(cache) + len(todo) > 50000:
            cache.clear()
        for s in range(0, len(todo), 1024):
            chunk = todo[s:s + 1024]
            for t, ids in zip(chunk, self.encode_many(chunk)):
                cache[t] = ids

    def parts(self, qtype: str, instructions: str, options: List[str], state_text: str):
        cache = self.__dict__.get("_scache")
        state_ids = cache.get(state_text) if cache else None
        if state_ids is None:
            state_ids = self.encode_many([state_text])[0]
        head, opts = self._question_parts(qtype, instructions, tuple(options))
        if not hasattr(self, "_pre"):
            self._pre = self.encode_many(["State:\n"])[0]
        return self._pre, state_ids, head, opts

    def _question_parts(self, qtype: str, instructions: str, options: tuple):
        """Question header + option token ids, cached: the same question is usually asked of many states."""
        cache = self.__dict__.setdefault("_qcache", {})
        key = (qtype, instructions, options)
        hit = cache.get(key)
        if hit is None:
            head = self.encode_many(["\n\n%s question: %s\nAnswer:" % (qtype.capitalize(), instructions)])[0]
            opts = [x[:self.opt_cap] for x in self.encode_many([" " + o for o in options])]
            if len(cache) > 4096:
                cache.clear()
            hit = cache[key] = (head, opts)
        return hit

    def assemble(self, pre, state_ids, head, opts, max_len=None, truncate_left=False):
        head = head[:256]
        if max_len is None and getattr(self, "dyn_cap", None):
            # option fit (inference only): widen the row so that no option is truncated, up to dyn_cap tokens.
            # Options are parallel branches whose positions restart after the prefix, so a wider row adds no
            # position the model has not seen; it only stops long option lists from being cut to 1-2 tokens each.
            need = len(pre) + len(head) + sum(min(len(o), self.opt_cap) + 1 for o in opts) + min(len(state_ids), self.max_len // 2)
            max_len = max(self.max_len, min(self.dyn_cap, need))
        max_len = max_len or self.max_len
        opt_room = max_len - len(pre) - len(head) - min(self.min_state, len(state_ids)) - len(opts)
        cap = waterfill_cap([len(o) for o in opts], max(0, opt_room))
        cut = [o[:cap] + [self.end_id] for o in opts]
        room = max(0, max_len - len(pre) - len(head) - sum(len(o) for o in cut))
        st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
        prefix = pre + st + head
        P = len(prefix)
        ids, pos, br, score = list(prefix), list(range(P)), [-1] * P, []
        for j, o in enumerate(cut):
            ids += o
            pos += list(range(P, P + len(o)))
            br += [j] * len(o)
            score.append(len(ids) - 1)
        info = {"opt_cap": cap, "state_truncated": len(state_ids) - len(st), "overflow": len(ids) > max_len}
        return ids, pos, br, score, info

    def build(self, state, question: Dict):
        ins = question["instructions"] if isinstance(question["instructions"], str) else json.dumps(
            question["instructions"], ensure_ascii=False)
        pre, st, head, opts = self.parts(question["type"], ins, render_options(question), serialize_state(state))
        return self.assemble(pre, st, head, opts, truncate_left=isinstance(state, list))


def assemble_tree(builder: "BranchedBuilder", pre, state_ids, heads: List[List[int]], opts: List[List[List[int]]],
                  truncate_left: bool = False):
    """One row answering several questions about one state: state encoded once (shared prefix), each
    question's header as a sub-prefix, each option as a branch of its question. Returns None when the
    tree does not fit `builder.max_len` (the caller falls back to one row per question)."""
    heads = [h[:256] for h in heads]
    cut = [[o + [builder.end_id] for o in q] for q in opts]
    need = len(pre) + sum(len(h) for h in heads) + sum(len(o) for q in cut for o in q)
    room = builder.max_len - need
    if room < min(builder.min_state, len(state_ids)):
        return None
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    prefix = pre + st
    P = len(prefix)
    ids, pos, br, qid = list(prefix), list(range(P)), [-1] * P, [-1] * P
    score, group, g = [], [], 0
    for qi, (h, q) in enumerate(zip(heads, cut)):
        ids += h; pos += list(range(P, P + len(h))); br += [-1] * len(h); qid += [qi] * len(h)
        H = P + len(h)
        for o in q:
            ids += o; pos += list(range(H, H + len(o))); br += [g] * len(o); qid += [qi] * len(o)
            score.append(len(ids) - 1); group.append(qi); g += 1
    return {"ids": ids, "pos": pos, "branch": br, "qid": qid, "score": score, "group": group,
            "state_truncated": len(state_ids) - len(st)}


def collate_tree(rows: List[Dict], pad_id: int, qtypes_per_slot: List[List[int]]):
    n = len(rows)
    L = max(len(r["ids"]) for r in rows)
    K = max(len(r["score"]) for r in rows)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    pos = torch.zeros((n, L), dtype=torch.long)
    br = torch.full((n, L), -2, dtype=torch.long)
    qd = torch.full((n, L), -2, dtype=torch.long)
    sp = torch.zeros((n, K), dtype=torch.long)
    sm = torch.zeros((n, K), dtype=torch.bool)
    sg = torch.full((n, K), -1, dtype=torch.long)
    qt = torch.zeros((n, K), dtype=torch.long)
    for i, r in enumerate(rows):
        m = len(r["ids"]); k = len(r["score"])
        ids[i, :m] = torch.tensor(r["ids"]); pos[i, :m] = torch.tensor(r["pos"])
        br[i, :m] = torch.tensor(r["branch"]); qd[i, :m] = torch.tensor(r["qid"])
        sp[i, :k] = torch.tensor(r["score"]); sm[i, :k] = True
        sg[i, :k] = torch.tensor(r["group"]); qt[i, :k] = torch.tensor(qtypes_per_slot[i])
    return {"input_ids": ids, "position_ids": pos, "branch": br, "score_pos": sp, "score_mask": sm,
            "qtype": qt, "qid": qd, "score_group": sg}


def collate_branched(rows: List[Tuple], pad_id: int, targets: Optional[List] = None, qtypes: Optional[List[int]] = None):
    n = len(rows)
    L = max(len(r[0]) for r in rows)
    K = max(len(r[3]) for r in rows)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    pos = torch.zeros((n, L), dtype=torch.long)
    br = torch.full((n, L), -2, dtype=torch.long)
    sp = torch.zeros((n, K), dtype=torch.long)
    sm = torch.zeros((n, K), dtype=torch.bool)
    for i, (a, p, b, s, _info) in enumerate(rows):
        ids[i, :len(a)] = torch.tensor(a)
        pos[i, :len(p)] = torch.tensor(p)
        br[i, :len(b)] = torch.tensor(b)
        sp[i, :len(s)] = torch.tensor(s)
        sm[i, :len(s)] = True
    out = {"input_ids": ids, "position_ids": pos, "branch": br, "score_pos": sp, "score_mask": sm,
           "qtype": torch.tensor(qtypes if qtypes is not None else [0] * n, dtype=torch.long)}
    if targets is not None:
        t = torch.zeros((n, K), dtype=torch.float32)
        for i, tt in enumerate(targets):
            t[i, :len(tt)] = torch.as_tensor(tt, dtype=torch.float32)
        out["target"] = t
    return out
