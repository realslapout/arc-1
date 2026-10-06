"""Sequence construction for ARC-1 cross-encoder decision models.

Layout (identical to Laya's so Laya checkpoints can be warm-started):

    [CLS] <qtype> question: <instructions> [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP] <state> [SEP]

The difference is the token budget. Laya gives all options a fixed `head_max_len` (192/256) and,
once they overflow it, cuts every option to (head_max_len - 16) // k tokens: 77 options get 2-4
tokens each and many become identical (issue #543). ARC-1 budgets the whole sequence instead:

  1. options keep up to `opt_cap` tokens each;
  2. if options + instructions + `min_state` tokens overflow `max_len`, the per-option cap is
     lowered by water-filling (short options keep all their tokens, only long ones are cut);
  3. the state gets whatever room is left (at least `min_state` unless options genuinely need it).

`option_order` permutes options for augmentation; markers follow the rendered order.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question: Dict) -> List[str]:
    """Laya-compatible option texts for a wire-format question (see laya.common.render_options)."""
    t, crit = question["type"], question.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            return [str(c) for c in crit]
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    labels = question.get("labels") or {"false": "false", "true": "true"}
    f, tr = crit.get("false"), crit.get("true")
    return [labels["false"] + ": " + (render_criterion(f) if f not in (None, "") else "no, the statement does not hold"),
            labels["true"] + ": " + (render_criterion(tr) if tr not in (None, "") else "yes, the statement holds")]


def waterfill_cap(lengths: Sequence[int], budget: int, floor: int = 2) -> int:
    """Largest per-item cap c >= floor with sum(min(l, c)) <= budget (floor if even that overflows)."""
    if sum(lengths) <= budget:
        return max(lengths) if lengths else floor
    lo, hi = floor, max(lengths)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(l, mid) for l in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


class SequenceBuilder:
    def __init__(self, tok, max_len: int = 1024, opt_cap: int = 48, min_state: int = 160):
        self.tok = tok
        self.max_len, self.opt_cap, self.min_state = max_len, opt_cap, min_state
        self.mask_id, self.cls_id, self.sep_id = tok.mask_token_id, tok.cls_token_id, tok.sep_token_id
        self.mask_tok = tok.mask_token

    def encode(self, text: str) -> List[int]:
        return self.tok(text.replace(self.mask_tok, " "), add_special_tokens=False)["input_ids"]

    def encode_many(self, texts: List[str]) -> List[List[int]]:
        if not texts:
            return []
        return self.tok([t.replace(self.mask_tok, " ") for t in texts], add_special_tokens=False)["input_ids"]

    def head_ids(self, qtype: str, instructions: str) -> List[int]:
        return self.encode("%s question: %s" % (qtype, instructions))

    def option_ids(self, option_texts: List[str]) -> List[List[int]]:
        return [ids[:self.opt_cap] for ids in self.encode_many([" " + o for o in option_texts])]

    def assemble(self, head: List[int], opts: List[List[int]], state: List[int], truncate_left: bool = False,
                 max_len: Optional[int] = None, head_cap: int = 256) -> Tuple[List[int], List[int], Dict]:
        """Assemble pre-tokenized parts into (ids, marker positions, info)."""
        max_len = max_len or self.max_len
        head = head[:head_cap]
        fixed = 1 + len(head) + 1 + 1 + 1          # CLS, head, SEP, SEP after options, final SEP
        opt_room = max_len - fixed - min(self.min_state, len(state)) - len(opts)   # minus [MASK] per option
        cap = waterfill_cap([len(o) for o in opts], max(0, opt_room))
        cut = [o[:cap] for o in opts]
        ids = [self.cls_id] + head + [self.sep_id]
        markers = []
        for o in cut:
            markers.append(len(ids))
            ids.append(self.mask_id)
            ids.extend(o)
        ids.append(self.sep_id)
        room = max(0, max_len - len(ids) - 1)
        st = state[max(0, len(state) - room):] if truncate_left else state[:room]
        ids = ids + st + [self.sep_id]
        info = {"opt_cap": cap, "opt_truncated": sum(len(o) > cap for o in opts),
                "state_truncated": len(state) - len(st), "overflow": len(ids) > max_len}
        if len(ids) > max_len:
            # Options alone exceed the window even at the floor cap: the caller must chunk.
            ids = ids[:max_len]
            markers = [m for m in markers if m < max_len]
        return ids, markers, info

    def build(self, state, question: Dict, option_order: Optional[List[int]] = None, max_len: Optional[int] = None):
        opts = render_options(question)
        order = option_order if option_order is not None else list(range(len(opts)))
        ins = question["instructions"] if isinstance(question["instructions"], str) else json.dumps(
            question["instructions"], ensure_ascii=False)
        head = self.head_ids(question["type"], ins)
        o_ids = self.option_ids([opts[i] for i in order])
        s_ids = self.encode(serialize_state(state))
        return self.assemble(head, o_ids, s_ids, truncate_left=isinstance(state, list), max_len=max_len)
