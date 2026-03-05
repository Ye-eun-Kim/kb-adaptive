"""
Build KB sub-graph (triples) for each table from entity_base.
One-hop: all entities linked from table cells + their claims as triples.
"""

import json
import os
from typing import Any

from . import serialization as ser


def _load_entity(entity_base_dir: str, entity_id: str) -> dict | None:
    path = os.path.join(entity_base_dir, f"{entity_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _entity_label(entity_data: dict) -> str:
    return entity_data.get("label", entity_data.get("entity id", ""))


def _claim_values_to_triples(
    entity_id: str,
    entity_label: str,
    prop_id: str,
    prop_label: str,
    values: list,
) -> list[tuple[str, str, str, str | None, str]]:
    """
    Convert one claim to list of (head_id, head_label, rel_label, tail_id_or_None, tail_label_or_value).
    For attribute: tail_id None, tail_label is the raw value string.
    For relational: tail_id set, tail_label is ℓ(tail).
    """
    out = []
    for v in values:
        if not v:
            continue
        datatype = v.get("datatype", "")
        if datatype == "wikibase-item":
            tail_id = v.get("id")
            tail_label = v.get("label", "")
            out.append((entity_id, entity_label, prop_label, tail_id, tail_label))
        else:
            # string, quantity, etc.
            raw = v.get("string", v.get("value", str(v)))
            out.append((entity_id, entity_label, prop_label, None, raw))
    return out


def build_subgraph_triples(
    table_entity_ids: set[str],
    entity_base_dir: str,
) -> list[dict]:
    """
    For a set of entity IDs (from table cells), load each entity's claims and build
    one-hop sub-graph triples. Returns list of triple dicts:
    {
        "head_id": str,
        "head_label": str,
        "rel_label": str,
        "tail_id": str | None,
        "tail_label": str,  # for attribute this is the value
        "serialized": str,  # t* string
        "is_relation": bool,
    }
    """
    triples = []
    seen_serialized = set()
    for eid in table_entity_ids:
        ent = _load_entity(entity_base_dir, eid)
        if not ent:
            continue
        head_label = _entity_label(ent)
        claims = ent.get("claims", {})
        for prop_id, claim in claims.items():
            prop_label = claim.get("label", prop_id)
            values = claim.get("value", [])
            for head_id, hlabel, rel_label, tail_id, tail_label in _claim_values_to_triples(
                eid, head_label, prop_id, prop_label, values
            ):
                if tail_id is not None:
                    s = ser.serialize_relational_triple(hlabel, rel_label, tail_label)
                else:
                    s = ser.serialize_attribute_triple(hlabel, rel_label, tail_label)
                if s in seen_serialized:
                    continue
                seen_serialized.add(s)
                triples.append({
                    "head_id": head_id,
                    "head_label": hlabel,
                    "rel_label": rel_label,
                    "tail_id": tail_id,
                    "tail_label": tail_label,
                    "serialized": s,
                    "is_relation": tail_id is not None,
                })
    return triples


def evidence_to_triple_dict(ev: dict) -> dict | None:
    """
    Convert one selected_evidence_candidate item (type In KB) to a triple dict
    compatible with build_subgraph_triples format, for matching gold evidence.
    """
    if ev.get("type") != "In KB":
        return None
    triple_arr = ev.get("triple")
    if not triple_arr or len(triple_arr) < 3:
        return None
    entity = ev.get("entity", {})
    prop = ev.get("property", {})
    head_label = (entity.get("label") or triple_arr[0]).strip()
    rel_label = (prop.get("label") or triple_arr[1]).strip()
    value = (ev.get("value") or triple_arr[2])
    if isinstance(value, str):
        value = value.strip()
    head_id = entity.get("id", "")
    return {
        "head_id": head_id,
        "head_label": head_label,
        "rel_label": rel_label,
        "tail_id": None,
        "tail_label": value,
        "serialized": ser.serialize_attribute_triple(head_label, rel_label, value),
        "is_relation": False,
    }
