"""
Input serialization for KET-QA Retriever (paper Section 4.3).
- Triple: relational t1* and attribute t2* with [HEAD],[REL],[TAIL]
- Table: T* with [HEAD], columns, [ROW], index, row cells
- Sub-table: rows where head entity appears (for triple-related sub-table)
"""

# Special tokens (should be added to tokenizer vocab or use existing tokens)
HEAD_TOKEN = "[HEAD]"
REL_TOKEN = "[REL]"
TAIL_TOKEN = "[TAIL]"
ROW_TOKEN = "[ROW]"


def serialize_relational_triple(head_label: str, relation_label: str, tail_label: str) -> str:
    """
    Relational triple t1 = (e1, r, e2) -> t1* = [HEAD], ℓ(e1), [REL], ℓ(r), [TAIL], ℓ(e2)
    """
    return f"{HEAD_TOKEN} {head_label} {REL_TOKEN} {relation_label} {TAIL_TOKEN} {tail_label}"


def serialize_attribute_triple(entity_label: str, attribute_label: str, value: str) -> str:
    """
    Attribute triple t2 = (e, a, v) -> t2* = [HEAD], ℓ(e), [REL], ℓ(a), [TAIL], v
    """
    return f"{HEAD_TOKEN} {entity_label} {REL_TOKEN} {attribute_label} {TAIL_TOKEN} {value}"


def serialize_table(header: list, rows: list, row_indices_one_based: bool = True) -> str:
    """
    Table T -> T* = [HEAD], c1, ..., cN, [ROW], 1, r1, [ROW], 2, r2, ...
    - header: list of column names (strings)
    - rows: list of row lists (each row = list of cell display strings)
    - row index starts from 1 after [ROW]
    """
    col_names = [h[0] if isinstance(h, (list, tuple)) else str(h) for h in header]
    head_part = f"{HEAD_TOKEN} " + " , ".join(col_names)
    row_parts = []
    for i, row in enumerate(rows):
        idx = (i + 1) if row_indices_one_based else i
        cell_strs = [str(c[0]) if isinstance(c, (list, tuple)) and len(c) else str(c) for c in row]
        row_str = " , ".join(cell_strs)
        row_parts.append(f"{ROW_TOKEN} {idx} {row_str}")
    return head_part + " " + " ".join(row_parts)


def get_cell_display_value(cell) -> str:
    """Extract display text from a table cell (data row cell format)."""
    if isinstance(cell, list):
        if len(cell) >= 1:
            first = cell[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict) and "label" in first:
                return first["label"]
        return ""
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict) and "label" in cell:
        return cell["label"]
    return str(cell)


def get_entity_ids_from_row(row) -> set:
    """From a table row (list of cells), collect all entity ids linked to cells."""
    entity_ids = set()
    for cell in row:
        if not isinstance(cell, list) or len(cell) < 3:
            continue
        # cell = [display_text, [links], [{"entity id": "Q...", "label": "..."}, ...]
        for item in cell[2]:
            if isinstance(item, dict) and "entity id" in item:
                entity_ids.add(item["entity id"])
    return entity_ids


def extract_sub_table_by_entity(table_data: list, header: list, head_entity_id: str) -> tuple:
    """
    Sub-table: keep only rows where head_entity_id appears in some cell.
    Returns (header, list of rows) where each row is list of display values.
    """
    out_rows = []
    for row in table_data:
        row_entity_ids = get_entity_ids_from_row(row)
        if head_entity_id in row_entity_ids:
            out_rows.append([get_cell_display_value(c) for c in row])
    return header, out_rows


def table_to_serialized_with_subtable(
    header: list,
    data: list,
    head_entity_id: str | None,
) -> str:
    """
    Serialize table. If head_entity_id is given, use triple-related sub-table (only rows
    containing that entity). Otherwise serialize full table.
    """
    if head_entity_id:
        sub_header, sub_rows = extract_sub_table_by_entity(data, header, head_entity_id)
        return serialize_table(sub_header, sub_rows)
    rows_as_cells = [[get_cell_display_value(c) for c in row] for row in data]
    return serialize_table(header, rows_as_cells)
