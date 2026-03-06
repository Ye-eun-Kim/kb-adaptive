#!/usr/bin/env python3
"""
KET-QA 테이블 JSON을 CSV로 변환.
header: [["ColName", []], ...]
data: 각 셀은 [value, ...] 또는 [str, link_list, [{"entity id", "label"}]]
"""

import json
import csv
import os
import sys


def cell_to_str(cell):
    """셀(리스트)에서 CSV에 쓸 문자열 하나 추출."""
    if not isinstance(cell, list) or not cell:
        return ""
    first = cell[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return first.get("label", str(first))
    # 리스트면 재귀 (entity가 리스트 안에 있는 경우)
    for item in cell:
        if isinstance(item, dict) and "label" in item:
            return item["label"]
        if isinstance(item, str):
            return item
    return str(cell[0])


def json_table_to_csv(json_path: str, csv_path: str = None) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    header_raw = obj.get("header", [])
    rows_raw = obj.get("data", [])

    # 헤더: [["Year", []], ...] -> ["Year", ...]
    headers = [h[0] if isinstance(h, list) else str(h) for h in header_raw]

    # 각 행: [cell, cell, ...], cell = [value, ...] 또는 [str, link, [entity]]
    rows = []
    for row in rows_raw:
        if not isinstance(row, list):
            continue
        cells = [cell_to_str(cell) for cell in row]
        rows.append(cells)

    if csv_path is None:
        csv_path = os.path.splitext(json_path)[0] + ".csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return csv_path


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "dataset_ketqa", "tables", "Jordan_Brand_Classic_0.json"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else None
    path = json_table_to_csv(json_path, out)
    print("Wrote:", path)
