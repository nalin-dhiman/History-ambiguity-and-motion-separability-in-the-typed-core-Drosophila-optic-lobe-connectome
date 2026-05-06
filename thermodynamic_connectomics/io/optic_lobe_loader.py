"""Robust loader for optic-lobe v1.1 neuPrint export tables.

The loader intentionally inspects the available columns and reports the mapping
it chooses. It avoids treating anatomical fan-in as literal heat: this module
only prepares a canonical directed weighted graph for later, explicit
coarse-graining analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.feather as feather

from thermodynamic_connectomics.io.roi import assign_major_roi_series


SOURCE_CANDIDATES = (
    ":START_ID(Body-ID)",
    "bodyId_pre",
    "bodyId_pre:string",
    "bodyId_pre:long",
    "pre",
    "from",
    "source",
    "src",
    "pre_id",
    "bodyId",
)
TARGET_CANDIDATES = (
    ":END_ID(Body-ID)",
    "bodyId_post",
    "bodyId_post:string",
    "bodyId_post:long",
    "post",
    "to",
    "target",
    "dst",
    "post_id",
)
WEIGHT_CANDIDATES = (
    "weight",
    "weight:int",
    "synapse_count",
    "syn_count",
    "count",
    "count:int",
    "n_synapses",
)
NODE_ID_CANDIDATES = (
    ":ID(Body-ID)",
    "bodyId:long",
    "bodyId",
    "root_id",
    "id",
)
TYPE_CANDIDATES = ("type:string", "type", "flywireType:string", "cell_type", "celltype")
INSTANCE_CANDIDATES = ("instance:string", "instance")
NT_CANDIDATES = (
    "consensusNt:string",
    "consensusNt",
    "celltypePredictedNt:string",
    "predictedNt:string",
    "nt",
)
ROI_CANDIDATES = ("roi:string", "primaryRoi:string", "primary_roi", "roi")
ROI_INFO_CANDIDATES = ("roiInfo:string", "roiInfo", "roi_info")
PREFERRED_COLUMNS = {
    "node id": (":ID(Body-ID)", "bodyId:long", "bodyId"),
    "cell type": ("type:string", "type", "flywireType:string"),
    "instance": ("instance:string", "instance"),
    "neurotransmitter": ("consensusNt:string", "consensusNt", "celltypePredictedNt:string", "predictedNt:string"),
    "roi": ("roi:string", "primaryRoi:string", "primary_roi", "roi"),
    "roi info": ("roiInfo:string", "roiInfo", "roi_info"),
    "weight": ("weight:int", "weight"),
}


@dataclass(frozen=True)
class ColumnMapping:
    """Detected source, target, and weight columns for an edge table."""

    source: str
    target: str
    weight: str


def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_column(columns: Iterable[str], candidates: Iterable[str], label: str, required: bool = True) -> str | None:
    cols = list(columns)
    exact = [c for c in candidates if c in cols]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        for preferred in PREFERRED_COLUMNS.get(label, ()):
            if preferred in exact:
                return preferred
        raise ValueError(f"Ambiguous {label} columns {exact}. Available columns: {cols}")

    norm_to_cols: dict[str, list[str]] = {}
    for col in cols:
        norm_to_cols.setdefault(_norm(col), []).append(col)
    matches: list[str] = []
    for cand in candidates:
        matches.extend(norm_to_cols.get(_norm(cand), []))
    matches = sorted(set(matches), key=cols.index)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for preferred in PREFERRED_COLUMNS.get(label, ()):
            if preferred in matches:
                return preferred
        raise ValueError(f"Ambiguous {label} columns {matches}. Available columns: {cols}")
    if required:
        raise ValueError(f"Could not detect {label} column. Available columns: {cols}")
    return None


def infer_column_mapping(edge_columns: Iterable[str]) -> ColumnMapping:
    """Infer source, target, and weight columns from a neuPrint edge table."""

    columns = list(edge_columns)
    return ColumnMapping(
        source=_find_column(columns, SOURCE_CANDIDATES, "source"),
        target=_find_column(columns, TARGET_CANDIDATES, "target"),
        weight=_find_column(columns, WEIGHT_CANDIDATES, "weight"),
    )


def _read_table(path: Path, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path, columns=columns)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        usecols = columns if columns else None
        return pd.read_csv(path, usecols=usecols, nrows=nrows)
    raise ValueError(f"Unsupported table format: {path}")


def _read_header(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".feather":
        return pd.DataFrame(columns=feather.read_table(path, columns=[]).schema.names)
    if path.suffix.lower() == ".parquet":
        return pd.DataFrame(columns=pd.read_parquet(path, columns=[]).columns.tolist())
    return pd.read_csv(path, nrows=0)


def _edge_path(data_dir: Path) -> Path:
    feather = data_dir / "Neuprint_Neuron_Connections.feather"
    csv = data_dir / "Neuprint_Neuron_Connections.csv"
    if feather.exists():
        return feather
    if csv.exists():
        return csv
    raise FileNotFoundError(f"No Neuprint_Neuron_Connections feather/csv found in {data_dir}")


def _node_path(data_dir: Path) -> Path:
    feather = data_dir / "Neuprint_Neurons.feather"
    if feather.exists():
        return feather
    csv = data_dir / "Neuprint_Neurons.csv"
    if csv.exists():
        return csv
    raise FileNotFoundError(f"No Neuprint_Neurons feather/csv found in {data_dir}")


def load_optic_lobe_edges(
    data_dir: str | Path,
    min_weight: float = 1.0,
    max_edges: int | None = None,
    sample_nodes: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load a canonical `src`, `dst`, `weight` edge dataframe."""

    data_dir = Path(data_dir)
    path = _edge_path(data_dir)
    header = _read_header(path)
    mapping = infer_column_mapping(header.columns)
    roi_info_col = _find_column(header.columns, ROI_INFO_CANDIDATES, "roi info", required=False)
    print(
        "Detected edge columns: "
        f"src={mapping.source!r}, dst={mapping.target!r}, weight={mapping.weight!r}"
    )
    read_cols = [mapping.source, mapping.target, mapping.weight]
    if roi_info_col is not None:
        read_cols.append(roi_info_col)
    raw = _read_table(path, columns=read_cols)
    rename = {mapping.source: "src", mapping.target: "dst", mapping.weight: "weight"}
    if roi_info_col is not None:
        rename[roi_info_col] = "edge_roi_info"
    edges = raw.rename(columns=rename)
    edges["src"] = pd.to_numeric(edges["src"], errors="coerce")
    edges["dst"] = pd.to_numeric(edges["dst"], errors="coerce")
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges = edges.dropna(subset=["src", "dst", "weight"])
    edges = edges[edges["weight"] >= min_weight]
    edges = edges.astype({"src": "int64", "dst": "int64"})
    agg = {"weight": "sum"}
    if "edge_roi_info" in edges.columns:
        agg["edge_roi_info"] = "first"
    edges = edges.groupby(["src", "dst"], as_index=False, sort=False).agg(agg)
    edges = edges[edges["weight"] > 0].reset_index(drop=True)

    if sample_nodes is not None and sample_nodes > 0:
        rng = pd.Series(pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True))).sample(
            n=min(sample_nodes, edges[["src", "dst"]].stack().nunique()),
            random_state=seed,
        )
        keep = set(rng.astype("int64").tolist())
        edges = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)].reset_index(drop=True)

    if max_edges is not None and max_edges > 0 and len(edges) > max_edges:
        edges = edges.sample(n=max_edges, random_state=seed).reset_index(drop=True)
    return edges, mapping


def _available_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".feather":
        return feather.read_table(path, columns=[]).schema.names
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_optic_lobe_nodes(data_dir: str | Path, node_ids: Iterable[int] | None = None) -> pd.DataFrame:
    """Load canonical body/segment metadata for the requested body IDs."""

    data_dir = Path(data_dir)
    path = _node_path(data_dir)
    columns = _available_columns(path)
    id_col = _find_column(columns, NODE_ID_CANDIDATES, "node id")
    type_col = _find_column(columns, TYPE_CANDIDATES, "cell type", required=False)
    instance_col = _find_column(columns, INSTANCE_CANDIDATES, "instance", required=False)
    nt_col = _find_column(columns, NT_CANDIDATES, "neurotransmitter", required=False)
    roi_col = _find_column(columns, ROI_CANDIDATES, "roi", required=False)
    usecols = [c for c in [id_col, type_col, instance_col, nt_col, roi_col] if c is not None]
    print(
        "Detected node columns: "
        f"id={id_col!r}, type={type_col!r}, instance={instance_col!r}, roi={roi_col!r}, nt={nt_col!r}"
    )
    raw = _read_table(path, columns=usecols)
    rename = {id_col: "bodyId"}
    if type_col:
        rename[type_col] = "type"
    if instance_col:
        rename[instance_col] = "instance"
    if roi_col:
        rename[roi_col] = "roi"
    if nt_col:
        rename[nt_col] = "nt"
    nodes = raw.rename(columns=rename)
    nodes["bodyId"] = pd.to_numeric(nodes["bodyId"], errors="coerce")
    nodes = nodes.dropna(subset=["bodyId"]).astype({"bodyId": "int64"})
    if node_ids is not None:
        keep = pd.Index(pd.Series(list(node_ids), dtype="int64").unique())
        nodes = nodes[nodes["bodyId"].isin(keep)]
    for col in ["type", "instance", "roi", "nt"]:
        if col not in nodes.columns:
            nodes[col] = pd.NA
    nodes["major_roi"] = pd.NA
    nodes = nodes.drop_duplicates("bodyId").reset_index(drop=True)
    return nodes[["bodyId", "type", "instance", "roi", "major_roi", "nt"]]


def attach_metadata(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Add source/target metadata columns to a canonical edge dataframe."""

    out = edges.copy()
    meta = nodes.set_index("bodyId")
    for side, id_col in [("src", "src"), ("dst", "dst")]:
        out[f"{side}_type"] = out[id_col].map(meta["type"]) if "type" in meta else pd.NA
        out[f"{side}_instance"] = out[id_col].map(meta["instance"]) if "instance" in meta else pd.NA
        out[f"{side}_roi"] = out[id_col].map(meta["roi"]) if "roi" in meta else pd.NA
        out[f"{side}_major_roi"] = out[id_col].map(meta["major_roi"]) if "major_roi" in meta else pd.NA
        out[f"{side}_nt"] = out[id_col].map(meta["nt"]) if "nt" in meta else pd.NA
    return out


def summarize_types(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed cell types among typed body IDs in the processed graph."""

    in_w = edges.groupby("dst", sort=False)["weight"].sum().rename("incoming_weight")
    out_w = edges.groupby("src", sort=False)["weight"].sum().rename("outgoing_weight")
    degree = pd.concat([in_w, out_w], axis=1).fillna(0).reset_index().rename(columns={"index": "bodyId"})
    merged = nodes.merge(degree, on="bodyId", how="left").fillna({"incoming_weight": 0, "outgoing_weight": 0})
    typed = merged.dropna(subset=["type"]).copy()
    typed = typed[typed["type"].astype(str).str.len() > 0]
    if typed.empty:
        return pd.DataFrame(columns=["type", "n_nodes", "incoming_weight", "outgoing_weight"])
    return (
        typed.groupby("type", dropna=False, as_index=False)
        .agg(n_nodes=("bodyId", "nunique"), incoming_weight=("incoming_weight", "sum"), outgoing_weight=("outgoing_weight", "sum"))
        .sort_values(["incoming_weight", "n_nodes"], ascending=False)
        .reset_index(drop=True)
    )


def assign_node_major_roi_from_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Assign each body ID a dominant major ROI from parsed incident edge ROI."""

    if "edge_major_roi" not in edges.columns or edges.empty:
        return nodes
    src = edges[["src", "edge_major_roi", "weight"]].rename(columns={"src": "bodyId"})
    dst = edges[["dst", "edge_major_roi", "weight"]].rename(columns={"dst": "bodyId"})
    incident = pd.concat([src, dst], ignore_index=True)
    incident = incident[incident["edge_major_roi"].notna()]
    if incident.empty:
        return nodes
    scores = incident.groupby(["bodyId", "edge_major_roi"], as_index=False)["weight"].sum()
    scores = scores.sort_values(["bodyId", "weight"], ascending=[True, False]).drop_duplicates("bodyId")
    mapper = scores.set_index("bodyId")["edge_major_roi"]
    out = nodes.copy()
    out["major_roi"] = out["bodyId"].map(mapper).fillna(out["major_roi"])
    return out


def prepare_optic_lobe_tables(
    data_dir: str | Path,
    out_dir: str | Path,
    min_weight: float = 1.0,
    max_edges: int | None = None,
    sample_nodes: int | None = None,
    seed: int = 0,
    typed_core: bool = False,
) -> dict[str, Path]:
    """Load neuPrint tables and write canonical processed graph files."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges, _ = load_optic_lobe_edges(data_dir, min_weight=min_weight, max_edges=max_edges, sample_nodes=sample_nodes, seed=seed)
    node_ids = pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True))
    nodes = load_optic_lobe_nodes(data_dir, node_ids=node_ids)
    missing_ids = pd.Index(node_ids).difference(nodes["bodyId"])
    if len(missing_ids) > 0:
        missing = pd.DataFrame(
            {
                "bodyId": missing_ids,
                "type": pd.NA,
                "instance": pd.NA,
                "roi": pd.NA,
                "major_roi": pd.NA,
                "nt": pd.NA,
            }
        )
        nodes = pd.concat([nodes, missing], ignore_index=True)
    if typed_core:
        type_text = nodes["type"].astype("string")
        nodes = nodes[nodes["bodyId"].notna() & nodes["type"].notna() & type_text.str.len().fillna(0).gt(0)].copy()
        typed_ids = pd.Index(nodes["bodyId"].astype("int64").unique())
        before_edges = len(edges)
        edges = edges[edges["src"].isin(typed_ids) & edges["dst"].isin(typed_ids)].reset_index(drop=True)
        print(
            "Typed-core filter: "
            f"kept {len(nodes):,} typed body IDs and {len(edges):,} of {before_edges:,} edges"
        )
    if "edge_roi_info" in edges.columns:
        edges["edge_major_roi"] = assign_major_roi_series(edges["edge_roi_info"])
        edges = edges.drop(columns=["edge_roi_info"])
    else:
        edges["edge_major_roi"] = "other"
    nodes = assign_node_major_roi_from_edges(nodes, edges)
    edges = attach_metadata(edges, nodes)
    type_summary = summarize_types(nodes, edges)

    paths = {
        "edges": out_dir / "edges.parquet",
        "nodes": out_dir / "nodes.parquet",
        "type_summary": out_dir / "type_summary.csv",
    }
    edges.to_parquet(paths["edges"], index=False)
    nodes.to_parquet(paths["nodes"], index=False)
    type_summary.to_csv(paths["type_summary"], index=False)
    print(f"Wrote {len(edges):,} edges and {len(nodes):,} nodes to {out_dir}")
    return paths


def write_schema_report(data_dir: str | Path, out_path: str | Path) -> None:
    """Write a compact schema/head report for the relevant neuPrint tables."""

    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = [
        "Neuprint_Neuron_Connections.feather",
        "Neuprint_Neurons.feather",
        "Neuprint_Neuron_Connections.csv",
        "Neuprint_Meta.csv",
        "Neuprint_SynapseSet.csv",
        "Neuprint_SynapseSet_to_SynapseSet.csv",
    ]
    chunks: list[str] = []
    for name in files:
        path = data_dir / name
        chunks.append(f"\n### {name}\n")
        if not path.exists():
            chunks.append("MISSING\n")
            continue
        if path.suffix.lower() == ".feather":
            df = pd.read_feather(path)
        else:
            df = pd.read_csv(path, nrows=5)
        chunks.append(f"{df.shape}\n")
        chunks.append(f"{df.columns.tolist()}\n")
        chunks.append(f"{df.head().to_string()}\n")
    out_path.write_text("".join(chunks), encoding="utf-8")
