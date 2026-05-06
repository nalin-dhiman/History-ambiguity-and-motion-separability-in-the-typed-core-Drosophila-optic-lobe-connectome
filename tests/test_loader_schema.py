import pandas as pd

from thermodynamic_connectomics.io.optic_lobe_loader import infer_column_mapping


def test_neuprint_schema_mapping():
    cols = pd.Index([":START_ID(Body-ID)", ":END_ID(Body-ID)", "weightHR:int", "weight:int"])
    mapping = infer_column_mapping(cols)
    assert mapping.source == ":START_ID(Body-ID)"
    assert mapping.target == ":END_ID(Body-ID)"
    assert mapping.weight == "weight:int"


def test_common_schema_mapping():
    mapping = infer_column_mapping(["bodyId_pre", "bodyId_post", "synapse_count"])
    assert mapping.source == "bodyId_pre"
    assert mapping.target == "bodyId_post"
    assert mapping.weight == "synapse_count"
