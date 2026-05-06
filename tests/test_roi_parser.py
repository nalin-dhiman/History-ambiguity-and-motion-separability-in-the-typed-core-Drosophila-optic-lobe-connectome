from thermodynamic_connectomics.io.roi import major_roi_from_info


def test_major_roi_prefers_specific_over_ol():
    info = '{"LO(R)":{"post":3},"ME(R)":{"post":5},"OL(R)":{"post":20}}'
    assert major_roi_from_info(info) == "ME(R)"


def test_major_roi_uses_ol_when_no_specific_roi():
    info = '{"OL(R)":{"post":2}}'
    assert major_roi_from_info(info) == "OL(R)"


def test_major_roi_handles_bad_json():
    assert major_roi_from_info("not-json") == "other"
