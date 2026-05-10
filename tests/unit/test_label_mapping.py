from src.data.label_mapping import map_iemocap_to_meld, to_label_id


def test_iemocap_mapping_decisions() -> None:
    assert map_iemocap_to_meld("excited") == "joy"
    assert map_iemocap_to_meld("frustrated") == "anger"


def test_label_id_mapping() -> None:
    assert to_label_id("neutral") == 0
    assert to_label_id("anger") == 6
