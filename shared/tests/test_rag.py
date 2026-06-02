from shared.rag import chunk_text_records


def test_chunk_text_records_preserves_metadata_and_overlap():
    words = [f"w{i}" for i in range(12)]

    chunks = chunk_text_records(
        [(" ".join(words), 3)],
        source="history.pdf",
        chunk_words=5,
        overlap_words=2,
    )

    assert len(chunks) == 4
    assert chunks[0].source == "history.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_id == 0
    assert chunks[1].text.split()[:2] == ["w3", "w4"]
