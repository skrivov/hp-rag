from src.server.instrumentation import EventFactory


def test_step_tracker_emits_matching_ids():
    factory = EventFactory(run_id="run-test")
    tracker = factory.step(phase="retrieval", name="rag.search", summary="Vector search")

    start_event = tracker.start_event()
    end_event = tracker.end_event()
    progress_event = tracker.progress_event(summary="retrieving")

    assert start_event["step_id"] == end_event["step_id"] == progress_event["step_id"]
    assert start_event["phase"] == "retrieval"
    assert end_event["metrics"]["latency_ms"] >= 0
