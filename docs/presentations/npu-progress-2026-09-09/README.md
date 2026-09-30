# NPU translation progress slides

- [npu-progress.pptx](npu-progress.pptx): eight editable slides, including a native chart with an embedded data workbook and speaker notes.
- [npu-progress.pdf](npu-progress.pdf): eight-page export from Microsoft PowerPoint.
- [preview.png](preview.png): overview of all eight slides.
- [speaker-notes.md](speaker-notes.md): presentation narrative, metric definitions, limitations and sources.
- [evidence.json](evidence.json): per-case historical results, current validation aggregates, source paths and SHA-256 hashes.
- [build_deck.py](build_deck.py): reproducible deck generator using the evidence snapshot.

The historical experiment completed 14/20 one-shot tasks and 19/20 agentic tasks, using different search budgets. AMD completion meant compilation; Intel also had historical CPU equivalence checks. The newer 22/22 result comes from deterministic evaluator validation with zero LLM calls. Neither result establishes exact NPU binary correctness, NPU latency or power consumption.

Verification performed: all eight source hashes matched the live reports; native chart values matched the recorded AMD/Intel results; PowerPoint rendered eight slides with zero text-height overflow findings; the PDF has eight pages and contains the key reported figures. The slide overview and corrected layouts were visually inspected. No new translation or hardware experiment was run for this presentation.

To rebuild the PowerPoint and notes from this directory:

```sh
uv run --with python-pptx==1.0.2 python build_deck.py
```

Regenerate the PDF and preview after any slide edits using PowerPoint export; the Python generator does not update these rendered files.
