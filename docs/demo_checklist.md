# Demo Checklist

Run this checklist before a project demonstration or delivery review.

- [ ] Activate the `journal-rag` Conda environment with Python 3.11.
- [ ] Confirm `.env` exists and `DEEPSEEK_API_KEY` is configured.
- [ ] Confirm `HF_HOME` points to a disk with enough free space.
- [ ] Run `python scripts/check_environment.py` successfully.
- [ ] Confirm `storage/journals.db` exists and the journal count is expected.
- [ ] Confirm the Chroma `journals` collection is built and matches SQLite count.
- [ ] Confirm `data/papers/test_paper.pdf` exists and has a text layer.
- [ ] Confirm BGE-M3 is available in the configured Hugging Face cache.
- [ ] Confirm BGE-Reranker-v2-m3 is available when reranking will be demonstrated.
- [ ] Run `python scripts/smoke_test.py --skip-llm` successfully.
- [ ] Run `pytest` successfully for the default fast unit suite.
- [ ] Run `python main.py data/papers/test_paper.pdf --skip-recommendation`.
- [ ] If network/API access is available, run `python main.py data/papers/test_paper.pdf`.
- [ ] Run `python scripts/test_agent_tool_calling.py` to verify an actual Tool call.
- [ ] Start `python scripts/run_agent.py`; verify `help` explains usage/memory and `exit`/`quit` ends the session.

Never display `.env`, API keys, Authorization headers, or debug output containing secrets during a demo.
