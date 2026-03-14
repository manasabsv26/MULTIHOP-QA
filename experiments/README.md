# Canonical Experiment Structure

Use grouped entry points under baselines and solution.

## Baselines

- baselines/llama_llm.py
  - Notebook-based LLaMA baseline over gold evidence_list context.
  - Canonical notebook path: LLama-code/LLM-llama32.ipynb

- baselines/plain_rag.py
  - Plain RAG baseline entry.
  - Pipeline stage is notebook-based: RAG/rag.ipynb
  - Script stages available: retrieval-eval and qa-eval

- baselines/plain_coa.py
  - Plain Chain of Agents baseline over gold evidence_list context.
  - Delegates to CoA/chainofagents.py

## Solution

- solution/rag_coa.py
  - Main proposed solution: single-retrieval RAG + CoA.
  - Delegates to RAG_and_CoA.py

- solution/rag_coa_reretrieval.py
  - Solution variant: RAG + CoA with periodic re-retrieval.
  - Delegates to RAG_and_CoA-experiment1.py

## Compatibility Aliases

These old canonical names are kept for compatibility and forward to the grouped structure:

- coa_gold_context.py -> baselines/plain_coa.py
- rag_coa_single_retrieval.py -> solution/rag_coa.py
- rag_coa_reretrieval.py -> solution/rag_coa_reretrieval.py

## Notes

- Help output for wrappers is lightweight and does not import heavy runtime dependencies.
- Original implementations remain unchanged and still runnable directly.