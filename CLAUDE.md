# Noisy RAG - Project Summary

## Overview
This Noisy-RAG project seeks to identify an optimal way to remove noises from excessive RAG retrieval. So that the model can dynamically generate by effectively "ignoring" the incorrect, irrelevant, and misleading retrieval.
The Noisy-RAG project adapts from the codebase from the SIGIR 2024 paper "The Power of Noise: Redefining Retrieval for RAG Systems". This study how noise and random documents affect RAG system performance.

## Project Structure

### Core Components
- **Retrieval**: Implements Contriever, ADORE, and BM25 retrievers
- **Generation**: Tests 4 LLMs (Llama-2-7b, mpt-7b, phi-2, falcon-7b) with different prompt structures
- **Evaluation**: Accuracy-based evaluation using exact answer matching

### Key Scripts
- `src/compute_corpus_embeddings.py` - Compute corpus embeddings for dense retrievers
- `src/index_embeddings.py` - Create FAISS indices from embeddings
- `src/compute_search_results.py` - Retrieve top-k documents
- `src/generate_answers_llm*.py` - Different generation configurations
- `src/read_generation_results.py` - Evaluation and accuracy computation

### Data
- Uses English Wikipedia (Dec 2018) corpus with NQ dataset
- 21M+ documents, filtered to <512 tokens
- Training: 10K sample, Test: 2.8K examples
- Available on HuggingFace: `florin-hf/wiki_dump2018_nq_open` and `florin-hf/nq_open_gold`

## Environment Setup
- **Python**: >=3.11 with UV package manager
- **GPU**: Required for FAISS-GPU and model inference
- **Memory**: ~25GB RAM for full corpus
- **Init**: Run `./init_uv.sh` to setup environment and Jupyter kernel

## Experiment Types
1. **Closed-Book QA**: Question only, no context
2. **Gold + Distracting**: Gold document with retrieved distractors
3. **Retrieved + NQ Random**: Retrieved docs + random NQ entries
4. **Retrieved + Other Random**: Retrieved docs + external random sources

## Dependencies
Key packages: transformers, datasets, faiss-gpu, vllm, accelerate, bitsandbytes

## Commands
- **Environment**: `uv sync` - sync dependencies
- **Testing**: Check README or scripts for test commands
- **Lint/Type**: Check for specific commands in project