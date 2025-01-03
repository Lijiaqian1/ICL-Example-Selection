# MTop Semantic Parsing with ICL and Contrastive Learning

This repository explores the use of contrastive learning to improve in-context learning (ICL) example selection for semantic parsing tasks, with a focus on the **MTop dataset**. The ultimate goal is to enhance ICL performance by leveraging latent representation similarities within large language models (LLMs) like LLaMA2-7B.

## Project Overview

### Problem Statement
Traditional ICL approaches often rely on embedding similarity for selecting examples. However, this method struggles with complex tasks like semantic parsing, where the structure and semantics of the parses play a critical role. This project aims to address these challenges by:

1. Developing methods to compute fine-grained parse similarity scores for MTop parsing trees.
2. Using these similarity scores to train contrastive probes on the hidden representations of LLMs.
3. Utilizing the trained probes to select optimal ICL exemplars dynamically.

### Key Contributions
- A hybrid **Faiss + ZSS** approach to identify and refine parse similarity between examples.
- A framework to leverage **contrastive learning** for improving ICL exemplar selection.
- Demonstrated the potential of **LLaMA2-7B** for semantic parsing with ICL.

## Current Progress

### Dataset
- **MTop**: Multi-domain parsing dataset, providing sentences and their semantic parses in a bracketed structure. 
- The project preprocesses the dataset to facilitate contrastive learning by creating positive and negative example pairs.

### Similarity Scoring
- **Faiss Embedding Filtering**: Embedding-based retrieval to find top-K semantically similar parses.
- **Parse-Level Refinement**: A custom similarity scoring method based on the **ZSS tree edit distance**, with enhancements for handling unordered children and embedding-based leaf comparisons.

### Model
- **LLaMA2-7B**: Utilized for ICL experiments.
- Sentence embeddings are generated using `SentenceTransformer` models like `all-MiniLM-L6-v2`.

### Contrastive Learning
- Example pairs (positive and negative) are gathered using the MTop dataset and custom similarity scoring methods.
- Future work includes training probes on LLaMA2-7B hidden states to facilitate example selection during ICL.

## Code Structure

