# Optimizing Attention Mechanisms in Transformer Models

## Team Members

- Chandler Cheung
- Charis Gao
- Jordan Hochman

## High-Level Summary

This project focuses on optimizing attention mechanisms in Transformer models to overcome the inherent O(n²) time and memory bottleneck. We explore three different approaches to optimize attention:

1. Linear Combination of Attention Masks
2. Performer
3. Native Sparse Attention (NSA)

Our key findings show that the simplest approach (linear combination of attention masks) performed best in terms of output coherence and similarity to the baseline GPT-2 model. The loss function we chose to minimize is the KL divergence, which measures the difference between two probability distributions of the next token for prediction. While we achieved low KL-divergence between our custom models and the baseline, we discovered that statistical similarity doesn't necessarily translate to human-perceived quality in generated text. Most of the text from the Performer and NSA model were not coherent. We hypothesize that this is due to the mismatch in the datasets used. GPT-2 was trained on a dataset excluding Wikipedia, while our models were trained on only Wikipedia data. This mismatch between the training data might be the cause for poor model performance.

We also tried calculating the loss based on the cross-entropy between the tested model and the actually correct output token. We added this loss term for next token prediction, but found that it did not improve output coherence either in any model.

A detailed report of the methodologies and all the findings can be found in [`docs/report.md`](docs/report.md).

#### Key Limitations

- Limited to WikiText-2 dataset: GPT2 was trained on Webtext (excludes Wikipedia) while we trained on Wikitext-2 (solely Wikipedia)
- Training constrained by compute resources of Google Colab
- Context window size limitations to reduce training speed/time
- Overall output coherence issues

#### Future Work

- Experiment with datasets similar to GPT-2's training data
- Further finetune hyperparameters
- Test with larger context windows
- Explore additional loss functions
- Evaluate with other baseline models

## Repository Structure

```
.
├── docs/           # Documentation and report files
│   ├── report.md   # Final project report
│   ├── Final_Presentation_Slides.pdf    # Final slides
│   └── figures/    # Project figures and visualizations
├── notebooks/      # Jupyter notebooks implementations
└── _archive/       # Development history
```

Development history: previous versions and drafts of report and notebooks of attention optimization implementations can be found in [`_archive`](_archive); older versions are labeled with the week when they were created.

## Setup Instructions

Most of the implementations are done as Google Colab notebooks, and can be directly run in Google Colab.

1. Python Environment:

   - Python 3.8+ recommended
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Unix/macOS
     # or
     .\venv\Scripts\activate   # On Windows
     ```

2. GPU Requirements:
   - CUDA-compatible GPU recommended
   - Google Colab T4 GPU sufficient for basic experiments

## Executable Demo

Each attention optimization implementation are self-contained in a Google Colab notebook. The training loop and inference for generating text are located in each appropriate notebook. When training, model parameters are regularly checkpointed to save progress.

- [Linear Combination Custom Masks Implementation Notebook](notebooks/Custom_masks_implementation.ipynb)
- [Performer Implementation Notebook](notebooks/Performer_implementation.ipynb)
- [NSA Implementation Notebook](notebooks/NSA_implementation.ipynb)
