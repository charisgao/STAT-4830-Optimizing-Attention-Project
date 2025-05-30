# Self Critique

## **OBSERVE**

Reading through the report, we notice it has a strong theoretical foundation and we have conducted a good literature review of ways to improve the attention mechanism, but our project is lacking in terms of implementation and results. There are still improvements to be made in both the NSA and Performer and the completions to the prompts are also definitely not as coherent as the reference response, suggesting more exploration should be done with hyperparameter finetuning the NSA and Performer implementations.

## **ORIENT**

### Strengths

- Clear problem statement with well-defined goals for optimizing attention in transformer models
- Strong technical approach section with detailed mathematical formulation and validation methods
- Started to record memory usage and computation time

### Areas for Improvement

- Output text is less coherent than GPT-2 output text
- Limited context window for GPT-2
- Relatively high KL-divergence for NSA and Performer implementation
- More detailed comparison between output text between different implementations

### Critical Risks/Assumptions

We assume the approach of using Attention Mask Combination/NSA/Performer will yield meaningful improvements based on literature review. However, since we have a limited context window and max token size, we can't verify if this implementation achieves the stated goal of reducing the $O(n^2)$ bottleneck at scale.

## **DECIDE**

### Concrete Next Actions

- Create visualization comparing attention patterns between baseline and custom models to demonstrate what the model is actually learning
- Fix context length issues
- Empirically compare all the created models

## **ACT**

### Resource Needs

We received access to a GPU with sufficient memory to run experiments with longer sequences to train our implementations for more epochs and properly measure efficiency gains. We should also use PyTorch's profiler tools (torch.profiler.profile()) to isolate and measure the attention operations specifically, rather than just overall model performance.
