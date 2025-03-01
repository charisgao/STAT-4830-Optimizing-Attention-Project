# Self Critique

## **OBSERVE**

Reading through my report, I notice it has a strong theoretical foundation but there are some arbitrary hyperparameter decisions such as using 5 candidate masks. It is not clear how the candidate masks are chosen. Also, more metrics are needed to confirm that this solution outperforms the previous linear combination. The completions to the prompts are also definitely not as coherent as the reference response, suggesting more exploration should be done with what candidate masks we use.

## **ORIENT**

### Strengths

- Clear problem statement with well-defined goals for optimizing attention in transformer models
- Strong technical approach section with detailed mathematical formulation and validation methods
- Thoughtful literature review connecting our work to recent relevant papers (Lexico and NSA)

### Areas for Improvement

- No actual memory or computational efficiency measurements
- Lack of comparison between baseline and custom model

### Critical Risks/Assumptions

I'm assuming the approach of using linear combinations of position-specific attention masks will yield meaningful improvements. However, without actual measurements of memory usage or computation time, I can't verify if this implementation achieves the stated goal of reducing the $O(n^2)$ bottleneck.

## **DECIDE**

### Concrete Next Actions

- Implement and execute memory/computation benchmarks comparing baseline and custom attention (using torch.cuda.max_memory_allocated() as described in the approach)
- Create visualization comparing attention patterns between baseline and custom models to demonstrate what the model is actually learning

## **ACT**

### Resource Needs

I need access to a GPU with sufficient memory to run experiments with longer sequences (at least 512 tokens as stated in constraints) to properly measure efficiency gains. I should use PyTorch's profiler tools (torch.profiler.profile()) to isolate and measure the attention operations specifically, rather than just overall model performance.
