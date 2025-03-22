# Optimizing Attention Mechanisms in Transformer Models

## Problem Statement

### What Are We Optimizing?

We seek to overcome the inherent $O(n^2)$ time and memory bottleneck in Transformer attention by **learning which tokens** in the sequence to focus on. Instead of attending to all previous tokens, we aim to develop a **customizable attention mask** that pinpoints only the most relevant parts of the input. We plan to train this customized attention to produce outputs similar to a baseline (unmodified) Transformer. By minimizing the difference (KL-divergence) between the baseline and our custom model, we aim to preserve model quality while reducing computational cost by minimizing the number of tokens required in the attention mechanism.

### Why Does This Problem Matter?

As more research has been done with large language models (LLMs), one common result is increasing the size of the model. In recent years, the size of models have grown exponentially, and models cannot fit in single GPU memory. Thus, one goal now is to use fewer parameters and find ways to represent large models more compactly. Existing research has been done to build more efficient LLMs, such as the Lottery Ticket hypothesis to make smaller networks (find important parts of the network, throw away the rest) and distillation. At the same time, another issue lies with attention.

Transformer-based language models have become central to a wide variety of NLP tasks, but they quickly become impractical for very long sequences due to quadratic complexity. Improving their attention efficiency can:

- **Enable Longer Contexts**: Handle documents or tasks requiring thousands of tokens.
- **Reduce Hardware Costs**: Lower memory usage means more feasible deployment.
- **Maintain Accuracy**: Achieve similar or near-equivalent performance as full attention.

### How Will We Measure Success?

1. **Accuracy Retention**: Does the custom attention model perform comparably to the baseline on text tasks (e.g., perplexity, F1, or other relevant metrics)?
2. **Computational Improvement**: We will track how well the approach scales with sequence length, aiming for reduced memory usage or speed gains.
3. **Distribution Alignment**: A lower KL-divergence between the custom model's outputs and the baseline model signals successful attention optimization.

### What Are Our Constraints?

We are currently focusing on **WikiText-2** as our primary dataset for language modeling. It is freely available, moderate in size (roughly 2 million tokens), and standard for benchmarking. We want to be able to:

- Process sequences of up to 512 tokens (as a starting point) on a single GPU without out-of-memory errors.
- Implement the code in standard PyTorch, avoiding highly specialized CUDA kernels.
- Retain acceptable generation quality while freezing most of the baseline model parameters.
- Compatibility with standard PyTorch APIs (e.g., `nn.MultiheadAttention`).

### What Data Do We Need?

- **WikiText-2** for initial experimentation and evaluation.
- Potentially **WikiText-103** or other larger corpora as we scale up the approach and test longer context windows.
- For thorough testing, we may also include smaller validation sets to measure perplexity and check overfitting.

### What Could Go Wrong?

- **Underfitting**: If the custom mask prunes too aggressively, performance or fidelity may drop significantly.
- **Overhead vs. Benefit**: A clever mask may still impose overhead that negates memory/computational gains if it's not efficiently implemented.
- **Instability**: With a learnable attention mask, training might become unstable or sensitive to hyperparameters.

---

## Technical Approach

### Mathematical Formulation

We define two Transformer models: a **baseline** and a **custom**. If $P_{\text{base}}(\cdot\mid X)$ is the baseline output distribution and $P_{\text{custom}}(\cdot\mid X)$ is our custom model's distribution, we minimize:

$$\mathcal{L} = \mathrm{KL}\bigl(P_{\text{base}} \,\|\, P_{\text{custom}}\bigr)$$

summed over all training examples $X$. This objective encourages the custom attention to preserve the baseline model's behavior.

### Algorithm/Approach Choice and Justification

- **Adaptive Attention Mask**: Long-term, we want to learn a sparse mask or restricted set of tokens that provide sufficient context with fewer computations.
- **KL-Divergence Alignment**: By aligning probabilities, we ensure that any modifications to attention remain faithful to the baseline's predictions.
- **Sub-Quadratic Focus**: The ultimate aim is to reduce attention complexity from $O(n^2)$ to something more tractable for large $n$.

- **Native Sparse Attention**: NSA is a hardware-optimized and end-to-end trainable sparse attention mechanism that reduces computational overhead through a hierarchical approach. It organizes tokens into compressed representations for global context, selectively retains the most relevant tokens for local precision, and employs a sliding window mechanism to maintain continuity in processing.

![Native Sparse Attention Diagram](./figures/NSA_structure.png)

- **Performers and Kerformers**: These use kernel-based approximations to replace softmax attention, reducing complexity to linear time. Performers rely on random feature maps, while Kerformers improve this with structured, data-aware projections that are learned—making both efficient for long-sequence modeling without major accuracy loss.

### PyTorch Implementation Strategy

1. **Baseline Model**: A larger, established Transformer architecture loaded from a standard library (e.g., Hugging Face).
2. **Custom Attention Module**: Replace the default attention with a mechanism that only processes a subset of tokens, with learnable parameters dictating which tokens matter most. We also experiment using techniques from NSA or the Performer in the custom attention module.

- Native Sparse Attention: implement hierarchical attention with compressed tokens, selective attention, and sliding window with a GPT2 model
- Performer: implement a Performer using the base GPT2 model, replacing the attention layer with a kernel-based linear attention module (e.g., FAVOR+, and potentially later causal FAVOR+).

3. **Loss Computation**: Compute logits from both models on the same input batch, then apply KL-divergence.
4. **Parameter Updates**: Use standard optimization (e.g., AdamW) to train the new attention module while freezing or partially freezing other layers.

### Measure of Success

1. Low KL divergence
2. Memory or space efficiency
3. Natural Language Flow\*\*
   - Evaluate the coherence and fluency of the generated text by comparing it to the baseline model's outputs. This will involve:
     - **Human Evaluation**: Conducting qualitative assessments where human judges rate the generated text on criteria such as grammatical correctness, logical flow, and overall readability.
     - **Diversity of Outputs**: Analyzing the variety in generated responses to the same prompts to ensure that the model does not produce repetitive or overly similar outputs, which can indicate a lack of creativity or flexibility in language generation.

### Validation Methods

- **Validation Loss**: Track KL-divergence on a held-out set to ensure the custom model matches the baseline distribution over time.
  - Load weights from baseline models (e.g., HuggingFace's `bert-base-uncased`), replace _only_ the attention module, and evaluate **without fine-tuning**.
  - Ensures optimization does not rely on retraining to "recover" lost accuracy.
  - L1 penalty for coefficients of attention masks
- **Perplexity/Accuracy**: Evaluate on standard tasks (e.g., language modeling or classification) to ensure minimal drop in performance.
- **Edge cases**: Sequences with extreme sparsity (e.g., all padding tokens) or high similarity (e.g., repeated tokens).
- **Scalability Tests**: Gradually increase input sequence lengths and measure memory usage, throughput, and any speed improvements.

### Resource Requirements and Constraints

- We plan to use a GPU for training and validation on WikiText-2. We will be utilizing Google Cloud for access to more powerful GPUs and TPUs, utilizing the provided credit.

## Results

### Native Sparse Attention

In our implementation of Native Sparse Attention, we manually created a hierarchical attention layer that incorporates compressed tokens, selective attention, and a sliding window mechanism using PyTorch. This approach aimed to optimize the attention computation by reducing the number of tokens processed while maintaining the model's performance. We trained the model for 10 epochs using 50% of the WikiText-2 dataset as our training data. However, the initial results were disappointing, as the generated outputs consisted of random symbols rather than coherent text.

To address these issues, we adjusted the implementation by increasing the context size and modifying the parameters for both the selective attention and sliding window mechanisms. Despite these changes, the new results still yielded outputs that entirely comprised exclamation points, indicating that the model was not effectively learning meaningful patterns in the data.

We could not find an official implementation of NSA by DeepSeek researchers. In our search for improvement, we found a library implementation of Native Sparse Attention by Philip Wang at Observe.AI and attempted to integrate it into our pipeline. However, we encountered challenges related to the return type and value of the forward method within the attention class. This issue rendered the model unable to generate coherent text, although it successfully completed the training loop. It remains unclear whether the model has learned anything meaningful or if the outputs are simply gibberish. Notably, the KL divergence loss has been decreasing with each epoch, suggesting some level of learning is occurring, but the quality of the generated text remains a significant concern.

### Performer

TODO
We ran into issues implementing the Performer as a replacement for the attention in GPT2, and are still working on fixing the issues. However, with one implementation that seems to work, we do get intelligible output, albeit not the best. The loss does decrease to zero, but we do want to measure the speed and accuracy better in the future.

In the future, we intend to extend these approaches to **measure the computational and memory usage** of our custom attention implementation, as well as experiment with regularization, penalty, and/or constraints (eg. low rank using SVD) to reduce complexity. We also want to test with both more advanced models beyond GPT-2, and additionally smaller models than can be run locally.

**Key Observations**

- KL-divergence decreases steadily, confirming that the custom model is aligning its output distribution to GPT-2's.
- However, many of the outputs for both Native Sparse Attention and the Performer are not coherent, with outputs that are gibberish.
- We did not measure or improve memory usage—the code as written does not yet aim for sub-quadratic complexity or large-scale efficiency gains.

### Evidence your implementation works

- **Successful Training Loop:** Over 100 epochs, the KL-divergence–based loss with the custom NSA attention layer decreased from about TODO down to TODO on our dataset, and the custom Performer-based attention layer decreased from TODO down to TODO on our dataset, indicating the custom attention can begin mimic the reference model's distributions.
- **Text Generation:** We tested with a few prompts, observing that our custom model produced text. The text is not as coherent as the reference model though.

### Test case results

Below are selected generation samples using the same prompts for both the reference and custom models. The custom model's outputs are not very coherent, but for Performer they still produce recognizable English words. This shows the model is capturing some of GPT-2's distribution, though lots of improvements can still be made.

#### Native Sparse Attention

Prompt: Artificial intelligence
Reference: [Artificial intelligence] is a new field of research that has been in the works for a while now.
Custom: [Artificial intelligence]""%&&&(&&,&&&0&0,0&,000&00&00000000000000000&0000,0&000,0,0&[0000000&000[00[0&0,0[0[0000&[00[,[0[000

Reference: Artificial intelligence is a new field of research that has been in the works for a while now.
Custom: Artificial intelligence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#### Performer

Prompt: In the year 2157, humanity had finally perfected interstellar travel. The first colony ship, brimming with hope and thousands of eager settlers
Reference: In the year 2157, humanity had finally perfected interstellar travel. The first colony ship, brimming with hope and thousands of eager settlers—the Arkion crew made it to hyperspace in January 3160 – was launched from this world's barren wasteland
Custom: In the year 2157, humanity had finally perfected interstellar travel. The first colony ship, brimming with hope and thousands of eager settlers When we are a few 1). We have only then to help: Rascal\'1-s at school ------------------------------------------------------------------------

### Current Limitations

- **Minimal Dataset**: Synthetic or small text corpora, offering limited insight into real-world performance (we only use 1000 training samples).
- **Limited Mask Optimization**: We used a simple weighted linear combination of attention masks that attend to tokens in specific positions.
- **No Large Model**: GPT-2 was used purely for demonstration; we have not tested on bigger or more modern architectures.

### Resource Usage Measurements

- On one T4 GPU on Google Colab, this took a while to run for 100 epochs (about 30 mins).
- These resource measurements are modest because our demonstration used a restricted sequence length and a small amount of data.

### Unexpected challenges

- **Limited Coherence**: We will need more sophisticated masking to handle longer contexts properly.
- **Overfitting**: Because our dataset was tiny, we saw the model quickly saturate or jump around in text quality, suggesting the need for better regularization.
- **Accuracy Concerns**: Potential accuracy degradation from over-sparsification.

## Next Steps

### Immediate improvements needed

- **Extend Training Data:** Use the full WikiText-2 dataset (rather than just 1000 samples) to get more realistic coverage and reduce overfitting.
- **Fine-Tune Hyperparameters:** Adjust learning rates, batch sizes, and sequence lengths to improve stability and convergence.
- **Efficiency and Memory Improvement:** Track speed and memory usage of attention masks.
- **Attention Restructuring for NSA**: adjust implementation so that words are produced instead of repetitive symbols
- **Fix Performer Implementation**: Fully flush out Performer implementation
- **Add Kerformer Implementation**: Add Kerformer implementation and compare with previous methods.

### What You've Learned So Far

- **KL-Divergence Feasibility**: Minimizing KL directly is an effective way to align two models' distributions.
- **Attention Substitution**: Swapping out the standard self-attention module is straightforward if we mirror the input-output shapes and track weights carefully.
