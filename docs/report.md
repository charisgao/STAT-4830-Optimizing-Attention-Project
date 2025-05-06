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
  - Built from Wikipedia articles and is common dataset among literature that is curated and easy to use. THere are abotu two million tokens in the training set, about 220,000 tokens in the validation set, and about 240,000 tokens in the test set.
  - Despite GPT2 being trained on a different dataset, the original dataset is not publically available (OpenAI scrapped web data and excluded Wikipedia pages)
- Potentially **WikiText-103** or other larger corpora as we scale up the approach and test longer context windows.
- For thorough testing, we may also include smaller validation sets to measure perplexity and check overfitting.

### What Could Go Wrong?

- **Underfitting**: If the custom mask prunes too aggressively, performance or fidelity may drop significantly.
- **Overhead vs. Benefit**: A clever mask may still impose overhead that negates memory/computational gains if it's not efficiently implemented.
- **Instability**: With a learnable attention mask, training might become unstable or sensitive to hyperparameters.

---

## Literature Review

- **Sparse Attention**: Each token attending to a subset of other tokens
  - BASED ([Arora et al., 2024](https://arxiv.org/abs/2402.18668)): combines linear attention + sliding window attention
  - Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention ([Yuan et al., 2025](https://arxiv.org/abs/2502.11089)): algorithmic innovation and hardware-aligned optimization for long-context optimization
- **Low-Rank Approximations**: approximate attention matrix with low-rank matrices
  - Linformer ([Wang et al., 2020](https://arxiv.org/abs/2006.04768)): projects $n \times d_m$ query/key matrices to smaller $k\times d_k$ where $k<<n$, which is a low-rank factorization of original attention
- **Efﬁcient Routing/Dynamic Attention**: dynamically determine which tokens should attend to each other
  - Routing Transformer ([Roy et al., 2021](https://arxiv.org/abs/2003.05997)): tokens mapped to routing space, uses online k-means clustering to assign tokens to clusters based on similarity in routing space, tokens attend only to tokens within same cluster
- **Kernel-based Methods**: reformulate attention with kernel functions
  - Kernel trick: performs this operation in the original space (implicitly compute dot products in some-dimensional feature space, without ever having to transform the vectors to that space)
  - Performer ([Choromanski et al., 2021](https://arxiv.org/abs/2009.14794)) – replaces softmax attention, uses random feature mappings to approximate the exponential kernel

---

## Technical Approach

### Mathematical Formulation

We define two Transformer models: a **baseline** and a **custom**. If $P_{\text{base}}(\cdot\mid X)$ is the baseline output distribution and $P_{\text{custom}}(\cdot\mid X)$ is our custom model's distribution, we minimize:

$$\mathcal{L} = \mathrm{KL}\bigl(P_{\text{custom}} \,\|\, P_{\text{base}}\bigr)$$

summed over all training examples $X$. This objective encourages the custom attention to preserve the baseline model's behavior by keeping the probability distribution for the next token similar.

### Algorithm/Approach Choice and Justification

- **Adaptive Attention Mask**: Long-term, we want to learn a sparse mask or restricted set of tokens that provide sufficient context with fewer computations.
- **KL-Divergence Alignment**: By aligning probabilities, we ensure that any modifications to attention remain faithful to the baseline's predictions.
- **Sub-Quadratic Focus**: The ultimate aim is to reduce attention complexity from $O(n^2)$ to something more tractable for large $n$.
- **Native Sparse Attention**: NSA is a hardware-optimized and end-to-end trainable sparse attention mechanism that reduces computational overhead through a hierarchical approach. It organizes tokens into compressed representations for global context, selectively retains the most relevant tokens for local precision, and employs a sliding window mechanism to maintain continuity in processing.

![Native Sparse Attention Diagram](./figures/NSA_structure.png)

- **Performers**: This uses a kernel-based approximations relying on random feature maps to replace softmax attention, reducing complexity to linear time.

### PyTorch Implementation Strategy

1. **Baseline Model**: A larger, established Transformer architecture loaded from a standard library (e.g., Hugging Face) (GPT2).
2. **Custom Attention Module**: Replace the default attention with a mechanism that uses a weighted combination of attention masks and learnable parameters dictating which tokens matter most. We also experiment using techniques from NSA or the Performer in the custom attention module.

- Native Sparse Attention: implement hierarchical attention with compressed tokens, selective attention, and sliding window with a GPT2 model
- Performer: implement a Performer using the base GPT2 model, replacing the attention layer with a kernel-based linear attention module (e.g., FAVOR+, and potentially later causal FAVOR+).

3. **Loss Computation**: Compute logits from both models on the same input batch, then apply KL-divergence.
4. **Parameter Updates**: Use standard optimization (e.g., AdamW) to train the new attention module while freezing or partially freezing other layers.

### Measure of Success

1. Low KL divergence
2. Memory or space efficiency
3. Natural Language Flow
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

---

## Results

### Linear Combination of Attention Masks

In this implementation, we used 5 different attention masks, each attending to just the last i'th token (i.e. one for the last token, one for the second to last token, and so on). We did see the loss decrease over time for both the training and validation datasets. It went from 1.4750 to 0.5875 over 10 epochs:

![Training Loss](./figures/week13_report_training_loss.png)
![Training Loss](./figures/week13_report_validation_loss.png)

Below are graphs of the values of the coefficients of the attention masks for the linear combination of the masks:

![Attention Block 0](./figures/week13_report_attention_block0.png)
![Attention Block 4](./figures/week13_report_attention_block4.png)
![Attention Block 8](./figures/week13_report_attention_block8.png)
![Attention Block 11](./figures/week13_report_attention_block11.png)

We also tracked the training and inference wall clock times, CPU clock times, CPU memory, and GPU memory usage. You can find more images of these in the [`/figures`](./figures) folder, but here is the CPU time for inference. It plots the output token length versus the time it took to run:

![Attention Block 8](./figures/week13_report_custom_inference_cpu_time.png)

As you can see, this appears to be linear, which is what we expect with this linear combination of attention masks approach (since the masks together only attend 5 tokens).

### Native Sparse Attention

In our implementation of Native Sparse Attention, we initially tried manually created a hierarchical attention layer that incorporates compressed tokens, selective attention, and a sliding window mechanism using PyTorch. This approach aimed to optimize the attention computation by reducing the number of tokens processed while maintaining the model's performance. We initially trained the model for 10 epochs using 50% of the WikiText-2 dataset as our training data. However, the initial results were disappointing, as the generated outputs consisted of random symbols rather than coherent text.

To address these issues, we adjusted the implementation by increasing the context size and modifying the parameters for both the selective attention and sliding window mechanisms. Despite these changes, the new results still yielded outputs that entirely comprised exclamation points, indicating that the model was not effectively learning meaningful patterns in the data.

We could not find an official implementation of NSA by DeepSeek researchers. In our search for improvement, we found a library implementation of Native Sparse Attention by Philip Wang et al. at Observe.AI and attempted to integrate it into our pipeline. He developed the [native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) open-sourced library of sparse attention pattern. They implemented CUDA kernel hacking, single transformer-based compression network, and included compression block hyperparameters.

We fixed our previous errors in tensor misalignment and generating output next. We ran our optimization algorithm for 5 epochs, given our current compute restraint. Notably, the KL divergence loss has been decreasing with each epoch, suggesting some level of learning is occurring. Initially, the loss was 331.665, but decreased to 175.68 after 8 epochs.

After getting a working implementation, we tested out different choices for hyperparameters such as the learning rate, temperature, epochs, and choice for optimizer. Initially, we ran only 5 epochs using the AdamW optimizer with initial learning rate of `5e-5` and temperature of 1. This did not yield ideal results, so we decided to run more epochs and change the hyperparameters. We changed the initial learning rate to be `1e-3` and a temperature of 0.7. With 0 epochs, the initial training loss decreased from 715.39 to 88.045. Even though this training loss is still relatively high, the output of the generated text is somewhat coherent. We changed up the calculation for the KL divergence from using a 'mean' reduction to 'sum' reduction when calculating the KL divergence across all the training samples in the batch. When running this optimization for 50 epochs, the training loss decreased from 713.74 to 33.46; the loss didn't change for the last few epochs, so we stopped the training loop.

Next, to isolate the contributing factor of whether the dataset was bad or the next token predicted was bad, we included an additional term in the loss function to measure the loss for the next predicted token. In this sense, if we still achieve a relatively low loss, then we know the main contributing factor for poor performance would be the dataset, since we used a dataset that's different than the original data that GPT2 was trained on. Running this attention optimization with the modified loss function, the loss decreased from 1181.643 to 83.756. From the graph below, we see that the loss seems to be bottoming out, so we didn't continue training more than 50 epochs.

![NSA Average KL + Next Predicted Token Loss per Epoch](./figures/nsa_avg_kl_next_token_loss.png)

We tracked the time for training with the NSA implementation and CPU/GPU usage. Below are the plots for the CPU/GPU and memory usage during training. These plots don't provide much insight into the time and space efficiency compared to normal, full attention.

![NSA (Loss: KL + Next Predicted Token) Wall TIme](./figures/nsa_wall_time.png)
![NSA (Loss: KL + Next Predicted Token) CPU Time](./figures/nsa_cpu_time.png)
![NSA (Loss: KL + Next Predicted Token) CPU Memory Usage](./figures/nsa_cpu_mem.png)
![NSA (Loss: KL + Next Predicted Token) GPU Memory Allocation](./figures/nsa_gpu_alloc.png)

### Performer

In the Performer implementation, we replace the original attention layer in GPT2 with a Performer attention that uses FAVOR+ to map Q and K to a different space using random projections. We faced issues with NaN and infinity values previously, possibly because of overflow/underflow, as well as with division by 0. We were able to resolve many of these by normalizing the data `x_norm = x / math.sqrt(self.head_dim)` or the query/key projections. With these changes, we saw the KL divergence decrease from 2.3446 to 2.3255 over 5 epochs. However, the new results still yielded outputs that were not very coherent, though the performance was significantly improved over past results.

**Key Observations**

- KL-divergence decreases steadily, confirming that the custom model is aligning its output distribution to GPT-2's.
- However, many of the outputs for both Native Sparse Attention and the Performer are not coherent or clearly lacking compared to GPT2.
- Achieved low KL-divergence between custom and baseline models
  - Probability distributions should be mostly aligned, but this statistical similarity doesn’t translate to human-perceived quality
  - KL-divergence alone might be an insufﬁcient metric for capturing language quality
and coherence. There is a gap between statistical and semantic performance (aligning token distribution patterns is not enough)

### Test Case Results

Below are selected generation samples using the same prompts for both the reference and custom models. With more epochs, we see that the custom model's outputs become more coherent, but they eventually divulge into gibberish. This is most likely due to limited context length.

The models still produce recognizable English words. This shows the model is capturing some of GPT-2's distribution, though lots of improvements can still be made.

#### Linear Combination of Attention Masks

**Prompt**: Hello, my name is

- **Reference**: Hello, my name is Michael. I am an avid and highly informed computer science student who has been teaching at the University of Maryland for over 20 years." The letter said that students should be able to "discuss any topic or situation related by their professor about which they have no knowledge" if it's not presented in a timely fashion on campus during school hours (see above). Students are expected only one day before commencement: from 9 p!m., unless explicitly instructed otherwise through instructor feedback form once all questions were received — see section 7-6 below.) The report continues with this line of inquiry as written; however Drs.
- **Custom**: Hello, my name is P.Nashin," she's a young woman with an infectious disease called Myalgic Fever. In March 2011 , the World Health Organization released statistics on 7th and 8 October of 2012 from 1 January 2013 to 15 April 2017 ( ). " The second part was already completed in December 2014". This had been decided by their manager 's decision : In his absence he made two signings as well at Swansea City for £1 million during that term - albeit without scoring twice since 2009-13

**Prompt**: The meaning of life is

- **Reference**: The meaning of life is not a function that we have to live in. Life requires us to be aware of what it means, how our bodies are shaped and changed by the world around us; living this way will lead to much greater success for ourselves as well as those who care about you." In other words: You're going to need people like me – or at least someone willing to give your body weight every day if so inclined - on an ongoing basis . I know all too intimately which types of women do best together (or don't) with whom but my advice would go out there…there's no one-size fits
- **Custom**: The meaning of life is that it has no intrinsic worth. The Lord knows his God and will give him power to do great things for others . " This would be the last year or we should have a few days like this, but I am still not sure how much money you'll pay in terms [for] some time]."

#### Native Sparse Attention

**Prompt**: In a shocking turn of events,

- **Reference**: In a shocking turn of events, the FBI is now investigating the attack. Federal agents say they have arrested a man wanted on terrorism charges for attempting to bomb a car in an Orlando nightclub, and the FBI has also seized two computers, laptops, and a phone, the Orlando [...]
- **Custom**: In a shocking turn of events, or two--
 points.
,.m. . The New York Times reported that I am proud to discuss how that is true and very often." is not what is known as the "d " . . . . the problem is that [...]

**Prompt**: The future of artificial intelligence

- **Reference**: The future of artificial intelligence is in the making. The US Department of Defense is developing a technology that could enable people to track and control the movement of robots, and can perform tasks such as alerting them to a threat. The Pentagon recently unveiled its " [...]
- **Custom**: The future of artificial intelligence, and the use of organic matter under conditions,,,,, 2000 to the following information gathered by the invention of protein products as a tool in the environment , where they were tested in accordance ( ). ) , and human beings for human and the [...]

**Prompt**: The meaning of life is

- **Reference**: The meaning of life is a whole and the body is a part of it. The human spirit is not a separate spirit from the human body. It must be regarded as a whole that is in the same. The human body is the whole of spirit and the human spirit is [...]
- **Custom**: The meaning of life is the right- or the person,,,,....@ @@.%!
, 9, @@.@.@ / .@/.@ .@ @@ ; - @@ =.@ @@ ;

#### Performer

**Prompt**: In a shocking turn of events,

- **Reference**: In a shocking turn of events, on 2 August last year he was arrested for the murder of two people in his flat. The victims were aged 21 and 22; both men are now dead...
- **Custom**: In a shocking turn of events, and to get drunk as well with little girls in the night before entering its relationship between friends who became an investigation. However much like their past...

**Prompt**: The future of artificial intelligence

- **Reference**: The future of artificial intelligence will involve creating a machine with the ability to solve complex problems and problem-solving skills that can be learned over time, by making choices based on which data is kept nearby...
- **Custom**: The future of artificial intelligence service, in the first instance; that both sides because there is a particular to be used for those who could not only one's power. Some are always have been very well-ease and his ability , it was at...

**Prompt**: As the sun set behind the towering mountains, the weary traveler finally caught sight of the distant village, its warm lights flickering like tiny stars

- **Reference**: As the sun set behind the towering mountains, the weary traveler finally caught sight of the distant village, its warm lights flickering like tiny stars. He was alone but in darkness for a moment before he heard his brother's cries and saw him pass by it...
- **Custom**: As the sun set behind the towering mountains, the weary traveler finally caught sight of the distant village, its warm lights flickering like tiny stars around a temple. The second floor which is still standing right and that has an ancient Egyptian tomb , so it was...

### Current Limitations

- **Minimal Dataset**: Synthetic or small text corpora, offering limited insight into real-world performance (we only use 1000 training samples).
- **Different Dataset**: trained attention optimization implementations on the WikiText2 dataset, which is different than the proprietary dataset that GPT2 was originally trainied on; could have contributed to weaker learning
- **Limited Training**: Currently our NSA and Performer implementations only train for 50 epochs.
- **No Large Model**: GPT-2 was used purely for demonstration; we have not tested on bigger or more modern architectures.
- **Compute Constraints**: Google Colab Notebooks only allowed us to trian on GPUs for ~3-4 hours per day; had to save checkpoints of models very often.
- **Limited Context Window**: compute and runtime constraints limited the size of the model and context window we could use

### Resource Usage Measurements

- On one T4 GPU on Google Colab, this took a while to run for the current number of epochs (~1 hr). These resource measurements are modest because our demonstration used a restricted sequence length and a small amount of data.

### Unexpected challenges

- **Limited Coherence**: We will need more sophisticated masking to handle longer contexts properly.
- **Loss Function**: Hard to identify the correct loss function to get similar, coherent outputs as original GPT2 model. Initially, we thought that using KL divergence would be enough, but the results were still not coherent. Adding next predicted token loss did not really improve the coherence of output text.
- **Compute constraints**: Google Colab has a bad user interface for coding and saving models and the runtime would often disconnect, leading us to having to rerun our notebooks multiple times.

## Future Steps

- **Different Dataset**: Use the original GPT2 model to generate training data to use so that the training data is similar to the original dataset used to train the GPT2 model. This would address some of the concerns of using a different dataset to train our attention optimizations than the proprietary dataset used to train GPT2.
- **Fine-Tune Hyperparameters:** Adjust learning rates and sequence lengths and tune hyperparameters more to improve stability and convergence.
- **Baseline Models**: Evaluate other baseline models besides GPT2
- **Loss Functions**: Try other loss functions besides KL divergence and other modifications to see if output text is more coherent.
- **Larger Context Windows**: train with more compute and larger context windows
- **Attention Restructuring**: Fix incoherent context later in the sentence generation

### What You've Learned So Far

- **Simple models work**: Out of our three implementations for attention optimizations, the simplest method of using linear combinations of attention masks had the best performance and was most similar to the original output. This is most likely due to the fact that the linear combination attention masks changed the GPT2 model the least so the model retained more information from it's original training.
- **Adaptive optimization methods**: We learned more about how different adaptive optimization methods change the performance of the final model. We tried different optimizers when determing the optimal parameters in the attention layers to . We ended up utilizing the AdamW optimizer for our final implementation because it decouples the weight decay from the adaptive udpate mechanism.
- **Learning rate schedulers**: We learned how learning rate schedulers can help training models when the parameters get stuck in a local minimum during training. When the training the Performer implementation, we initially hit a noise floor and the scheduler helped get past this. We settled on using the CosineAnnealing scheduler.
- **Attention optimizations**: Experimenting with differnet attention optimizers gave us better insight into how transformers work and latest research into how people are optimizing attention layers to improve memory and time efficiency during training.
- **Attention Substitution**: Swapping out the standard self-attention module is straightforward if we mirror the input-output shapes and track weights carefully.
