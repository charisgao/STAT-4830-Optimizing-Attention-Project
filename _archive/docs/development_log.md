# Development Log

## Week 3

### Project Ideation

We started the week exploring multiple project directions through our ideation process and working through the `finding_project_ideas.md` doc. Initial candidate ideas included:

1. AI-Driven GPU Scheduling for resource optimization
2. Neural Architecture Search (NAS)
3. Multimodal Math Problem Solver for processing handwritten equations and diagrams

### NAS Deep-Dive

We initially settled on Neural Architecture Search and wanted to focus on:

- Systematic optimization of neural network architectures to balance performance metrics
- Exploration of methods including RL, DARTS, and ENAS
- Multi-objective optimization across accuracy, model size, FLOPs, and memory usage

Our proof-of-life plan involved a scaled-down experiment on CIFAR-10 to validate the core pipeline.

### 1/28 Meeting with Prof. Davis

We presented our idea to Prof. Davis and he expressed concerns that a basic architecture might already work pretty well and there could be no further optimization steps. Also, there are no downstream objectives beyond predictive accuracy and working within hardware constraints.

After further discussion, we pivoted to optimizing the attention mechanisms in transformer models. We will work with a pretrained transformer model and our goal will be to approximate it as well as possible (maintaining downstream performance) while making the attention mechanism cheaper to compute. We plan to start with having fixed attention masks and having the model dynamically optimize a linear combination weighting (which attention components are more important). Potential stretch goals involve the model learning the best attention patterns (without attention masks).

### Last-10-Tokens Implementation

We coded a toy implementation that replaced the full self-attention layer of GPT-2 with a fixed last-10-tokens window, and froze most GPT-2 parameters except for our custom attention block and some MLP layers. We trained by minimizing KL-divergence between the custom model's outputs and the reference GPT-2 outputs. The purpose of this implementation was for a proof of concept and to ensure we were actually able to change the attention layer.

## Week 4

### 2/4 Meeting with Prof. Davis

We presented our progress and demonstrated our proof of concept implementation with Prof. Davis. We also outlined next steps, which focus on training and optimizing weights for a linear combination of attention masks. To improve efficiency, we plan to explore various penalties, including L1 regularization, which could help drive some coefficients to zero.

Prof. Davis provided valuable feedback and suggested two key approaches to enhance our work. First, he recommended incorporating a regularizer or constraint to find low-rank masks, which could improve computational efficiency. Second, he proposed optimizing over different types of matrix families, particularly highlighting sliding windows. He also noted that circulant matrices are cheap to store (only need one vector), which could help improve memory issues.

## Week 5

### Weighted Combination of Attention Masks

As a next step, we replaced the fixed last-10-tokens window with a learnable weighted linear combination of three candidate masks: candidate 0 only attends to the last 5 tokens, candidate 1 only attends to the last 10 tokens, and candidate 2 only attends to the first 5 tokens. We also added a L1 penalty when optimizing the coefficients and similarly trained to minimize the KL-divergence between the custom model's outputs and reference model. Our next step will be including more candidate masks in the linear combination and experimenting with different types of masks beyond first _ or last _ tokens.

## Week 6

### 2/18 Meeting with Prof. Davis

We discussed our progress from a fixed last-10-tokens window to a linear combination of candidate masks. We also explained another experimential implementation with learning an attention mask per position, but did not achieve promising results since there's too much variability in what tokens each position should attend to for each sentence. We also outlined next steps, specifically to add more candidate masks, explore regularization, and add speed and memory usage tracking.

Prof. Davis mentioned that a DeepSeek paper came out today (https://arxiv.org/abs/2502.11089) and that we should do more literature review to explore how they optimize for sparse attention. He also suggested we can look into other models such as Mistral that already optimize attention and learn techniques from them.

## Week 7

### Weighted Combination of Attention Masks

We replaced the learnable weighted linear combination of three candidate masks with five candidate masks: attend to the last token, 2nd to last token, 3rd to last token, 4th to last token, and 5th to last token. We also conducted experiments exploring the effects of the L1 penalty and attempted to tune the hyperparameter of how much L1 penalty to apply. Our next step will be including even more candidate masks in the linear combination and also looking into more state of the art approaches towards this attention problem.

## Week 8 - Midterm Presentation

### 3/6 Meeting with Prof. Davis

Prof. Davis gave us feedback from our midterm presentation. Regarding the presentation itself, he mentioned that visualizations throughout would help, such as showing what attention masks look like, what transformers look like, and displaying the key mathematical formulas as formulas. With our implementation so far, he recommended exploring the context length and the bottleneck to see how long should our context actually be? He also explained kernels and how we could use kernels to optimize the attention mechanism. Instead of forming the query and key from some weight matrix, we could learn a neural network and use that for self attention.

## Week 9

### Implementing Performer and NSA

Instead of continuing with learning weighted combinations of attention masks, we decided to explore other more current methods of improving the self attention mechanism. Performers use a linear transformer with random feature approximations of the softmax kernel. NSA is part of our previous literature review, and a state of the art approach that uses a sparse attention mechanism that incorporates predefined sparsity directly into model training. We aimed to implement both of these approaches.

## Week 10

### 3/25 Meeting with Prof. Davis

We met with Prof. Davis to discuss our implementations and shift to exploring existing state of the art approaches to addressing quadratic attention. We mentioned difficulties we had with our code, namely lots of NaNs, infinity values, and incoherent output. Prof. Davis responded with how these issues are pretty common and gave us tips to look into. He also described that it wasn't exactly clear what were were optimizing in our approach if we're still using KL divergence, and we mentioned that we wanted to use the Performer / NSA as a replacement for the attention layer in GPT2, and still want similar performance as GPT2.

## Week 11

### Implementing Performer, Kerformer, NSA

We improved on our past implementations of Performer and NSA by debugging and now are able to get more coherent output. For Performer, a key improvement involved standardization, and for NSA, we were able to align the tensors and increase the context length. We also began implementing Kerformer, but ran into some issues with out of GPU and CPU memory.

## Week 12

### 4/8 Meeting with Prof. Davis

We discussed our progress with Prof. Davis. For the Performer, from looking at our loss values, he hypothesized that the KL-divergence doesn't decrease further since we are using a constant step size and hit the noise floor. He recommended that we decrease the learning rate, specifically by exploring the cosine annealing scheduler. Prof. Davis also suggested looking into changing the context length and number of features. For the NSA, he similarly suggested increasing the context length since the output becomes incoherent after a few tokens.
