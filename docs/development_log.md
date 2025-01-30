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
