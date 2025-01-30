# Development Log

## Week 3

### Project Evolution

Started the week exploring multiple project directions through our ideation process. We worked through the `finding_project_ideas.md` doc. Initial candidate ideas included:

1. AI-Driven GPU Scheduling for resource optimization
2. Neural Architecture Search (NAS)
3. Multimodal Math Problem Solver for processing handwritten equations and diagrams

### NAS Deep-Dive

We initially settled on Neural Architecture Search, developing a proposal focused on:

- Systematic optimization of neural network architectures to balance performance metrics
- Exploration of methods including RL, DARTS, and ENAS
- Multi-objective optimization across accuracy, model size, FLOPs, and memory usage

Our proof-of-life plan involved a scaled-down experiment on CIFAR-10 to validate the core pipeline.

### Direction Change

After meeting with Prof. Davis, we pivoted to focus on optimizing attention mechanisms in transformer models. New direction highlights:

- Dynamic determination of important attention components
- Exploration of attention approximation methods
- Focus on making attention computation more efficient
- Investigation of structured attention approaches (e.g., Longformer)
- Working with pretrained transformers to benchmark approximation quality
