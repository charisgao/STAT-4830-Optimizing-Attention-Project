# Week 3 LLM Exploration Summary

## Session Focus

Our goal was to find and develop a project idea.

## Surprising Insights

Claude was more helpful compared to ChatGPT at generating ideas. When presented with the MNIST example idea, ChatGPT's suggestions were all very similar to the example, such as "Dynamic Object Tracking in Traffic Video Feeds" to use dynamic motion patterns to improve accuracy for object detection in traffic scenarios or "Sentiment Drift Detection in Social Media" to analyze how public sentiment about a topic evolves over time. However, these responses did not have clear optimization focuses and were not as helpful for our project ideation.

Also, Claude was better than ChatGPT at long conversations and following the `finding_project_ideas.md` doc. After a few chat messages, ChatGPT forgot many of the focuses of the `finding_project_ideas.md` doc, while Claude continued to follow the structure of Problem Exploration, Reality Check, Mathematical Formulation, etc. without extra prompting.

### Conversation: Finding Preliminary Ideas

**Prompt That Worked:**
"Walk through this guide with me. I'm not sure about problems I've encountered but here is some information about me that could be useful for the quickstart. I am a computer science major and have enjoyed taking psychology and criminology courses as part of my electives. In particular, I took a psychology course about judgments and decisions, and it mentioned a book, Thinking Fast and Slow, which was referenced by OpenAI in the development of o1. Course projects that have been cool involved using AI to build a classifier of different abstracts to subjects areas. Please help me think of some ideas for my project following this guide and using the information I just shared."

**Key Insights:**
Instead of immediately jumping to produce a list of ideas, Claude asked follow up questions about specific inefficiencies I might have encountered at different intersections (eg. decision making and CS). It wanted to use my responses to its questions as starting points to identify a problem that combines my technical skills with my interests before diving deeper into specific aspects. I was also surprised by how Claude was really good at following the guide's structured approach to develop a concrete project idea.

### Conversation: Choosing the Best Idea

**Prompt That Worked:**
[List of project ideas and brief outline of objectives and possible approaches.] "Which idea is best for an optimization project in PyTorch. Think about the ideas beyond the samples outlines we gave and also look at the project information doc attached for technical feasibility and goals."

**Key Insights:**
I was surprised that Claude gave detailed pros/cons about the ideas that were very realistic. For example, for our AI-driven GPU scheduling ideas, Claude was able to recognize that it's hard to validate without real GPU infrastructure which we likely would not have a lot of access to because we are students. After explaining that NAS is the strongest choice, it also gave an example PyTorch implementation, scalability ideas, and learning value for a school project.

### Conversation: Refining NAS Idea

**Prompt That Worked:**
"Tell me more about this project idea. Is the neural architecture search for a particular task?"... and "How do OFA networks work?"

**Key Insights:**
I was surprised by Claude's ability to suggest focused applications of NAS beyond general architecture search, such as privacy-aware NAS and multi-modal fusion architecture search. Claude also explained the evolution of NAS methods, specifically from DARTS to more modern approaches like OFA, PC-DARTS, and FairDARTS showing that Claude does a lot of research. Also when discussing datasets, Claude went beyond just listing options and provided structured recommendations based on complexity levels and computational requirements, which was helpful since it doesn't know the resources we have available for a student project.

## Overall Techniques That Worked

- Providing links and documents and asking the model to follow along.
- Asking for specific techncial implementations rather than just a list of examples or theoretical explanations.
- Expressing concerns about existing solutions led to more focused suggestions.
- Following up on technical concepts I was confused about– Claude still knew the history of the conversation and the relevant context.

## Dead Ends Worth Noting

- Giving ChatGPT an example and tell me to give me more suggestions similar to the example --> Scope is too limited and ChatGPT fixates on the example and finds very very similar examples.

## Next Steps

- Potentially explore other ChatGPT models. Overall experience with 4o wasn't the best but could be worth trying o1 due to its chain-of-thought prompting ability.
- Use LLMs to analyze technical implementations and give feedback on our report.
