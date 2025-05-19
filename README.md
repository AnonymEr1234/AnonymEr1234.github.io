# AnonymEr1234.github.io
Differential Transformer: Enhancing Attention for Large Language Models

In the rapidly evolving field of AI, researchers at Microsoft Research and Tsinghua University have introduced a new architecture called Differential Transformer (DIFF Transformer) that addresses a fundamental limitation in traditional Transformer models.
The Problem: Attention Noise

Traditional Transformers struggle with a critical issue: they tend to overallocate attention to irrelevant context. This "attention noise" makes it difficult for models to focus on the most important information, leading to problems with long context use cases.

The Differential Transformer architecture provides a solution for this by trying to cancel out attention-noise just like noise cancelling headphones. The paper shows promising results especially with long contexts and long sentences. 
