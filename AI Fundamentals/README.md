# AI Fundamentals


---

# Section 1: LLM and Introduction to LLM

---

## 1.1 What is an LLM?

### Simple Definition

**LLM = Large Language Model**

An LLM is a machine-learning model trained on a very large amount of text so that it can learn **patterns and relationships in language**.

A simple way to think about it:

> **An LLM is a very powerful next-token prediction system.**

For example:

```text
Input:
"Happy birthday to"

Possible continuation:
"you"
```

The important idea is:

```text
Given previous context
        ↓
Predict the next token
        ↓
Add that token to the context
        ↓
Predict the next token
        ↓
Repeat
```

### Important clarification

The phrase **"LLM predicts the next word"** is a useful beginner analogy, but technically an LLM generally predicts the **next token**, not necessarily an entire word.

That distinction becomes important once we understand tokenization.

---

## 1.2 Why Does an LLM Seem Intelligent?

A normal autocomplete system may know common patterns such as:

```text
"Happy birthday to" → "you"
```

An LLM is trained on vastly larger datasets and can learn much more complicated patterns involving:

- language
- facts present in its training data
- writing styles
- relationships between concepts
- patterns across sentences
- programming code
- mathematical patterns

### Simple analogy

Imagine two people completing a sentence:

```text
Person A:
Has read 100 books

Person B:
Has read 10 million books
```

Person B has encountered far more language patterns.

An LLM works on the same basic idea, but at a vastly larger computational scale.

---

## 1.3 Does an LLM Read the Internet Every Time You Ask a Question?

**No.**

This is an important distinction.

A standard LLM does **not automatically search the internet for every question**.

Instead, during training, the model learns patterns from large amounts of training data. Later, when you provide a prompt, it processes that input using what it has learned.

Some AI systems **can** use web-search tools, databases, RAG systems, APIs, etc., but that is an additional capability around the model rather than the basic definition of an LLM.

---

## 1.4 The LLM Does Not Directly Process Human Text

Suppose we give an LLM:

```text
I love AI
```

Humans immediately recognize:

```text
"I"
"love"
"AI"
```

as meaningful language.

A neural network, however, ultimately performs numerical computation.

At a very high level:

```text
Human text
    ↓
Tokenization
    ↓
Tokens
    ↓
Token IDs
    ↓
Vectors / representations
    ↓
Mathematical processing
    ↓
Predicted next token
    ↓
Generated response
```

---

# Section 2: Tokens

---

## 2.1 What is a Token?

### Simple Definition

A **token is a small piece of text that the model processes.**

For example:

```text
"I love AI"
```

might be broken into something conceptually similar to:

```text
"I"
" love"
" AI"
```

But this is only an illustration.

---

## 2.2 Important: Token ≠ Word

This is one of the most important concepts.

A token is **not necessarily one complete word**.

For example:

```text
playing
```

could be split conceptually into:

```text
play
ing
```

A longer or less common word can be divided into several pieces. A word such as `"unbelievable"` can be broken into multiple token pieces. Spaces and punctuation can also participate in tokenization depending on the tokenizer.

So:

```text
1 word ≠ always 1 token
```

Instead:

```text
Text
 ↓
Tokenizer
 ↓
Tokens
```

---

## 2.3 What is Tokenization?

### Definition

**Tokenization is the process of breaking text into tokens.**

Example:

```text
Input:
"AI is amazing!"

Possible conceptual tokenization:

"AI"
" is"
" amazing"
"!"
```

The exact result depends on the tokenizer.

Different models can use different tokenization schemes, so the same text does **not necessarily produce the same tokens across different models**. Systems such as ChatGPT, Claude, and LLaMA can have different tokenizers and vocabularies.

---

## 2.4 Why Do We Need Tokens?

Computers ultimately perform computation on **numbers**.

So instead of giving the neural network raw human language directly:

```text
"I love AI"
```

the text is transformed into pieces that can ultimately be represented numerically.

A simplified pipeline is:

```text
"I love AI"
     ↓
Tokenization
     ↓
["I", " love", " AI"]
     ↓
Token IDs
     ↓
[ ... numbers ... ]
     ↓
Neural network processing
```

---

## 2.5 What is a Token ID?

Once text is tokenized, each token is associated with an **integer ID** from the model's vocabulary.

For example, purely as an illustration:

```text
Token       Token ID
--------------------
apple       745
the         2
model       81
```

These numbers are **identifiers**, not meanings.

Different models/tokenizers have different vocabularies and token-ID mappings.

---

## 2.6 Very Important — Token ID Does NOT Represent Meaning

This is the bridge to embeddings.

Imagine:

```text
dog      → 500
airplane → 501
puppy    → 9421
```

Looking only at the IDs:

```text
500
501
9421
```

we cannot conclude:

```text
dog ≈ puppy
```

and

```text
dog ≠ airplane
```

because the numbers are simply identifiers.

**Token IDs alone cannot represent semantic similarity.**

### Think of a token ID like a database ID

```text
Employee ID:
101 → Rahul
102 → Priya
999 → Amit
```

Does:

```text
101 and 102
```

mean Rahul and Priya are similar?

No.

The number is just an identifier.

The same basic principle applies to token IDs.

---

# Section 3: What Are Embeddings?

---

## 3.1 So How Does the Model Represent Meaning?

This is where **embeddings** enter the picture.

Instead of treating the token as only:

```text
dog → 500
```

the model associates it with a numerical vector.

Conceptually:

```text
dog → [0.9, 0.9, 0.0, ...]
```

A token ID selects the corresponding row/vector from a matrix, which contains many floating-point values.

---

## 3.2 What is an Embedding?

### Simple Definition

> **An embedding is a numerical vector representation of something such as a token, sentence, document, or other data.**

For our current discussion:

```text
Token
  ↓
Numerical vector
  ↓
Embedding
```

An embedding represents a token using a dense vector of numbers.

---

## 3.3 What is a Vector?

A **vector is simply a list of numbers.**

Example:

```text
[0.9, 0.2, 0.7]
```

A real model may use a vector with hundreds or thousands of dimensions rather than only three.

For learning purposes, imagine:

```text
dog → [0.9, 0.9, 0.0]
```

Each number can be thought of as representing some learned feature in an intuitive illustration.

---

## 3.4 Understanding Vector Dimensions

A simplified example with three dimensions:

```text
Dimension 1 → pet-like
Dimension 2 → has a tail
Dimension 3 → can fly
```

Then imagine:

### Dog

```text
dog → [0.9, 0.9, 0.0]
```

### Puppy

```text
puppy → [0.8, 0.8, 0.0]
```

### Airplane

```text
airplane → [0.0, 0.0, 1.0]
```

So conceptually:

```text
            Pet-like   Tail   Can-fly
Dog            0.9      0.9      0.0
Puppy          0.8      0.8      0.0
Airplane       0.0      0.0      1.0
```

These values are **not manually assigned by a person**.

The model **learns its numerical representations during training**.

### Important clarification

The "pet-like", "has a tail", etc. example is only an intuition-building visualization.

Real neural-network embedding dimensions generally do **not** correspond cleanly to human-readable labels such as:

```text
dimension 37 = "has tail"
```

Instead, the dimensions are learned numerical features whose meaning can be distributed and difficult to interpret individually.

---

## 3.5 How Does the Model Learn These Numbers?

During training, the model processes huge amounts of text and learns statistical patterns and relationships.

Over time, its parameters are adjusted so that it becomes better at its prediction task.

That learning produces numerical representations that capture useful relationships.

Conceptually:

```text
Huge amount of training text
          ↓
     Training process
          ↓
Learned model parameters
          ↓
Useful numerical representations
          ↓
Relationships between tokens/concepts
```

---

## 3.6 How Can Vectors Represent Similarity?

Imagine:

```text
Dog   → [0.9, 0.9, 0.0]
Puppy → [0.8, 0.8, 0.0]
```

They have similar numerical patterns.

Whereas:

```text
Airplane → [0.0, 0.0, 1.0]
```

looks quite different.

So we can use mathematics to compare vectors.

One mathematical operation used for this is the **dot product**.

---

## 3.7 Dot Product — Simple Understanding

Suppose:

```text
A = [0.9, 0.9, 0.0]

B = [0.8, 0.8, 0.0]
```

The dot product is:

```text
(0.9 × 0.8)
+
(0.9 × 0.8)
+
(0.0 × 0.0)
```

= **1.44**

Mathematically similar representations produce a stronger similarity signal.

---

## 3.8 Vector Space — The Big Picture

We can imagine each vector as a point in a mathematical space.

Conceptually:

```text
             Puppy
               ●
            ● Dog
         ● Cat


                         ● Airplane
```

Similar concepts can occupy nearby regions, while very different concepts may be farther apart.

### Important clarification

"Closer = more similar" is a useful intuition, but the exact similarity operation depends on the application.

For embedding search, **cosine similarity** and **dot product** are both common choices, among others.

---

## 3.9 The Complete Pipeline

Now we can connect everything.

Suppose the user asks:

```text
How do I apply for medical reimbursement?
```

A simplified pipeline is:

```text
User Query
    ↓
"How do I apply for medical reimbursement?"
    ↓
Tokenization
    ↓
Tokens
    ↓
Token IDs
    ↓
Numerical representations
    ↓
Model processing
    ↓
Learned relationships / contextual representations
    ↓
Prediction / retrieval / generation
    ↓
Answer
```

Semantically related concepts such as *reimbursement* and *refund* can be connected through numerical representations rather than exact string matching.

---

## 3.10 Exact Match vs Semantic Understanding

This distinction is extremely important for modern AI applications.

### Traditional keyword matching

Suppose a document contains:

```text
"Medical refund policy"
```

User asks:

```text
"How do I claim medical reimbursement?"
```

A simple exact keyword system may struggle because:

```text
refund ≠ reimbursement
```

as exact strings.

### Semantic approach

Embeddings allow related concepts to be represented numerically.

Conceptually:

```text
"refund"
      ↘
       Similar semantic region
      ↗
"reimbursement"
```

So a semantic retrieval system can find relevant information even when the exact words differ.

This idea leads directly into:

```text
Embeddings
     ↓
Semantic Search
     ↓
Vector Database
     ↓
RAG
```

---

## 3.11 Embeddings Are Not Only for Tokens

The beginner explanation starts with token embeddings, but the broader concept is much more powerful.

We can create embeddings for things such as:

```text
Token
Sentence
Paragraph
Document
Image
Audio
Product
User profile
```

The basic idea remains:

```text
Real-world information
        ↓
Numerical representation
        ↓
Vector
```

Then vectors can be compared mathematically.

---

## 3.12 Why Embeddings Matter in Real Applications

Suppose a company has:

```text
10,000 internal documents
```

A user asks:

```text
"How can I claim money spent on hospital treatment?"
```

The relevant document might say:

```text
"Medical reimbursement policy"
```

The wording is different, but the **meaning is related**.

A semantic retrieval system can use embeddings to identify the relevant document.

Then a RAG system can use the retrieved information to generate an answer.

---

## 3.13 Embeddings + Vector Database + RAG

This is an important mental model for your AI journey.

```text
                 USER
                   │
                   ▼
              User Query
                   │
                   ▼
              Embedding
                   │
                   ▼
           Vector Representation
                   │
                   ▼
          ┌─────────────────┐
          │ Vector Database │
          └─────────────────┘
                   │
             Similar Documents
                   │
                   ▼
              Retrieved Data
                   │
                   ▼
                 LLM
                   │
                   ▼
               Final Answer
```

This is the basic foundation behind many modern **RAG applications**.

---

## 3.14 An Important Correction: Embedding ≠ Complete "Meaning"

It is tempting to say:

> "The embedding is the meaning of the word."

That is useful for beginners, but technically it is an oversimplification.

A better statement is:

> **An embedding is a learned numerical representation that captures useful patterns and relationships in the data.**

It does not contain a perfect dictionary definition of a word.

For example, the meaning of a word can depend heavily on context:

```text
"I deposited money in the bank."

"The plane reached the river bank."
```

The word:

```text
bank
```

has different meanings depending on context.

This is one reason modern transformer models create **contextual representations** during processing rather than relying only on one fixed vector for a word/token.

---

## 3.15 Token Embedding vs Contextual Representation

This distinction will become very important later.

### At the beginning

A token ID can be mapped to an initial learned vector:

```text
Token ID
   ↓
Embedding matrix
   ↓
Initial token embedding
```

### Inside the Transformer

The model processes these representations using mechanisms such as:

```text
Attention
+
Neural-network layers
+
Other transformations
```

The representation becomes **context-dependent**.

So:

```text
"bank"
```

in:

```text
I went to the bank.
```

and:

```text
The boat reached the river bank.
```

can end up with different contextual representations.

We will study this deeply when we reach **Transformers and Attention**.

---

## 3.16 What is an Embedding Matrix?

This is a useful technical concept.

Imagine the model has:

```text
Vocabulary size = 50,000 tokens
Embedding dimension = 768
```

Conceptually, the model can have an embedding matrix:

```text
             768 dimensions
        ┌───────────────────────┐
Token 0 │ 0.12  0.91 ...        │
Token 1 │ 0.44  0.17 ...        │
Token 2 │ 0.72  0.63 ...        │
Token 3 │ 0.15  0.29 ...        │
   .    │          .            │
   .    │          .            │
Token N │ 0.31  0.78 ...        │
        └───────────────────────┘
```

A token ID effectively points to a row in this matrix.

Conceptually:

```text
Token ID = 500
       ↓
Embedding Matrix
       ↓
Row 500
       ↓
[0.21, 0.73, 0.44, ...]
```

---

# Summary & Review

---

## Tokens vs Token IDs vs Embeddings

This is one of the most important comparisons to remember.

| Concept | Meaning | Example |
|---|---|---|
| **Text** | What humans write | `"puppy"` |
| **Token** | A piece of text | `"puppy"` or a subword piece |
| **Token ID** | Integer identifier for that token | `9421` |
| **Embedding** | Numerical vector representing the token | `[0.8, 0.8, 0.0, ...]` |
| **Contextual representation** | Representation after processing context | Changes depending on surrounding text |

### Easy analogy

Think about a product in an e-commerce system:

```text
Product name
     ↓
"iPhone"
     ↓
Product ID
     ↓
48291
     ↓
Feature representation
     ↓
[screen, price, category, ...]
```

The ID identifies the item.

The feature representation helps describe its characteristics.

Similarly:

```text
Token → Token ID → Vector representation
```

---

## The Three Fundamental Concepts

At this stage, keep these three ideas very clear:

### 1. Token

> A small piece of text.

```text
"playing"
   ↓
"play" + "ing"
```

### 2. Token ID

> A numerical identifier assigned to a token in a tokenizer's vocabulary.

```text
"dog" → 500
```

### 3. Embedding

> A learned numerical vector associated with a token or other data.

```text
"dog"
  ↓
[0.9, 0.9, 0.0, ...]
```

---

## The Full Mental Model

Remember this flow:

```text
                 HUMAN
                   │
                   ▼
               Text Input
                   │
                   ▼
              Tokenization
                   │
                   ▼
                 Tokens
                   │
                   ▼
               Token IDs
                   │
                   ▼
          Embedding / Vectors
                   │
                   ▼
       Transformer Processing
                   │
                   ▼
      Contextual Representations
                   │
                   ▼
          Next-Token Prediction
                   │
                   ▼
             Generated Text
```

This is the mental model you should carry forward.

---

## Why These Concepts Matter for Your AI Journey

You will repeatedly encounter these terms when learning:

```text
LLM
Token
Context Window
Embedding
Vector
Vector Database
Semantic Search
RAG
Attention
Transformer
AI Agents
```

And they are connected.

A useful roadmap is:

```text
LLM
 │
 ├── Tokens
 │     │
 │     └── Token IDs
 │
 ├── Embeddings
 │     │
 │     ├── Vector Search
 │     └── Semantic Search
 │             │
 │             └── Vector Database
 │                     │
 │                     └── RAG
 │
 └── Transformer
       │
       ├── Attention
       └── Contextual Representations
```

---

## Must Know ⭐

These are the concepts you should be able to explain in an interview.

### LLM

An **LLM (Large Language Model)** is a model trained on large amounts of text to learn language patterns and generate text by predicting tokens based on context.

### Token

A **token is a piece of text** processed by the model. It may be a whole word, part of a word, punctuation, or another tokenizable piece.

### Tokenization

The process of converting text into tokens.

### Token ID

A numerical identifier corresponding to a token in the tokenizer's vocabulary.

### Embedding

A learned numerical vector representation of data such as tokens, text, documents, or other objects.

### Semantic Similarity

Comparing vector representations to estimate how closely related pieces of information are.

### Vector Database

A system designed to efficiently store and search vector representations.

---

## Good to Know 👍

You should understand these conceptually:

- Different models can use different tokenizers.
- Token IDs are identifiers, not semantic scores.
- Embedding dimensions are learned rather than manually assigned.
- Similar vectors can represent related concepts.
- Similarity can be measured mathematically.
- Embeddings are fundamental to semantic search.
- Embeddings are heavily used in RAG systems.

---

## Advanced / Optional 🧠

These topics are worth learning after the fundamentals:

```text
BPE / WordPiece / SentencePiece
        ↓
Embedding Matrix
        ↓
Positional Information
        ↓
Self-Attention
        ↓
Transformer Architecture
        ↓
Contextual Embeddings
        ↓
Cosine Similarity
        ↓
Vector Indexing
        ↓
ANN Search
        ↓
RAG
```

These should be learned **after** the basic mental model is clear.

---

## Common Beginner Confusions

### ❌ Token = Word

Not always.

```text
word → one or more tokens
```

---

### ❌ Token ID = Meaning

No.

```text
500
```

is an identifier.

It does not mean:

```text
"dog is very similar to puppy"
```

---

### ❌ Embedding is a manually created feature list

No.

The model learns its numerical representations during training.

---

### ❌ LLM searches Google for every question

Not inherently.

An LLM can answer using its learned model parameters; web search or retrieval requires additional tooling/system design.

---

### ❌ LLM predicts only complete words

Technically, the model predicts **tokens**, which may be smaller than a complete word.

---

## Interview Questions

### Q1. What is an LLM?

> An LLM, or Large Language Model, is a machine-learning model trained on large amounts of text to learn language patterns and generate text. At a high level, it predicts the next token based on the previous context.

---

### Q2. What is a token?

> A token is a small piece of text processed by an LLM. It can be a complete word, part of a word, punctuation, or another tokenized unit.

---

### Q3. Is every word one token?

> No. A word can be represented by multiple tokens depending on the tokenizer.

---

### Q4. What is a token ID?

> A token ID is an integer assigned to a token in the tokenizer's vocabulary. It identifies the token but does not represent its semantic meaning.

---

### Q5. Why can't we use token IDs to measure similarity?

> Because token IDs are arbitrary identifiers. Numerically close IDs do not necessarily represent semantically similar tokens.

---

### Q6. What is an embedding?

> An embedding is a learned numerical vector representation of an object such as a token, sentence, or document.

---

### Q7. Why are embeddings useful?

> Embeddings allow information to be represented numerically so that relationships and similarities can be measured mathematically.

---

### Q8. Where are embeddings used?

> Common applications include semantic search, recommendation systems, clustering, vector databases, and RAG systems.

---

### Q9. What is the difference between a token and an embedding?

```text
Token
→ piece of text

Embedding
→ numerical vector representation
```

---

### Q10. What is the high-level LLM pipeline?

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Numerical representations
 ↓
Transformer processing
 ↓
Next-token prediction
 ↓
Generated response
```

---

## Quick Revision ⚡

Remember this chain:

```text
TEXT
 ↓
TOKENIZATION
 ↓
TOKENS
 ↓
TOKEN IDs
 ↓
EMBEDDINGS / VECTORS
 ↓
MATHEMATICAL PROCESSING
 ↓
CONTEXTUAL REPRESENTATIONS
 ↓
NEXT-TOKEN PREDICTION
 ↓
OUTPUT
```

### The easiest way to remember it

> **Tokens tell the model what pieces of text it received.**

> **Token IDs identify those pieces.**

> **Embeddings turn them into learned numerical representations.**

> **The model processes those representations to understand patterns in context and predict what comes next.**

And this is the foundation for:

```text
Embeddings
     ↓
Semantic Search
     ↓
Vector Databases
     ↓
RAG
     ↓
Modern AI Applications
```