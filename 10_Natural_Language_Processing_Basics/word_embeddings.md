# 📘 Word Embeddings — Theory

---

## 📌 What are Word Embeddings?

Word embeddings are **dense, low-dimensional vector representations** of words  
that capture **semantic and syntactic relationships** — words with similar  
meanings are represented by vectors that are close together in the embedding space.

```
Traditional (one-hot encoding):        Word Embeddings:
  "king"  → [1,0,0,0,0,0,...]           "king"  → [0.21, -0.45, 0.83, ...]
  "queen" → [0,1,0,0,0,0,...]           "queen" → [0.18, -0.41, 0.79, ...]
  "man"   → [0,0,1,0,0,0,...]           "man"   → [0.22, -0.47, 0.02, ...]
  "woman" → [0,0,0,1,0,0,...]           "woman" → [0.19, -0.43, 0.01, ...]

  d = vocabulary size (sparse)           d = 50–300 (dense)
  No semantic meaning                    Semantics encoded in geometry
```

**Famous analogy:**
```
king − man + woman ≈ queen

Vector arithmetic captures semantic relationships:
  Paris − France + Italy ≈ Rome   (capitals)
  walked − walk + run ≈ ran       (verb tenses)
  bigger − big + small ≈ smaller  (comparatives)
```

> 💡 "Word embeddings turn language into geometry — where distance  
>      and direction encode meaning."

---

## 🔍 Why Word Embeddings?

```
One-Hot Encoding problems:
  ❌ Vocabulary size = thousands to millions → very high dimensional
  ❌ Completely sparse (one 1, rest 0s) → wasteful
  ❌ No semantic information — "cat" and "kitten" are orthogonal
  ❌ Cannot generalize — model learns nothing about unseen words

Word Embeddings solve this:
  ✅ Dense, low-dimensional (50–300 dimensions vs 50,000+)
  ✅ Semantic similarity encoded as cosine distance
  ✅ Generalizes across similar words
  ✅ Can be pre-trained on huge corpora → transfer learning
```

---

## 🧮 Word2Vec — The Foundation

Word2Vec (Mikolov et al., 2013) learns embeddings by training a shallow  
neural network to **predict words from context** (or context from words).

### Architecture 1: Continuous Bag of Words (CBOW)

```
Idea: predict center word from surrounding context words

Context words → [avg embedding] → hidden layer → predict center word

Example (window=2):
  "The quick [brown] fox jumps"
  Context: {The, quick, fox, jumps} → predict "brown"

Training signal: maximize P(brown | The, quick, fox, jumps)
```

### Architecture 2: Skip-Gram

```
Idea: predict surrounding context words from center word (reverse of CBOW)

Center word → hidden layer → predict each context word

Example (window=2):
  "The quick [brown] fox jumps"
  Input: "brown" → predict: {The, quick, fox, jumps}

Training signal: maximize P(The|brown) × P(quick|brown) × ...

Skip-gram learns better embeddings for rare words.
CBOW is faster and works better for frequent words.
```

### Negative Sampling

```
Full softmax over vocabulary is too expensive (50,000+ classes).
Negative Sampling approximates it:

For each positive (word, context) pair:
  Sample k negative words randomly
  Train binary classifier: real pair (1) vs random pair (0)

Typical k: 5–20 for small datasets, 2–5 for large

Loss = log σ(vᵥᵀ vᵤ) + Σₖ log σ(−vₙᵢᵀ vᵤ)
```

---

## 🔠 GloVe — Global Vectors

GloVe (Pennington et al., 2014) learns embeddings from the **global  
word-word co-occurrence matrix** of the entire corpus.

```
Co-occurrence matrix X:
  X[i,j] = number of times word j appears in context of word i

Objective:
  Minimize: Σᵢⱼ f(Xᵢⱼ) × (wᵢᵀ w̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²

  f(x) = weighting function that reduces influence of very frequent pairs

Key insight:
  The RATIO of co-occurrence probabilities encodes meaning:
  P(ice|solid) / P(steam|solid) >> 1  (ice is more related to solid than steam)

GloVe vs Word2Vec:
  Word2Vec: local context (sliding window)
  GloVe:    global corpus statistics (full co-occurrence matrix)
  Both produce similar quality embeddings in practice
```

---

## 🔤 FastText — Subword Embeddings

FastText (Bojanowski et al., 2017) extends Word2Vec by representing  
words as **bags of character n-grams**.

```
Word: "playing"
Character n-grams (n=3): <pl, pla, lay, ayi, yin, ing, ng>
                          <playing>  (whole word)

Word vector = sum of n-gram vectors

Advantages:
  ✅ Handles out-of-vocabulary (OOV) words
  ✅ Better for morphologically rich languages
  ✅ Works well for rare words (builds from subwords)
  ✅ "unhappiness" benefits from "happy", "happiness", "un-"

Disadvantage:
  ❌ Larger model (stores n-gram vectors)
  ❌ Slower training

OOV handling:
  Word2Vec: maps OOV to random/zero vector
  FastText: builds from subword n-grams → meaningful representation!
```

---

## 🤖 Contextual Embeddings — BERT and Beyond

Word2Vec / GloVe / FastText are **static** — each word has one fixed vector  
regardless of context. But "bank" means different things in different sentences:

```
Static embeddings (Word2Vec):
  "river bank"  → [0.3, 0.5, ...]  ← same vector
  "bank account"→ [0.3, 0.5, ...]  ← same vector
  (polysemy ignored)

Contextual embeddings (BERT):
  "river bank"  → [0.8, -0.2, ...]  ← different vector per context
  "bank account"→ [-0.3, 0.7, ...]  ← context-aware!

BERT (Bidirectional Encoder Representations from Transformers):
  → Reads entire sentence bidirectionally
  → Each token's embedding depends on ALL surrounding tokens
  → Pre-trained on masked language modeling + next sentence prediction
  → Fine-tuned on downstream tasks

ELMo: LSTM-based contextual embeddings
GPT:  Unidirectional transformer
BERT: Bidirectional transformer (most popular)
```

---

## 📐 Measuring Word Similarity

```
Cosine Similarity (most common):
  sim(u, v) = (u · v) / (||u|| × ||v||)

  Range: [−1, 1]
  1 → identical direction (semantically similar)
  0 → orthogonal (unrelated)
  −1 → opposite direction (antonyms)

Euclidean Distance:
  d(u, v) = ||u − v||
  → Sensitive to vector magnitude (less preferred for embeddings)

Nearest Neighbor Search:
  Find the k most similar words to a given word vector
  from gensim.models import Word2Vec
  model.wv.most_similar('king', topn=5)
```

---

## 🛠️ Using Pre-trained Embeddings in Practice

```python
# ── gensim Word2Vec ────────────────────────────────────────────────────────
from gensim.models import Word2Vec, KeyedVectors

# Train from scratch
sentences = [['the', 'quick', 'brown', 'fox'], ['she', 'loves', 'nlp']]
model = Word2Vec(sentences, vector_size=100, window=5,
                  min_count=1, workers=4, sg=1)  # sg=1: skip-gram
model.wv['fox']                   # 100-dim vector
model.wv.most_similar('fox')      # nearest neighbors

# Load pre-trained Google News vectors
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin.gz', binary=True)

# ── FastText (gensim) ──────────────────────────────────────────────────────
from gensim.models import FastText
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, sg=1)
ft_model.wv['unseen_word']        # works for OOV!

# ── Hugging Face BERT embeddings ──────────────────────────────────────────
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertModel.from_pretrained('bert-base-uncased')

text   = "The bank was flooded by the river."
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

# CLS token embedding (sentence-level)
cls_embedding    = outputs.last_hidden_state[:, 0, :]   # shape: (1, 768)

# All token embeddings (word-level, contextual)
token_embeddings = outputs.last_hidden_state             # shape: (1, seq_len, 768)

# Mean pooling (common sentence embedding)
mean_embedding   = token_embeddings.mean(dim=1)          # shape: (1, 768)
```

---

## 📊 Pre-trained Embedding Comparison

| Model | Dim | Vocab | Context-Aware | OOV | Best For |
|-------|:---:|:-----:|:-------------:|:---:|---------|
| Word2Vec (CBOW) | 100–300 | Fixed | ❌ | ❌ | General NLP |
| Word2Vec (Skip-gram) | 100–300 | Fixed | ❌ | ❌ | Rare words |
| GloVe | 50–300 | Fixed | ❌ | ❌ | Large corpora |
| FastText | 100–300 | Open | ❌ | ✅ | Morphology, OOV |
| ELMo | 1024 | Open | ✅ | ✅ | Transfer learning |
| BERT | 768 | Subword | ✅ | ✅ | Classification, QA |
| RoBERTa | 768 | Subword | ✅ | ✅ | Better BERT |
| sentence-BERT | 768 | Subword | ✅ | ✅ | Semantic similarity |

---

## 🎛️ Embedding Layer in Neural Networks

```python
import torch.nn as nn

# Trainable embedding layer (learns from scratch)
embedding = nn.Embedding(
    num_embeddings=vocab_size,   # vocabulary size
    embedding_dim=128,           # embedding dimension
    padding_idx=0,               # index of <PAD> token
)

# Initialize with pre-trained weights
embedding.weight.data.copy_(torch.tensor(pretrained_vectors))
embedding.weight.requires_grad = False  # freeze (fine-tune = True)

# Input: token indices → Output: dense vectors
token_ids = torch.tensor([[1, 5, 23, 7]])   # (batch=1, seq_len=4)
embedded  = embedding(token_ids)             # (1, 4, 128)
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Using one-hot for large vocab | Extremely high-dimensional, no semantics | Use embeddings instead |
| Ignoring OOV words | Drop or zero-vector for unseen words | Use FastText or subword models |
| Static embedding for polysemous words | "bank" has one vector | Use BERT/contextual embeddings |
| Training embeddings on small corpus | Poor quality — not enough context | Use pre-trained embeddings |
| Not normalizing embeddings | Cosine similarity assumes unit vectors | Normalize before similarity |
| Fine-tuning all BERT layers on tiny dataset | Overfits | Freeze lower layers, fine-tune top 2–3 |

---

## 🔗 Related Topics

- `10_NLP_Basics/text_preprocessing.md` — Tokenization, stopwords, stemming
- `10_NLP_Basics/text_classification.ipynb` — Using embeddings for classification
- `10_NLP_Basics/sentiment_analysis.ipynb` — BERT fine-tuning for sentiment
- `04_Unsupervised_Learning/PCA` — Visualize embeddings in 2D
- `04_Unsupervised_Learning/tSNE` — t-SNE visualization of word vectors

---

## 📚 References

- Word2Vec (Mikolov et al., 2013): [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
- GloVe (Pennington et al., 2014): [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- FastText (Bojanowski et al., 2017): [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
- BERT (Devlin et al., 2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- Gensim Documentation: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
- Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
