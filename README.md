<h1 align="center">🚀 DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization</h1>

The success of **DeepSeek-R1** has spotlighted **GRPO (Group Relative Policy Optimization)** as a key reinforcement learning method for large reasoning models.

**But how can we improve GRPO? What are its key limitations?**

We analyzed GRPO under a binary reward setting and uncovered two core insights:

* ⚠️ GRPO suffers from **question-level difficulty bias**
* 🔍 GRPO has a surprising connection to **discriminative learning** techniques, particularly AUC maximization

---

### 💡 Introducing **DisCO** — *Discriminative Constrained Optimization*

**DisCO** is a new RL framework grounded in **discriminative learning**. It trains models by **increasing scores for positive answers while decreasing those for negatives**, enabling:

* ⚡ Faster convergence
* 🔒 More stable optimization
* 🔁 Longer-lasting training dynamics for large reasoning models

---

### 🔍 Why DisCO?

* ❌ **No more difficulty bias** – replaces group-relative rewards with discriminative scoring
* 🔄 **No clip operations** – uses non-clipping scoring functions for smoother learning
* 📉 **Stable training** – via simple constrained optimization to keep KL divergence in check
* ⚖️ **Handles sparse rewards** – robust to imbalanced data with more negatives than positives

---

### 📈 Results

On six math reasoning benchmarks with a 1.5B model, **DisCO outperforms GRPO and its variants**:

* **+7% vs GRPO**
* **+6% vs DAPO**

**Table of Contents**
- [Experimental Results](#experimental-results)
- [Getting Started](#getting-started)
    - [Environment Setup](#environment-setup)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Citing DRRho](#citing-drrho)


