# open-jev

an open-source, from-first-principles reconstruction of the ideas behind
[TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
written in pytorch

> [!IMPORTANT]
> this is an unofficial research implementation with random weights. it is not
> the production Jev model, does not reproduce TypeSafe AI's training data or
> weights, and makes no claim of matching their published results.

## the idea

Jev is presented as a system one model: unstructured program state goes in,
typed probabilistic decisions come out.

Instead of generating text one token at a time, this implementation encodes the
state once and answers every question through small typed readout heads.

```text
json-like state ──> bidirectional state encoder ──> shared state cache
                                                        │
                         question slots ── cross-attend ┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
                noul          choice         score
               p(true)      p(options)     p(levels)
```

questions are isolated from one another and folded into the batch dimension.
they may attend to the shared state, but not to other questions. the result is a
single parallel forward pass with outputs constrained by their declared types.

## architecture

- a bidirectional transformer encodes nested state without a causal mask
- structural path embeddings preserve where values occur in dictionaries and arrays
- learned query slots cross-attend into one shared, cacheable state representation
- `Noul` returns the probability that a statement is true
- `Choice` softmaxes only over the options supplied at runtime
- `Score` uses ordinal thresholds and returns a distribution plus its expectation
- an evidential head models epistemic confidence separately from class probability
- `RLCDLoss` combines soft-target NLL, Brier score, consistency, confidence, and ECE terms

the implementation is intentionally small and legible. the included tokenizer
is a deterministic hash tokenizer, suitable for running the architecture but
not for training a useful model.

## install

```bash
git clone https://github.com/kyegomez/open-jev
cd open-jev
python -m pip install "torch>=2.0"
```

python 3.10 or newer is recommended.

## usage

```python
import torch

from open_jev.main import Choice, Jev, JevConfig, Noul, Score

model = Jev(
    JevConfig(
        vocab_size=4096,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_state_layers=3,
        n_readout_layers=4,
    )
).eval()

state = {
    "customer": {"tier": "enterprise", "tenure_months": 34},
    "message": "This is the third duplicate charge. Please fix it.",
    "policy": "Duplicate charges are refundable within 60 days.",
}

questions = [
    Noul("The customer is requesting a refund.", key="wants_refund"),
    Choice(
        "Which team should handle this?",
        options=["billing", "technical", "account"],
        key="route",
    ),
    Score(
        "How frustrated is the customer?",
        labels=["calm", "annoyed", "frustrated", "very frustrated"],
        key="frustration",
    ),
]

with torch.no_grad():
    answers = model([state], questions)[0]

for answer in answers:
    print(answer)
```

the model is untrained, so the values from this example are random. the useful
guarantee is structural: a `Choice` answer can only be one of the options that
were declared.

to run the complete demo, including one calibration-oriented training step:

```bash
python open_jev/main.py
```

## training

`RLCDLoss` expects probability distributions rather than one-hot labels. this
allows disagreement and ambiguity to remain visible instead of forcing every
example toward certainty.

```python
from open_jev.main import RLCDLoss

targets = [
    torch.tensor([[0.08, 0.92]]),
    torch.tensor([[0.80, 0.05, 0.15]]),
    torch.tensor([[0.02, 0.10, 0.48, 0.40]]),
]

loss_fn = RLCDLoss()
loss = loss_fn(model, [state], questions, targets)
loss.backward()
```

for consistency training, pass semantically equivalent states through
`augmented_states`—for example, paraphrases or shuffled dictionary keys.

## notes

this repository explores an architecture inferred from public material. details
such as the encoder design, query slots, ordinal head, confidence objective, and
training recipe are hypotheses, not disclosed details of TypeSafe AI's system.
see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full reasoning and
tradeoffs behind the reconstruction.

in particular, typed output prevents schema violations; it does not guarantee
that a prediction is correct. calibration must also be measured again under
distribution shift before deploying a model in an unattended workflow.

## todo

- [ ] replace the hash tokenizer with a trained tokenizer
- [ ] add a real pretraining and distillation pipeline
- [ ] benchmark state-cache reuse and multi-question scaling
- [ ] evaluate calibration, consistency, and distribution shift
- [ ] publish trained checkpoints

## acknowledgements

appreciation to TypeSafe AI for introducing
[System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and for sharing enough public detail to inspire independent experimentation.

## license

Apache 2.0
