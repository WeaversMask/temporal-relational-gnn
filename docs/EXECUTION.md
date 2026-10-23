# Execution contract and equivalence argument

Let a model have L layers. For each node type, a sampled occurrence batch is
ordered as seeds, first-hop occurrences, ..., L-th-hop occurrences. Edges are
grouped in sampling-hop order. Sampling is directed and disjoint by query time.
Bidirectional and induced sampled subgraphs are outside this prototype's scope.

At layer i (zero indexed), blocks consume the prefix through hop L-i and produce
only the prefix through hop L-i-1. Keep the first sum(edge_counts[:L-i]) edges
for each relation. Source and destination feature matrices have separate sizes;
this applies even when a relation connects the same table to itself.

For every retained destination, all sampled incoming edges needed by that layer
remain. Its neighbor mean, root transform, and bias are therefore unchanged.
Normalization is over features within each node (RelBench uses `mode="node"`),
and ReLU is node-local. Inductively, seed outputs agree after all layers, subject
to floating-point reduction differences. Batch/graph normalization, dropout,
attention, and different sampling semantics require separate arguments.

An empty relation remains in the convolution dictionary and edge dictionary.
Even without messages, SAGEConv's root transform and bias may affect outputs.
Dropping that key changes the model. Nodes with no retained outputs still have
zero-sized tensors so that subsequent relation dispatch preserves semantics.

Current tests compare seed embeddings, a seed-only loss, input and parameter
gradients, and one Adam step. The empty-relation test also demonstrates that
dropping its key changes retained outputs. Tests cover 1-3 layers, zero fanout,
isolated seeds, repeated global IDs, and same-table relations.

The synthetic generator is an adversarial fixture, not a replacement for the
actual temporal sampler. Correctness on saved RelBench batches, full-model logits,
and the final pipeline remains a separate milestone.
