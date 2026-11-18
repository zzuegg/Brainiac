/**
 * Graph learning blocks beyond the basic GNN layers of {@code experimental.nn} (GCN, GAT,
 * GraphSAGE, GIN, which stay there and are reused here): positional encodings and the
 * GraphGPS transformer layer, decoupled propagation (SGC, APPNP), deep GNNs (GCNII,
 * PairNorm, DropEdge), heterophily (H2GCN, GPR-GNN, sheaf diffusion), continuous diffusion
 * (GRAND), rewiring against over-squashing, expressivity beyond 1-WL (random node features,
 * subgraph GNNs, PNA), hierarchical pooling (DiffPool, top-k/SAGPool), self-supervision
 * (DGI, GraphCL, GraphMAE), random-walk embeddings, knowledge-graph embeddings, neural
 * algorithmic reasoning, NeuroSAT, discrete graph diffusion, random-walk sequence layers
 * and label propagation with Correct and Smooth. Layers implement the core
 * {@code MultiInputLayer} with node features and a dense adjacency as inputs, like the
 * existing graph layers.
 */
package io.github.zzuegg.brainiac.experimental.graph;
