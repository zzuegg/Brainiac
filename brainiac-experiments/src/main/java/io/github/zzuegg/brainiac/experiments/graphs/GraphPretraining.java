package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.graph.GraphSelfSupervision;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Self-supervised graph pretraining (wave 7, 7.1.11): a two-layer GCN encoder is
 * pretrained without labels on a stochastic block model by Deep Graph Infomax, GraphCL or
 * GraphMAE ({@link GraphSelfSupervision}), frozen, and read by a linear probe trained on 1%
 * or 5% of the labels. Baselines: the same probe on a randomly initialised encoder, on a
 * graph-autoencoder encoder (link reconstruction, as in {@link GnnBenchmarks}), and the GCN
 * trained end to end on the same few labels.
 */
public final class GraphPretraining {

    private GraphPretraining() {
    }

    private static final class Encoder implements GraphSelfSupervision.Encoder {
        private final GcnLayer l1;
        private final GcnLayer l2;

        Encoder(int in, int width, SplittableRandom r) {
            l1 = new GcnLayer(in, width, Activation.RELU, r);
            l2 = new GcnLayer(width, width, Activation.IDENTITY, r);
        }

        @Override
        public Var encode(Var x, Tensor adjacency, boolean training) {
            Var adj = Var.of(adjacency);
            return l2.forward(List.of(l1.forward(List.of(x, adj), training), adj), training);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(l1.parameters());
            p.addAll(l2.parameters());
            return p;
        }
    }

    private static void pretrainAutoencoder(Encoder e, Var x, Tensor adj, int steps) {
        int n = adj.shape().dim(0);
        float density = adj.sum().item() / ((float) n * n);
        float posWeight = (1 - density) / Math.max(density, 1e-6f);
        Tensor weight = adj.mul(posWeight - 1).add(1f);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Var z = e.encode(x, adj, true);
            Var logits = z.matmul(z.transpose());
            logits.softplus().sub(logits.mul(adj)).mul(weight).mean().backward();
            opt.step(e.parameters());
        }
    }

    private static float[][] embed(Encoder e, Var x, Tensor adj) {
        Tensor z = Var.noGrad(() -> e.encode(x, adj, false)).value();
        int n = z.shape().dim(0);
        int d = z.shape().dim(1);
        float[] v = z.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * Linear-probe accuracy after each pretraining objective.
     *
     * @param labelRates label rates of the probe
     * @param steps pretraining (and supervised training) steps
     * @param seeds label masks averaged
     * @param seed the seed
     * @return "method @ rate" to mean probe accuracy on unlabelled nodes
     */
    public static Map<String, Double> linearProbe(double[] labelRates, int steps, int seeds, long seed) {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 50, 0.12, 0.03, 3.0, seed);
        Tensor adj = g.adjacencyTensor();
        Var x = Var.of(g.featureTensor());
        int width = 16;
        Map<String, float[][]> embeddings = new LinkedHashMap<>();
        embeddings.put("random encoder", embed(new Encoder(3, width, new SplittableRandom(seed)), x, adj));
        Encoder gae = new Encoder(3, width, new SplittableRandom(seed));
        pretrainAutoencoder(gae, x, adj, steps);
        embeddings.put("GAE", embed(gae, x, adj));
        for (GraphSelfSupervision.Method method : GraphSelfSupervision.Method.values()) {
            SplittableRandom r = new SplittableRandom(seed);
            Encoder e = new Encoder(3, width, r);
            GraphSelfSupervision ssl = switch (method) {
                case DGI -> GraphSelfSupervision.dgi(e, width, r);
                case GRAPH_CL -> GraphSelfSupervision.graphCl(e, width, 0.2f, r);
                case GRAPH_MAE -> GraphSelfSupervision.graphMae(e, 3, width, 0.5f, r);
            };
            Optimizer opt = Optimizer.adam(0.01f);
            for (int s = 0; s < steps; s++) {
                ssl.loss(x, adj).backward();
                opt.step(ssl.parameters());
            }
            embeddings.put(method.name(), embed(e, x, adj));
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (double rate : labelRates) {
            for (Map.Entry<String, float[][]> e : embeddings.entrySet()) {
                out.put(e.getKey() + " @ " + rate, GraphTraining.round(GraphTraining.overSeeds(seeds, seed, sd -> {
                    boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, rate, sd);
                    return GraphTraining.probe(e.getValue(), g.nodeLabels(), train, 3, sd);
                })));
            }
            out.put("supervised GCN @ " + rate, GraphTraining.round(GraphTraining.overSeeds(seeds, seed, sd -> {
                boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, rate, sd);
                return GraphTraining.trainNodes(GraphTraining.gcn(3, width, 3, new SplittableRandom(sd)), g, adj, train,
                        3, steps, 0.02f);
            })));
        }
        return out;
    }
}
