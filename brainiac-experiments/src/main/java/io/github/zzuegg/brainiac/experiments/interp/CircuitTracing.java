package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.AttributionGraph;
import io.github.zzuegg.brainiac.experimental.nn.CrossLayerTranscoder;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Circuit tracing on a planted two-hop task. A residual stack of two MLPs reads a token
 * {@code A} (one of 12) and a distractor, and must output {@code C = perm(B)} where the
 * intermediate {@code B = A mod 4} is planted in the residual stream after the first MLP by
 * an auxiliary readout. A {@link CrossLayerTranscoder} replaces the MLPs; for each input
 * an {@link AttributionGraph} for the target logit is built and pruned to 80% of the
 * influence. The test of the method: is the strongest first-layer feature left in the
 * pruned graph a <em>B feature</em> (one that fires for exactly the three inputs sharing
 * that {@code B}), rather than a feature of {@code A} itself? Per-layer transcoders with
 * the same size and sparsity are the baseline, both for this recovery and for the fidelity
 * of the replacement model (top-1 agreement with the real model when the MLPs are
 * replaced).
 *
 * <p>Ameisen et al. 2025, "Circuit Tracing", <a href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html">Transformer
 * Circuits</a>; Lindsey et al. 2025, "On the Biology of a Large Language Model" (two-hop
 * reasoning), <a href="https://transformer-circuits.pub/2025/attribution-graphs/biology.html">Transformer
 * Circuits</a>.
 */
public final class CircuitTracing {

    private static final int A = 12;
    private static final int D = 4;
    private static final int DIM = 16;
    private static final int HIDDEN = 32;
    private static final int[] PERM = {2, 0, 3, 1};

    /**
     * The outcome for one kind of transcoder.
     *
     * @param transcoder "cross-layer" or "per-layer"
     * @param explainedVariance the share of MLP-output variance the transcoder explains
     * @param replacementAgreement top-1 agreement of the replacement model with the real one
     * @param bFeatureRecovery the share of inputs whose pruned graph's strongest first-layer
     *     feature is a B feature
     * @param meanPrunedNodes the mean node count of the pruned graphs
     * @param completeness the largest relative gap between the target logit rebuilt from the
     *     unpruned graph's edges and the real one (0 means the graph is exact)
     */
    public record Outcome(String transcoder, double explainedVariance, double replacementAgreement,
            double bFeatureRecovery, double meanPrunedNodes, double completeness) {
    }

    /** The planted two-hop residual MLP model. */
    private static final class TwoHopModel {
        final Var embed;
        final Var[] w1 = new Var[2];
        final Var[] b1 = new Var[2];
        final Var[] w2 = new Var[2];
        final Var[] b2 = new Var[2];
        final Var readout;
        final Var readoutBias;
        final Var aux;

        TwoHopModel(SplittableRandom r) {
            embed = Var.param(Tensor.normal(Shape.of(A + D, DIM), 0, 1, r));
            for (int l = 0; l < 2; l++) {
                w1[l] = Var.param(Tensor.normal(Shape.of(DIM, HIDDEN), 0, (float) (1 / Math.sqrt(DIM)), r));
                b1[l] = Var.param(Tensor.zeros(Shape.of(HIDDEN)));
                w2[l] = Var.param(Tensor.normal(Shape.of(HIDDEN, DIM), 0, (float) (1 / Math.sqrt(HIDDEN)), r));
                b2[l] = Var.param(Tensor.zeros(Shape.of(DIM)));
            }
            readout = Var.param(Tensor.normal(Shape.of(DIM, 4), 0, 0.25f, r));
            readoutBias = Var.param(Tensor.zeros(Shape.of(4)));
            aux = Var.param(Tensor.normal(Shape.of(DIM, 4), 0, 0.25f, r));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, readout, readoutBias, aux));
            for (int l = 0; l < 2; l++) {
                p.addAll(List.of(w1[l], b1[l], w2[l], b2[l]));
            }
            return p;
        }

        Var mlp(int l, Var h) {
            return h.matmul(w1[l]).add(b1[l]).relu().matmul(w2[l]).add(b2[l]);
        }

        /** @return {h0, m0, h1, m1, h2} */
        Var[] run(Tensor x) {
            Var h0 = Var.of(x).matmul(embed);
            Var m0 = mlp(0, h0);
            Var h1 = h0.add(m0);
            Var m1 = mlp(1, h1);
            return new Var[] {h0, m0, h1, m1, h1.add(m1)};
        }
    }

    private CircuitTracing() {
    }

    private static int b(int a) {
        return a % 4;
    }

    private static int c(int a) {
        return PERM[b(a)];
    }

    private static float[] input(int a, int d, double noise, SplittableRandom r) {
        float[] x = new float[A + D];
        x[a] = 1;
        x[A + d] = 1;
        for (int i = 0; i < x.length && noise > 0; i++) {
            x[i] += (float) (r.nextGaussian() * noise);
        }
        return x;
    }

    /**
     * Runs the experiment.
     *
     * @param cltSteps transcoder training steps
     * @param seed the seed
     * @return the cross-layer outcome, then the per-layer baseline
     */
    public static List<Outcome> run(int cltSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        TwoHopModel model = new TwoHopModel(r);
        Optimizer opt = Optimizer.adam(0.01f);
        int n = A * D;
        float[][] clean = new float[n][];
        float[] labelsC = new float[n];
        float[] labelsB = new float[n];
        for (int i = 0; i < n; i++) {
            clean[i] = input(i % A, i / A, 0, r);
            labelsC[i] = c(i % A);
            labelsB[i] = b(i % A);
        }
        for (int step = 0; step < 400; step++) {
            float[][] noisy = new float[n][];
            for (int i = 0; i < n; i++) {
                noisy[i] = input(i % A, i / A, 0.05, r);
            }
            Var[] run = model.run(Tensor.of(noisy));
            Var loss = Functional.crossEntropy(run[4].matmul(model.readout).add(model.readoutBias),
                    Tensor.of(Shape.of(n), labelsC))
                    .add(Functional.crossEntropy(run[2].matmul(model.aux), Tensor.of(Shape.of(n), labelsB)));
            loss.backward();
            opt.step(model.parameters());
        }
        // activations for the transcoders: noisy copies of every input
        int samples = 1500;
        float[][][] x = new float[2][samples][];
        float[][][] y = new float[2][samples][];
        float[][] inputs = new float[samples][];
        for (int s = 0; s < samples; s++) {
            inputs[s] = input(r.nextInt(A), r.nextInt(D), 0.1, r);
        }
        Var[] act = model.run(Tensor.of(inputs));
        for (int l = 0; l < 2; l++) {
            x[l] = rows(act[2 * l]);
            y[l] = rows(act[2 * l + 1]);
        }
        List<Outcome> out = new ArrayList<>();
        for (boolean cross : new boolean[] {true, false}) {
            CrossLayerTranscoder clt = cross ? CrossLayerTranscoder.of(2, DIM, DIM, 16, 2, seed + 3)
                    : CrossLayerTranscoder.perLayer(2, DIM, DIM, 16, 2, seed + 3);
            clt.train(x, y, cltSteps, 128, 3e-3f, seed + 4);
            out.add(evaluate(cross ? "cross-layer" : "per-layer", clt, model, x, y, inputs, clean));
        }
        return out;
    }

    private static Outcome evaluate(String name, CrossLayerTranscoder clt, TwoHopModel model, float[][][] x,
            float[][][] y, float[][] inputs, float[][] clean) {
        // replacement model: MLPs replaced by the transcoder, features computed on its own stream
        Var[] real = model.run(Tensor.of(inputs));
        int[] realTop = argmax(real[4].matmul(model.readout).add(model.readoutBias));
        Var h0 = real[0];
        List<Var> codes = new ArrayList<>();
        codes.add(clt.encode(0, h0));
        Var h1 = h0.add(clt.decode(0, codes));
        codes.add(clt.encode(1, h1));
        Var h2 = h1.add(clt.decode(1, codes));
        int[] replTop = argmax(h2.matmul(model.readout).add(model.readoutBias));
        int agree = 0;
        for (int i = 0; i < realTop.length; i++) {
            agree += realTop[i] == replTop[i] ? 1 : 0;
        }
        // which first-layer features fire for which A (distractor 0, no noise)
        float[][] firstCodes = rows(clt.encode(0, model.run(Tensor.of(cleanRows(clean, 0)))[0]));
        float[] readoutT = model.readout.value().to(DType.FLOAT32).toArray();
        float[] embed = model.embed.value().to(DType.FLOAT32).toArray();
        int recovered = 0;
        double nodes = 0;
        double gap = 0;
        for (int a = 0; a < A; a++) {
            Var[] run = model.run(Tensor.of(new float[][] {clean[a]}));
            float[] u = new float[DIM];
            for (int d = 0; d < DIM; d++) {
                u[d] = readoutT[d * 4 + c(a)];
            }
            float[][] contributions = {slice(embed, a), slice(embed, A)};
            AttributionGraph g = AttributionGraph.ofResidualMlp(clt, contributions, new String[] {"A=" + a, "D=0"},
                    new float[][] {rows(run[0])[0], rows(run[2])[0]}, new float[][] {rows(run[1])[0], rows(run[3])[0]},
                    u, "C=" + c(a));
            gap = Math.max(gap, Math.abs(g.targetFromEdges() - g.target().activation())
                    / Math.max(1e-6, Math.abs(g.target().activation())));
            AttributionGraph pruned = g.prune(0.8);
            nodes += pruned.nodes().size() / (double) A;
            AttributionGraph.Node top = pruned.strongest(AttributionGraph.Kind.FEATURE, 0);
            if (top != null && isBFeature(firstCodes, top.index(), b(a))) {
                recovered++;
            }
        }
        return new Outcome(name, clt.explainedVariance(x, y), agree / (double) realTop.length, recovered / (double) A,
                nodes, gap);
    }

    /** Fires for every A with this B and for at most one other A. */
    private static boolean isBFeature(float[][] codes, int feature, int bValue) {
        int hits = 0;
        int others = 0;
        for (int a = 0; a < A; a++) {
            boolean on = codes[a][feature] > 0;
            if (b(a) == bValue) {
                hits += on ? 1 : 0;
            } else {
                others += on ? 1 : 0;
            }
        }
        return hits == 3 && others <= 1;
    }

    private static float[][] cleanRows(float[][] clean, int distractor) {
        float[][] out = new float[A][];
        for (int a = 0; a < A; a++) {
            out[a] = clean[distractor * A + a];
        }
        return out;
    }

    private static float[] slice(float[] embed, int row) {
        float[] out = new float[DIM];
        System.arraycopy(embed, row * DIM, out, 0, DIM);
        return out;
    }

    private static float[][] rows(Var v) {
        int d = v.shape().dim(-1);
        float[] flat = v.value().to(DType.FLOAT32).toArray();
        float[][] out = new float[flat.length / d][d];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(flat, i * d, out[i], 0, d);
        }
        return out;
    }

    private static int[] argmax(Var logits) {
        float[][] p = rows(logits);
        int[] out = new int[p.length];
        for (int i = 0; i < p.length; i++) {
            for (int j = 1; j < p[i].length; j++) {
                if (p[i][j] > p[i][out[i]]) {
                    out[i] = j;
                }
            }
        }
        return out;
    }
}
