package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.interp.AlignmentSearch;
import io.github.zzuegg.brainiac.experimental.interp.ConceptErasure;
import io.github.zzuegg.brainiac.experimental.interp.ModelStitcher;
import io.github.zzuegg.brainiac.experimental.nn.LinearProbe;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 items 9.4.1 and 9.4.2: finding mechanisms with known ground truth.
 *
 * <p><b>Edges</b> ({@link #edges}): a residual network with a planted circuit
 * ({@link io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.PlantedCircuit}: 14
 * nodes, 10 planted edges among the rest, which carry weights 50 times smaller, and one
 * planted edge whose target saturates). ACDC, edge attribution patching (EAP) and EAP-IG are
 * scored by the ROC-AUC of their edge rankings against the planted circuit and by their
 * cost; node-level activation patching and attribution patching (AtP) are the baselines.
 * The clean inputs are random signs and the corrupted inputs flip every sign.
 *
 * <p><b>Alignment</b> ({@link #alignment}): an MLP (40-48-48-10) learns the tens digit of
 * the sum of two two-digit numbers, which needs the carry from the ones digits. Distributed
 * alignment search ({@link AlignmentSearch}) looks for a k-dimensional subspace of the
 * second hidden layer that holds the carry, scored by interchange-intervention accuracy
 * (IIA) on base/source pairs whose carries differ, so the counterfactual tens digit differs
 * from the base's. Baselines: a {@link LinearProbe} direction for the carry, the top
 * principal components of the layer, and the best single neuron.
 */
public final class CircuitDiscovery {

    private CircuitDiscovery() {
    }

    // ------------------------------------------------------------------ 9.4.1 edges

    /**
     * One edge-discovery method's scores.
     *
     * @param method the method
     * @param auc ROC-AUC of its edge ranking against the planted circuit
     * @param saturatedRank the rank (0 = best) of the saturating planted edge
     * @param forwardPasses forward passes used
     * @param millis wall time
     */
    public record EdgeScore(String method, double auc, int saturatedRank, int forwardPasses, double millis) {
    }

    /**
     * The edge-discovery comparison.
     *
     * @param seed the seed
     * @return ACDC, EAP, EAP-IG, node activation patching and node AtP, plus the faithfulness
     *     of ACDC's circuit as a final pseudo-row ({@code auc} holds faithfulness,
     *     {@code saturatedRank} the circuit size)
     */
    public static List<EdgeScore> edges(long seed) {
        var model = new io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.PlantedCircuit(6, 6, 3, 4, 2,
                0.03, seed);
        Tensor[] in = model.inputs(64, seed + 1);
        boolean[] truth = model.circuit();
        int saturating = saturatingEdge(model);
        List<EdgeScore> rows = new ArrayList<>();
        long t0 = System.nanoTime();
        var acdc = io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.acdc(model, in[0], in[1], 1e-3);
        double acdcMs = (System.nanoTime() - t0) / 1e6;
        rows.add(row("ACDC", acdc.scores(), truth, saturating, acdc.forwardPasses(), acdcMs));
        t0 = System.nanoTime();
        double[] eap = io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.eap(model, in[0], in[1]);
        rows.add(row("EAP", eap, truth, saturating, 2, (System.nanoTime() - t0) / 1e6));
        t0 = System.nanoTime();
        double[] ig = io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.eapIg(model, in[0], in[1], 16);
        rows.add(row("EAP-IG (16 steps)", ig, truth, saturating, 2 * 16, (System.nanoTime() - t0) / 1e6));
        t0 = System.nanoTime();
        double[] node = io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.nodePatching(model, in[0],
                in[1]);
        rows.add(row("node activation patching", node, truth, saturating, model.nodes(),
                (System.nanoTime() - t0) / 1e6));
        t0 = System.nanoTime();
        double[] atp = io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.nodeAttribution(model, in[0],
                in[1]);
        rows.add(row("node AtP", atp, truth, saturating, 2, (System.nanoTime() - t0) / 1e6));
        int size = 0;
        for (boolean k : acdc.kept()) {
            size += k ? 1 : 0;
        }
        double faithful = io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.faithfulness(model, in[0],
                in[1], acdc.kept());
        rows.add(new EdgeScore("ACDC circuit faithfulness", faithful, size, 0, 0));
        return rows;
    }

    private static int saturatingEdge(io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.PlantedCircuit m) {
        // the planted edge from the embedding into the last node of the first layer
        int target = 1 + 4 - 1;
        int[][] edges = m.edges();
        for (int i = 0; i < edges.length; i++) {
            if (edges[i][0] == 0 && edges[i][1] == target) {
                return i;
            }
        }
        return -1;
    }

    private static EdgeScore row(String name, double[] scores, boolean[] truth, int saturating, int passes,
            double ms) {
        int rank = 0;
        for (double s : scores) {
            rank += s > scores[saturating] ? 1 : 0;
        }
        return new EdgeScore(name, io.github.zzuegg.brainiac.experimental.interp.CircuitDiscovery.rocAuc(scores, truth),
                rank, passes, ms);
    }

    // ------------------------------------------------------------------ 9.4.2 alignment

    private static final int WIDTH = 48;

    private static float[] encode(int a, int b) {
        float[] x = new float[40];
        x[a / 10] = 1;
        x[10 + a % 10] = 1;
        x[20 + b / 10] = 1;
        x[30 + b % 10] = 1;
        return x;
    }

    private static int tens(int a, int b, int carry) {
        return (a / 10 + b / 10 + carry) % 10;
    }

    private static int carry(int a, int b) {
        return a % 10 + b % 10 >= 10 ? 1 : 0;
    }

    private static Tensor rows(List<float[]> xs) {
        float[] f = new float[xs.size() * 40];
        for (int i = 0; i < xs.size(); i++) {
            System.arraycopy(xs.get(i), 0, f, i * 40, 40);
        }
        return Tensor.of(Shape.of(xs.size(), 40), f);
    }

    private static Tensor ids(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    /**
     * Trains the adder.
     *
     * @param steps Adam steps of 256 random sums
     * @param seed the seed
     * @return the model
     */
    public static Model trainAdder(int steps, long seed) {
        Model m = Model.sequential().input(40).dense(WIDTH, Activation.RELU).dense(WIDTH, Activation.RELU)
                .dense(10, Activation.IDENTITY).seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            List<float[]> xs = new ArrayList<>();
            int[] y = new int[256];
            for (int i = 0; i < 256; i++) {
                int a = r.nextInt(100);
                int b = r.nextInt(100);
                xs.add(encode(a, b));
                y[i] = tens(a, b, carry(a, b));
            }
            Functional.crossEntropy(m.forward(Var.of(rows(xs)), true), ids(y)).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static double accuracy(Model m, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> xs = new ArrayList<>();
        int[] y = new int[1000];
        for (int i = 0; i < y.length; i++) {
            int a = r.nextInt(100);
            int b = r.nextInt(100);
            xs.add(encode(a, b));
            y[i] = tens(a, b, carry(a, b));
        }
        float[] p = Var.noGrad(() -> m.forward(Var.of(rows(xs)))).value().argmax(1).toArray();
        int hit = 0;
        for (int i = 0; i < y.length; i++) {
            hit += p[i] == y[i] ? 1 : 0;
        }
        return (double) hit / y.length;
    }

    /**
     * Interchange pairs whose carries differ; the label is the tens digit with the base's
     * tens digits and the source's carry.
     *
     * @param n pairs
     * @param seed the seed
     * @return the batch
     */
    public static AlignmentSearch.Interchange interchange(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> base = new ArrayList<>();
        List<float[]> source = new ArrayList<>();
        int[] y = new int[n];
        int i = 0;
        while (i < n) {
            int a = r.nextInt(100);
            int b = r.nextInt(100);
            int a2 = r.nextInt(100);
            int b2 = r.nextInt(100);
            if (carry(a, b) == carry(a2, b2)) {
                continue;
            }
            base.add(encode(a, b));
            source.add(encode(a2, b2));
            y[i++] = tens(a, b, carry(a2, b2));
        }
        return new AlignmentSearch.Interchange(rows(base), rows(source), ids(y));
    }

    /**
     * Alignment results.
     *
     * @param taskAccuracy the adder's accuracy
     * @param dasK1 DAS interchange accuracy with a 1-dimensional subspace
     * @param dasK2 with 2 dimensions
     * @param probe the carry probe's direction
     * @param probeAccuracy how well that probe decodes the carry (it is decodable)
     * @param pca1 the top principal component
     * @param pca2 the top two principal components
     * @param bestNeuron the best single neuron
     */
    public record AlignmentScores(double taskAccuracy, double dasK1, double dasK2, double probe, double probeAccuracy,
            double pca1, double pca2, double bestNeuron) {
    }

    /**
     * The alignment comparison on the second hidden layer.
     *
     * @param trainSteps adder training steps
     * @param dasSteps DAS training steps per subspace size
     * @param seed the seed
     * @return the scores
     */
    public static AlignmentScores alignment(int trainSteps, int dasSteps, long seed) {
        Model m = trainAdder(trainSteps, seed);
        double acc = accuracy(m, seed + 2);
        int step = 1;
        AlignmentSearch.Interchange test = interchange(400, seed + 3);
        double[] das = new double[2];
        for (int k = 1; k <= 2; k++) {
            AlignmentSearch search = new AlignmentSearch(m, step, WIDTH, k, WIDTH, seed + 10 + k);
            long base = seed + 100L * k;
            search.train(i -> interchange(128, base + i), dasSteps, 1e-2f);
            das[k - 1] = search.interchangeAccuracy(test);
        }
        // the activations of the layer, with carry labels
        SplittableRandom r = new SplittableRandom(seed + 4);
        List<float[]> xs = new ArrayList<>();
        int[] carry = new int[600];
        for (int i = 0; i < carry.length; i++) {
            int a = r.nextInt(100);
            int b = r.nextInt(100);
            xs.add(encode(a, b));
            carry[i] = carry(a, b);
        }
        Tensor acts = ModelStitcher.activation(m, step, rows(xs));
        float[][] h = new float[carry.length][WIDTH];
        float[] flat = acts.toArray();
        for (int i = 0; i < carry.length; i++) {
            System.arraycopy(flat, i * WIDTH, h[i], 0, WIDTH);
        }
        LinearProbe probe = LinearProbe.fit(h, carry, 2, 300, 0.05, 1e-4);
        double probeAcc = probe.accuracy(h, carry);
        double[] dir = ConceptErasure.probeDirections(probe, WIDTH, 2).get(0);
        double probeIia = AlignmentSearch.interchangeAccuracy(m, step,
                AlignmentSearch.orthonormal(new double[][] {dir}), test);
        double[][] centred = new double[h.length][WIDTH];
        double[] mean = new double[WIDTH];
        for (float[] row : h) {
            for (int j = 0; j < WIDTH; j++) {
                mean[j] += row[j] / h.length;
            }
        }
        for (int i = 0; i < h.length; i++) {
            for (int j = 0; j < WIDTH; j++) {
                centred[i][j] = h[i][j] - mean[j];
            }
        }
        double[][] v = ThinSvd.of(centred).v();
        double[] pc1 = new double[WIDTH];
        double[] pc2 = new double[WIDTH];
        for (int j = 0; j < WIDTH; j++) {
            pc1[j] = v[j][0];
            pc2[j] = v[j][1];
        }
        double pca1 = AlignmentSearch.interchangeAccuracy(m, step, AlignmentSearch.orthonormal(new double[][] {pc1}),
                test);
        double pca2 = AlignmentSearch.interchangeAccuracy(m, step,
                AlignmentSearch.orthonormal(new double[][] {pc1, pc2}), test);
        double best = 0;
        for (int j = 0; j < WIDTH; j++) {
            double[] e = new double[WIDTH];
            e[j] = 1;
            best = Math.max(best, AlignmentSearch.interchangeAccuracy(m, step,
                    AlignmentSearch.orthonormal(new double[][] {e}), test));
        }
        return new AlignmentScores(acc, das[0], das[1], probeIia, probeAcc, pca1, pca2, best);
    }
}
