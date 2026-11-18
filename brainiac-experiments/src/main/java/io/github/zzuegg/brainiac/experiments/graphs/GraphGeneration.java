package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.graph.GraphDiffusion;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Graph generation (wave 7, 7.1.16): DiGress-style discrete diffusion
 * ({@link GraphDiffusion}) against an Erdős–Rényi model with matched density, a
 * GraphRNN-lite (a GRU writes the BFS-ordered adjacency row by row) and a variational graph
 * autoencoder decoded from sampled latents. Data: 12-node stochastic block models with
 * two or three communities. Metrics, against held-out graphs: the maximum mean discrepancy
 * (Gaussian kernel on total variation) of the degree, clustering-coefficient and
 * Laplacian-spectrum histograms, and validity, the fraction of samples whose best
 * spectral 2- or 3-way split has modularity above 0.3 (community structure recovered).
 */
public final class GraphGeneration {

    private static final int N = 12;

    private GraphGeneration() {
    }

    /** SBM graphs with two or three communities, node ids permuted. */
    static List<float[]> communityGraphs(int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> out = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int blocks = 2 + g % 2;
            int[] perm = permutation(N, r);
            float[] a = new float[N * N];
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    boolean same = i * blocks / N == j * blocks / N;
                    if (r.nextDouble() < (same ? 0.7 : 0.05)) {
                        a[perm[i] * N + perm[j]] = 1;
                        a[perm[j] * N + perm[i]] = 1;
                    }
                }
            }
            out.add(a);
        }
        return out;
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static double[] degreeHistogram(float[] a) {
        double[] h = new double[N];
        for (int i = 0; i < N; i++) {
            int d = 0;
            for (int j = 0; j < N; j++) {
                d += a[i * N + j] > 0 ? 1 : 0;
            }
            h[Math.min(N - 1, d)] += 1.0 / N;
        }
        return h;
    }

    private static double[] clusteringHistogram(float[] a) {
        double[] h = new double[10];
        for (int i = 0; i < N; i++) {
            List<Integer> nb = new ArrayList<>();
            for (int j = 0; j < N; j++) {
                if (a[i * N + j] > 0) {
                    nb.add(j);
                }
            }
            double c = 0;
            if (nb.size() > 1) {
                int links = 0;
                for (int x = 0; x < nb.size(); x++) {
                    for (int y = x + 1; y < nb.size(); y++) {
                        links += a[nb.get(x) * N + nb.get(y)] > 0 ? 1 : 0;
                    }
                }
                c = links / (nb.size() * (nb.size() - 1) / 2.0);
            }
            h[Math.min(9, (int) (c * 10))] += 1.0 / N;
        }
        return h;
    }

    private static double[] spectrumHistogram(float[] a) {
        float[] lap = new float[N * N];
        double[] d = new double[N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                d[i] += a[i * N + j];
            }
        }
        for (int i = 0; i < N; i++) {
            lap[i * N + i] = d[i] > 0 ? 1 : 0;
            for (int j = 0; j < N; j++) {
                if (a[i * N + j] > 0) {
                    lap[i * N + j] = (float) (-1 / Math.sqrt(d[i] * d[j]));
                }
            }
        }
        float[] ev = Linalg.eigh(Tensor.of(Shape.of(N, N), lap))[0].toArray();
        double[] h = new double[10];
        for (float e : ev) {
            h[Math.max(0, Math.min(9, (int) (e / 2 * 10)))] += 1.0 / N;
        }
        return h;
    }

    private static double kernel(double[] x, double[] y) {
        double tv = 0;
        for (int i = 0; i < x.length; i++) {
            tv += Math.abs(x[i] - y[i]);
        }
        tv /= 2;
        return Math.exp(-tv * tv / 2);
    }

    static double mmd(List<double[]> a, List<double[]> b) {
        double xx = 0;
        double yy = 0;
        double xy = 0;
        for (double[] x : a) {
            for (double[] y : a) {
                xx += kernel(x, y);
            }
            for (double[] y : b) {
                xy += kernel(x, y);
            }
        }
        for (double[] x : b) {
            for (double[] y : b) {
                yy += kernel(x, y);
            }
        }
        return xx / (a.size() * (double) a.size()) + yy / (b.size() * (double) b.size())
                - 2 * xy / (a.size() * (double) b.size());
    }

    private static double modularity(float[] a, int[] part) {
        double m = 0;
        double[] d = new double[N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                d[i] += a[i * N + j];
            }
            m += d[i];
        }
        if (m == 0) {
            return 0;
        }
        double q = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (part[i] == part[j]) {
                    q += a[i * N + j] - d[i] * d[j] / m;
                }
            }
        }
        return q / m;
    }

    /** Grid validity: connected with maximum degree 4. */
    static boolean gridLike(float[] a) {
        boolean[] seen = new boolean[N];
        ArrayDeque<Integer> q = new ArrayDeque<>(List.of(0));
        seen[0] = true;
        int count = 1;
        while (!q.isEmpty()) {
            int v = q.poll();
            int d = 0;
            for (int u = 0; u < N; u++) {
                if (a[v * N + u] > 0) {
                    d++;
                    if (!seen[u]) {
                        seen[u] = true;
                        count++;
                        q.add(u);
                    }
                }
            }
            if (d > 4) {
                return false;
            }
        }
        return count == N;
    }

    static boolean valid(float[] a) {
        double[][] aff = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                aff[i][j] = a[i * N + j];
            }
        }
        double best = 0;
        for (int k = 2; k <= 3; k++) {
            best = Math.max(best, modularity(a, SpectralClustering.cluster(aff, k, 1)));
        }
        return best > 0.3;
    }

    private static void score(String name, List<float[]> samples, List<float[]> reference, boolean grids,
            Map<String, Double> out) {
        out.put(name + " degree MMD", GraphTraining.round(mmd(samples.stream().map(GraphGeneration::degreeHistogram)
                .toList(), reference.stream().map(GraphGeneration::degreeHistogram).toList())));
        out.put(name + " clustering MMD", GraphTraining.round(mmd(samples.stream()
                .map(GraphGeneration::clusteringHistogram).toList(), reference.stream()
                .map(GraphGeneration::clusteringHistogram).toList())));
        out.put(name + " spectral MMD", GraphTraining.round(mmd(samples.stream().map(GraphGeneration::spectrumHistogram)
                .toList(), reference.stream().map(GraphGeneration::spectrumHistogram).toList())));
        out.put(name + " valid", GraphTraining.round(samples.stream()
                .filter(grids ? GraphGeneration::gridLike : GraphGeneration::valid).count() / (double) samples.size()));
    }

    /**
     * Trains every generator and scores its samples.
     *
     * @param grids whether the data are 3 x 4 grids with dropped edges (validity: connected,
     *     degree at most 4) instead of community graphs
     * @param trainGraphs training graphs
     * @param samples graphs sampled per model (and held-out reference graphs)
     * @param steps training steps (batches of 8 graphs)
     * @param seed the seed
     * @return "model metric" for degree, clustering and spectral MMD and validity, plus a
     *     "held-out" row (a second real sample, the noise floor)
     */
    public static Map<String, Double> compare(boolean grids, int trainGraphs, int samples, int steps, long seed) {
        java.util.function.BiFunction<Integer, Long, List<float[]>> data = grids ? GraphGeneration::grids
                : GraphGeneration::communityGraphs;
        List<float[]> train = data.apply(trainGraphs, seed);
        List<float[]> reference = data.apply(samples, seed + 1);
        Map<String, Double> out = new LinkedHashMap<>();
        score("held-out", data.apply(samples, seed + 2), reference, grids, out);
        double density = 0;
        for (float[] a : train) {
            for (float v : a) {
                density += v;
            }
        }
        density /= trainGraphs * (double) N * (N - 1);
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> er = new ArrayList<>();
        for (int k = 0; k < samples; k++) {
            float[] a = new float[N * N];
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    if (r.nextDouble() < density) {
                        a[i * N + j] = 1;
                        a[j * N + i] = 1;
                    }
                }
            }
            er.add(a);
        }
        score("Erdos-Renyi", er, reference, grids, out);
        score("GraphRNN-lite", graphRnn(train, samples, steps, seed), reference, grids, out);
        score("VGAE", vgae(train, samples, steps, seed), reference, grids, out);
        GraphDiffusion diffusion = new GraphDiffusion(N, 16, 2, 16, new SplittableRandom(seed));
        diffusion.fitMarginal(train);
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom noise = new SplittableRandom(seed + 5);
        for (int s = 0; s < steps; s++) {
            List<float[]> batch = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                batch.add(train.get(noise.nextInt(train.size())));
            }
            diffusion.loss(batch, noise).backward();
            opt.step(diffusion.parameters());
        }
        score("DiGress", diffusion.sample(samples, noise), reference, grids, out);
        return out;
    }

    /** Breadth-first node order from a random start, as GraphRNN uses. */
    private static float[] bfsOrder(float[] a, SplittableRandom r) {
        int[] order = new int[N];
        boolean[] seen = new boolean[N];
        int k = 0;
        while (k < N) {
            int start;
            do {
                start = r.nextInt(N);
            } while (seen[start]);
            ArrayDeque<Integer> q = new ArrayDeque<>(List.of(start));
            seen[start] = true;
            while (!q.isEmpty()) {
                int v = q.poll();
                order[k++] = v;
                for (int u = 0; u < N; u++) {
                    if (a[v * N + u] > 0 && !seen[u]) {
                        seen[u] = true;
                        q.add(u);
                    }
                }
            }
        }
        float[] out = new float[N * N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                out[i * N + j] = a[order[i] * N + order[j]];
            }
        }
        return out;
    }

    /** GraphRNN-S: h_i = GRU(h_{i-1}, row_{i-1}); row_i ~ Bernoulli(sigmoid(MLP(h_i))), lower triangle. */
    private static List<float[]> graphRnn(List<float[]> train, int samples, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        GruCell gru = new GruCell(N, 32, r);
        Var w = Var.param(Tensor.normal(Shape.of(32, N), 0, 0.1f, r));
        Var b = Var.param(Tensor.zeros(Shape.of(1, N)));
        List<Var> params = new ArrayList<>(gru.parameters());
        params.addAll(List.of(w, b));
        float[] lower = new float[N * N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < i; j++) {
                lower[i * N + j] = 1;
            }
        }
        Optimizer opt = Optimizer.adam(0.01f);
        int batch = 8;
        for (int s = 0; s < steps; s++) {
            float[][] rows = new float[batch][];
            for (int k = 0; k < batch; k++) {
                rows[k] = bfsOrder(train.get(r.nextInt(train.size())), r);
            }
            Var h = gru.zeroState(batch);
            Var loss = null;
            for (int i = 0; i < N; i++) {
                float[] prev = new float[batch * N];
                float[] target = new float[batch * N];
                float[] mask = new float[batch * N];
                for (int k = 0; k < batch; k++) {
                    if (i > 0) {
                        System.arraycopy(rows[k], (i - 1) * N, prev, k * N, N);
                    }
                    System.arraycopy(rows[k], i * N, target, k * N, N);
                    System.arraycopy(lower, i * N, mask, k * N, N);
                }
                h = gru.step(Var.of(Tensor.of(Shape.of(batch, N), prev)), h);
                Var logits = h.matmul(w).add(b);
                Var l = logits.softplus().sub(logits.mul(Tensor.of(Shape.of(batch, N), target)))
                        .mul(Tensor.of(Shape.of(batch, N), mask)).sum();
                loss = loss == null ? l : loss.add(l);
            }
            loss.div((float) batch * N).backward();
            opt.step(params);
        }
        List<float[]> out = new ArrayList<>();
        for (int k = 0; k < samples; k++) {
            float[] a = new float[N * N];
            Var h = gru.zeroState(1);
            float[] prev = new float[N];
            for (int i = 0; i < N; i++) {
                float[] p = prev;
                Var hh = h;
                h = Var.noGrad(() -> gru.step(Var.of(Tensor.of(Shape.of(1, N), p)), hh));
                Var hs = h;
                float[] logits = Var.noGrad(() -> hs.matmul(w).add(b)).value().toArray();
                prev = new float[N];
                for (int j = 0; j < i; j++) {
                    if (r.nextDouble() < 1 / (1 + Math.exp(-logits[j]))) {
                        prev[j] = 1;
                        a[i * N + j] = 1;
                        a[j * N + i] = 1;
                    }
                }
            }
            out.add(a);
        }
        return out;
    }

    /** A variational graph autoencoder; samples decode {@code sigmoid(z z^T + c)} from {@code z ~ N(0, I)}. */
    private static List<float[]> vgae(List<float[]> train, int samples, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = 8;
        GcnLayer hidden = new GcnLayer(N, 16, Activation.RELU, r);
        GcnLayer mu = new GcnLayer(16, d, Activation.IDENTITY, r);
        GcnLayer logVar = new GcnLayer(16, d, Activation.IDENTITY, r);
        Var c = Var.param(Tensor.zeros(Shape.of(1, 1)));
        List<Var> params = new ArrayList<>(hidden.parameters());
        params.addAll(mu.parameters());
        params.addAll(logVar.parameters());
        params.add(c);
        float[] eye = new float[N * N];
        for (int i = 0; i < N; i++) {
            eye[i * N + i] = 1;
        }
        Var x = Var.of(Tensor.of(Shape.of(N, N), eye));
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Tensor a = Tensor.of(Shape.of(N, N), train.get(r.nextInt(train.size())));
            Var adj = Var.of(a);
            Var h = hidden.forward(List.of(x, adj), true);
            Var m = mu.forward(List.of(h, adj), true);
            Var lv = logVar.forward(List.of(h, adj), true);
            Var z = m.add(lv.mul(0.5f).exp().mul(Tensor.normal(Shape.of(N, d), 0, 1, r)));
            Var logits = z.matmul(z.transpose()).add(c);
            Var recon = logits.softplus().sub(logits.mul(a)).mean();
            Var kl = lv.exp().add(m.square()).sub(lv).sub(1f).mean().mul(0.5f / N);
            recon.add(kl).backward();
            opt.step(params);
        }
        List<float[]> out = new ArrayList<>();
        float bias = c.value().item();
        for (int k = 0; k < samples; k++) {
            float[] z = Tensor.normal(Shape.of(N, d), 0, 1, r).toArray();
            float[] a = new float[N * N];
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    double dot = bias;
                    for (int q = 0; q < d; q++) {
                        dot += z[i * d + q] * z[j * d + q];
                    }
                    if (r.nextDouble() < 1 / (1 + Math.exp(-dot))) {
                        a[i * N + j] = 1;
                        a[j * N + i] = 1;
                    }
                }
            }
            out.add(a);
        }
        return out;
    }

    static List<float[]> grids(int count, long seed) {
        // 3 x 4 = N nodes
        return SyntheticGraphs.grids(count, 3, 4, 0.1, seed).stream().map(SyntheticGraphs.Graph::adjacency).toList();
    }
}
