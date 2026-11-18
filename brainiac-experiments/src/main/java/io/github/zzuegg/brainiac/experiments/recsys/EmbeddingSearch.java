package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.recsys.IvfPqIndex;
import io.github.zzuegg.brainiac.experimental.recsys.MatryoshkaEmbedding;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.BinaryOperator;

/**
 * Embedding search experiments.
 * <ul>
 *   <li>{@link #ann} (W7 2.12): IVF-PQ, plain and with ScaNN's anisotropic quantization loss,
 *       against the exact core {@link VectorIndex#flat} and the graph index
 *       {@link VectorIndex#hnsw}, on maximum-inner-product search over Gaussian clusters with
 *       varied vector norms: recall@10 of the exact top 10, queries per second and bytes per
 *       vector;</li>
 *   <li>{@link #matryoshka} (W7 2.14): two-tower retrieval on {@link SyntheticRecsys#preferences}
 *       trained with the Matryoshka loss, against truncating a plainly trained embedding,
 *       PCA-truncating it and separately trained models at each size, by recall@10 against the
 *       dimension, plus adaptive retrieval (shortlist by a 4-dimensional prefix, re-rank with
 *       all 32).</li>
 * </ul>
 */
public final class EmbeddingSearch {

    private EmbeddingSearch() {
    }

    /**
     * Approximate nearest neighbours.
     *
     * @param n stored vectors
     * @param dim their width
     * @param queries queries
     * @param seed the seed
     * @return index name to {@code {recall@10, queries per second, bytes per vector}}
     */
    public static Map<String, double[]> ann(int n, int dim, int queries, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int clusters = 50;
        float[][] centres = new float[clusters][dim];
        for (float[] c : centres) {
            for (int j = 0; j < dim; j++) {
                c[j] = (float) r.nextGaussian();
            }
        }
        float[] data = new float[n * dim];
        for (int i = 0; i < n; i++) {
            float[] c = centres[r.nextInt(clusters)];
            double scale = Math.exp(0.3 * r.nextGaussian());
            for (int j = 0; j < dim; j++) {
                data[i * dim + j] = (float) (scale * (c[j] + 0.6 * r.nextGaussian()));
            }
        }
        float[][] q = new float[queries][dim];
        int[][] truth = new int[queries][];
        for (int k = 0; k < queries; k++) {
            float[] c = centres[r.nextInt(clusters)];
            for (int j = 0; j < dim; j++) {
                q[k][j] = (float) (c[j] + 0.6 * r.nextGaussian());
            }
            float[] s = new float[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < dim; j++) {
                    s[i] += q[k][j] * data[i * dim + j];
                }
            }
            truth[k] = RankingMetrics.topK(s, 10, null);
        }
        Tensor all = Tensor.of(Shape.of(n, dim), data);
        Map<String, double[]> out = new LinkedHashMap<>();
        VectorIndex flat = VectorIndex.flat(dim, VectorIndex.Metric.DOT, Device.cpu());
        flat.add(all);
        out.put("flat (exact)", measure(flat, q, truth, 4.0 * dim));
        VectorIndex hnsw = VectorIndex.hnsw(dim, VectorIndex.Metric.DOT, 16, 100, 50, seed);
        hnsw.add(all);
        out.put("HNSW", measure(hnsw, q, truth, 4.0 * dim + 2 * 16 * 4));
        for (double eta : new double[] {1, 4}) {
            for (int probe : new int[] {4, 16}) {
                IvfPqIndex pq = IvfPqIndex.of(dim, VectorIndex.Metric.DOT, 64, 8, 64, probe, eta, seed);
                pq.add(all);
                out.put("IVF-PQ" + (eta == 1 ? "" : " anisotropic") + " probe " + probe,
                        measure(pq, q, truth, pq.bytesPerVector()));
            }
        }
        return out;
    }

    private static double[] measure(VectorIndex index, float[][] queries, int[][] truth, double bytes) {
        long start = System.nanoTime();
        double recall = 0;
        for (int k = 0; k < queries.length; k++) {
            List<VectorIndex.Hit> hits = index.search(Tensor.of(Shape.of(queries[k].length), queries[k]), 10);
            Set<Integer> want = new HashSet<>();
            for (int t : truth[k]) {
                want.add(t);
            }
            for (VectorIndex.Hit h : hits) {
                recall += want.contains(h.id()) ? 0.1 : 0;
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        return new double[] {recall / queries.length, queries.length / seconds, bytes};
    }

    private record Towers(Model user, Model item) {
    }

    private static Towers train(SyntheticRecsys.Preferences data, int trainUsers, int dim,
            BinaryOperator<Var> loss, int steps, long seed) {
        int f = data.userFeatures().shape().dim(1);
        Model user = Model.sequential().input(f).dense(64, Activation.RELU).dense(dim).seed(seed).build();
        Model item = Model.sequential().input(f).dense(64, Activation.RELU).dense(dim).seed(seed + 1).build();
        List<Var> params = new ArrayList<>(user.parameters());
        params.addAll(item.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 2);
        float[] uf = data.userFeatures().toArray();
        float[] itf = data.itemFeatures().toArray();
        for (int s = 0; s < steps; s++) {
            float[] bu = new float[64 * f];
            float[] bi = new float[64 * f];
            for (int b = 0; b < 64; b++) {
                int u = r.nextInt(trainUsers);
                int it = data.relevant()[u][r.nextInt(data.relevant()[u].length)];
                System.arraycopy(uf, u * f, bu, b * f, f);
                System.arraycopy(itf, it * f, bi, b * f, f);
            }
            loss.apply(user.forward(Var.of(Tensor.of(Shape.of(64, f), bu)), true),
                    item.forward(Var.of(Tensor.of(Shape.of(64, f), bi)), true)).backward();
            opt.step(params);
        }
        return new Towers(user, item);
    }

    private static float[][] embed(Model m, Tensor x) {
        Tensor e = Var.noGrad(() -> m.forward(Var.of(x), false)).value();
        int n = e.shape().dim(0);
        int d = e.shape().dim(1);
        float[] a = e.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * d, out[i], 0, d);
        }
        return out;
    }

    private static double recall(float[][] users, float[][] items, SyntheticRecsys.Preferences data, int from,
            int size) {
        double total = 0;
        for (int u = from; u < users.length; u++) {
            float[] s = new float[items.length];
            for (int i = 0; i < items.length; i++) {
                s[i] = (float) MatryoshkaEmbedding.prefixCosine(users[u], items[i], size);
            }
            Set<Integer> rel = new HashSet<>();
            for (int i : data.relevant()[u]) {
                rel.add(i);
            }
            total += RankingMetrics.recall(RankingMetrics.topK(s, 10, null), rel, 10);
        }
        return total / (users.length - from);
    }

    private static float[][] project(float[][] x, double[][] basis, int size) {
        float[][] out = new float[x.length][size];
        for (int i = 0; i < x.length; i++) {
            for (int k = 0; k < size; k++) {
                double s = 0;
                for (int j = 0; j < x[i].length; j++) {
                    s += x[i][j] * basis[k][j];
                }
                out[i][k] = (float) s;
            }
        }
        return out;
    }

    /**
     * Matryoshka embeddings.
     *
     * @param steps training steps of batch 64 per model
     * @param seed the seed
     * @return method name to recall@10 at dimensions 4, 8, 16 and 32, and "adaptive" rows to
     *     {@code {recall@10}}
     */
    public static Map<String, double[]> matryoshka(int steps, long seed) {
        int users = 500;
        int trainUsers = 400;
        SyntheticRecsys.Preferences data = SyntheticRecsys.preferences(users, 400, 8, 16, 0.1, 10, seed);
        int[] sizes = {4, 8, 16, 32};
        BinaryOperator<Var> infoNce = (a, b) -> Functional.infoNce(a, b, 0.1f);
        Map<String, double[]> out = new LinkedHashMap<>();
        Towers plain = train(data, trainUsers, 32, infoNce, steps, seed);
        float[][] pu = embed(plain.user(), data.userFeatures());
        float[][] pi = embed(plain.item(), data.itemFeatures());
        Towers mrl = train(data, trainUsers, 32, MatryoshkaEmbedding.of(sizes, infoNce)::compute, steps, seed);
        float[][] mu = embed(mrl.user(), data.userFeatures());
        float[][] mi = embed(mrl.item(), data.itemFeatures());
        double[][] cov = new double[32][32];
        for (float[] v : pi) {
            for (int a = 0; a < 32; a++) {
                for (int b = 0; b < 32; b++) {
                    cov[a][b] += v[a] * v[b];
                }
            }
        }
        MlMatrices.Eigen eig = MlMatrices.eigh(cov);
        double[][] basis = new double[32][];
        int[] order = RankingMetrics.topK(toFloat(eig.values()), 32, null);
        for (int k = 0; k < 32; k++) {
            basis[k] = eig.vector(order[k]);
        }
        double[] truncated = new double[sizes.length];
        double[] matryoshka = new double[sizes.length];
        double[] pca = new double[sizes.length];
        double[] separate = new double[sizes.length];
        for (int k = 0; k < sizes.length; k++) {
            int m = sizes[k];
            truncated[k] = recall(pu, pi, data, trainUsers, m);
            matryoshka[k] = recall(mu, mi, data, trainUsers, m);
            pca[k] = recall(project(pu, basis, m), project(pi, basis, m), data, trainUsers, m);
            if (m == 32) {
                separate[k] = truncated[k];
            } else {
                Towers own = train(data, trainUsers, m, infoNce, steps, seed);
                separate[k] = recall(embed(own.user(), data.userFeatures()), embed(own.item(), data.itemFeatures()),
                        data, trainUsers, m);
            }
        }
        out.put("plain truncated", truncated);
        out.put("PCA of plain", pca);
        out.put("separate models", separate);
        out.put("Matryoshka", matryoshka);
        out.put("adaptive plain (4 -> 32)", new double[] {adaptive(pu, pi, data, trainUsers)});
        out.put("adaptive Matryoshka (4 -> 32)", new double[] {adaptive(mu, mi, data, trainUsers)});
        return out;
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    private static double adaptive(float[][] users, float[][] items, SyntheticRecsys.Preferences data, int from) {
        double total = 0;
        for (int u = from; u < users.length; u++) {
            Set<Integer> rel = new HashSet<>();
            for (int i : data.relevant()[u]) {
                rel.add(i);
            }
            total += RankingMetrics.recall(MatryoshkaEmbedding.adaptiveSearch(users[u], items, 4, 40, 10), rel, 10);
        }
        return total / (users.length - from);
    }
}
