package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An approximate nearest-neighbour index of an inverted file with product quantization, IVF-PQ
 * (Jégou, Douze and Schmid, 2011, "Product Quantization for Nearest Neighbor Search"), as a core
 * {@link VectorIndex}. A coarse k-means splits the space into {@code lists} cells; each vector is
 * stored in its cell as the product-quantized code of its residual from the cell centre: the
 * residual is cut into {@code subspaces} sub-vectors, and each is replaced by the id of the nearest
 * of {@code codes} codewords of its own k-means codebook (one byte for 256 codes). A search visits
 * only the {@code probe} cells closest to the query and computes distances by table lookup: one
 * table of query-to-codeword distances per subspace, then one addition per subspace per vector.
 *
 * <p>With {@code eta > 1} codes are chosen with the score-aware (anisotropic) loss of ScaNN (Guo et
 * al., 2020, <a href="https://arxiv.org/abs/1908.10396">arXiv:1908.10396</a>): quantization error
 * parallel to the vector costs {@code eta} times more than orthogonal error, which preserves the
 * inner products that maximum-inner-product search ranks by. The codebooks themselves stay k-means.
 *
 * <p>The first {@link #add} trains the coarse centres and codebooks on (a sample of) its vectors.
 * Searches return the {@link Metric} score computed from the codes, so it is approximate.
 */
public final class IvfPqIndex implements VectorIndex {

    private final int dim;
    private final Metric metric;
    private final int lists;
    private final int subspaces;
    private final int codes;
    private final int probe;
    private final double eta;
    private final long seed;
    private final int sub;
    private double[][] centres;
    private double[][][] codebooks;
    private final List<List<Integer>> members = new ArrayList<>();
    private final List<byte[]> stored = new ArrayList<>();
    private final List<Integer> cell = new ArrayList<>();
    private final BitSet removed = new BitSet();

    private IvfPqIndex(int dim, Metric metric, int lists, int subspaces, int codes, int probe, double eta,
            long seed) {
        if (dim % subspaces != 0) {
            throw new IllegalArgumentException(subspaces + " subspaces do not divide " + dim);
        }
        if (codes > 256) {
            throw new IllegalArgumentException("at most 256 codes per subspace (one byte)");
        }
        if (metric == Metric.COSINE) {
            throw new IllegalArgumentException("normalize vectors and use DOT for cosine");
        }
        this.dim = dim;
        this.metric = metric;
        this.lists = lists;
        this.subspaces = subspaces;
        this.codes = codes;
        this.probe = probe;
        this.eta = eta;
        this.seed = seed;
        this.sub = dim / subspaces;
    }

    /**
     * Creates an empty index.
     *
     * @param dim the vector width
     * @param metric {@link Metric#DOT} or {@link Metric#EUCLIDEAN}
     * @param lists coarse cells
     * @param subspaces PQ sub-vectors per vector, dividing {@code dim}
     * @param codes codewords per subspace, at most 256
     * @param probe cells visited per search
     * @param eta the anisotropic weight of parallel error, 1 for plain PQ
     * @param seed the k-means seed
     * @return the index
     */
    public static IvfPqIndex of(int dim, Metric metric, int lists, int subspaces, int codes, int probe, double eta,
            long seed) {
        return new IvfPqIndex(dim, metric, lists, subspaces, codes, probe, eta, seed);
    }

    /** @return bytes of code per stored vector (not counting the cell id lists) */
    public int bytesPerVector() {
        return subspaces;
    }

    private void train(double[][] x) {
        SplittableRandom r = new SplittableRandom(seed);
        int sample = Math.min(x.length, 4000);
        double[][] s = new double[sample][];
        for (int i = 0; i < sample; i++) {
            s[i] = x[r.nextInt(x.length)];
        }
        KMeans coarse = KMeans.fit(s, Math.min(lists, sample), KMeans.Init.PLUS_PLUS, 1, 15, seed);
        centres = coarse.centres();
        for (int i = 0; i < centres.length; i++) {
            members.add(new ArrayList<>());
        }
        codebooks = new double[subspaces][][];
        for (int m = 0; m < subspaces; m++) {
            double[][] part = new double[sample][sub];
            for (int i = 0; i < sample; i++) {
                double[] c = centres[coarse.labels()[i]];
                for (int j = 0; j < sub; j++) {
                    part[i][j] = s[i][m * sub + j] - c[m * sub + j];
                }
            }
            codebooks[m] = KMeans.fit(part, Math.min(codes, sample), KMeans.Init.PLUS_PLUS, 1, 15, seed + m)
                    .centres();
        }
    }

    @Override
    public int add(Tensor vectors) {
        int n = vectors.shape().dim(0);
        float[] v = vectors.toArray();
        double[][] x = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                x[i][j] = v[i * dim + j];
            }
        }
        if (centres == null) {
            train(x);
        }
        int first = stored.size();
        for (double[] row : x) {
            int c = KMeans.nearest(centres, row);
            double[] residual = new double[dim];
            for (int j = 0; j < dim; j++) {
                residual[j] = row[j] - centres[c][j];
            }
            members.get(c).add(stored.size());
            cell.add(c);
            stored.add(encode(row, residual));
        }
        return first;
    }

    private byte[] encode(double[] x, double[] r) {
        byte[] code = new byte[subspaces];
        double[][] err = new double[subspaces][];
        for (int m = 0; m < subspaces; m++) {
            double[] part = java.util.Arrays.copyOfRange(r, m * sub, (m + 1) * sub);
            int best = KMeans.nearest(codebooks[m], part);
            code[m] = (byte) best;
            err[m] = error(part, codebooks[m][best]);
        }
        if (eta == 1) {
            return code;
        }
        double norm = Math.sqrt(dot(x, x)) + 1e-12;
        double[] unit = new double[dim];
        for (int j = 0; j < dim; j++) {
            unit[j] = x[j] / norm;
        }
        for (int pass = 0; pass < 2; pass++) {
            for (int m = 0; m < subspaces; m++) {
                double parRest = 0;
                for (int o = 0; o < subspaces; o++) {
                    if (o != m) {
                        for (int j = 0; j < sub; j++) {
                            parRest += err[o][j] * unit[o * sub + j];
                        }
                    }
                }
                double[] part = java.util.Arrays.copyOfRange(r, m * sub, (m + 1) * sub);
                double bestLoss = Double.POSITIVE_INFINITY;
                int best = 0;
                for (int c = 0; c < codebooks[m].length; c++) {
                    double sq = 0;
                    double par = parRest;
                    for (int j = 0; j < sub; j++) {
                        double e = part[j] - codebooks[m][c][j];
                        sq += e * e;
                        par += e * unit[m * sub + j];
                    }
                    double loss = sq + (eta - 1) * par * par;
                    if (loss < bestLoss) {
                        bestLoss = loss;
                        best = c;
                    }
                }
                code[m] = (byte) best;
                err[m] = error(part, codebooks[m][best]);
            }
        }
        return code;
    }

    private static double[] error(double[] a, double[] b) {
        double[] e = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            e[i] = a[i] - b[i];
        }
        return e;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private double coarseScore(double[] q, double[] c) {
        if (metric == Metric.DOT) {
            return dot(q, c);
        }
        double s = 0;
        for (int j = 0; j < dim; j++) {
            s -= (q[j] - c[j]) * (q[j] - c[j]);
        }
        return s;
    }

    @Override
    public List<Hit> search(Tensor query, int k) {
        float[] qf = query.toArray();
        double[] q = new double[dim];
        for (int j = 0; j < dim; j++) {
            q[j] = qf[j];
        }
        float[] coarse = new float[centres.length];
        for (int c = 0; c < centres.length; c++) {
            coarse[c] = (float) coarseScore(q, centres[c]);
        }
        int[] cells = RankingMetrics.topK(coarse, Math.min(probe, centres.length), null);
        List<Hit> hits = new ArrayList<>();
        double[][] table = new double[subspaces][];
        for (int c : cells) {
            for (int m = 0; m < subspaces; m++) {
                table[m] = new double[codebooks[m].length];
                for (int w = 0; w < codebooks[m].length; w++) {
                    double s = 0;
                    for (int j = 0; j < sub; j++) {
                        double cw = codebooks[m][w][j];
                        if (metric == Metric.DOT) {
                            s += q[m * sub + j] * cw;
                        } else {
                            double d = q[m * sub + j] - centres[c][m * sub + j] - cw;
                            s -= d * d;
                        }
                    }
                    table[m][w] = s;
                }
            }
            double base = metric == Metric.DOT ? dot(q, centres[c]) : 0;
            for (int id : members.get(c)) {
                if (removed.get(id)) {
                    continue;
                }
                byte[] code = stored.get(id);
                double s = base;
                for (int m = 0; m < subspaces; m++) {
                    s += table[m][code[m] & 0xff];
                }
                hits.add(new Hit(id, (float) s));
            }
        }
        hits.sort((a, b) -> Float.compare(b.score(), a.score()));
        return hits.size() > k ? new ArrayList<>(hits.subList(0, k)) : hits;
    }

    @Override
    public int size() {
        return stored.size() - removed.cardinality();
    }

    @Override
    public boolean remove(int id) {
        if (!contains(id)) {
            return false;
        }
        removed.set(id);
        return true;
    }

    @Override
    public boolean contains(int id) {
        return id >= 0 && id < stored.size() && !removed.get(id);
    }

    @Override
    public String toString() {
        return "IvfPqIndex(" + lists + " lists, " + subspaces + "x" + codes + " codes, probe " + probe
                + (eta != 1 ? ", eta " + eta : "") + ")";
    }
}
