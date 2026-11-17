package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * AQLM-style additive quantization: each group of {@code g} consecutive input weights of an
 * output column is coded as the <em>sum</em> of {@code M} codewords, one from each of
 * {@code M} learned codebooks of {@code 2^B} vectors, so it costs {@code M B / g} bits per
 * weight: {@code M = 2, B = 8, g = 8} is 2 bits. Additive codes fit the weight distribution far
 * better than scalar grids at 2 bits; their codes cannot be found greedily, so AQLM alternates
 * a beam search over code tuples (with the codebooks fixed) and a codebook update (with the
 * codes fixed).
 *
 * <p>Here the codebooks start from residual k-means (each codebook clusters what the previous
 * ones left), then {@code rounds} times: every group is re-encoded by beam search (each
 * candidate tuple tries every code of every codebook; the {@code beam} best tuples survive),
 * then every codeword is set to the weighted mean of what its groups still need, which is the
 * exact minimiser for one codebook with the others fixed. Errors are weighted per input
 * channel by the calibration inputs' second moments {@code diag(X^T X)}, the diagonal of the
 * layer's output-error Hessian, so channels that carry large activations are coded more
 * carefully (AQLM uses the full Hessian and fine-tunes codebooks end to end; this keeps the
 * diagonal and the closed-form update). Each column is scaled to unit RMS first; the scales
 * are kept in full precision. Codebook storage ({@code M 2^B g} values per layer) is not
 * counted in the bits per weight, as it is negligible for the large layers AQLM targets.
 *
 * <p>Egiazarian et al., "Extreme Compression of Large Language Models via Additive
 * Quantization" (ICML 2024), <a href="https://arxiv.org/abs/2401.06118">arXiv:2401.06118</a>.
 */
public final class AdditiveQuant {

    private final int codebooks;
    private final int codeBits;
    private final int group;
    private final int beam;
    private final int rounds;
    private final long seed;

    /**
     * @param codebooks the codebooks {@code M}
     * @param codeBits bits per code {@code B}
     * @param group weights per group {@code g}
     * @param beam the beam width of the encoder
     * @param rounds alternations of beam search and codebook update
     * @param seed seeds the k-means start
     */
    public AdditiveQuant(int codebooks, int codeBits, int group, int beam, int rounds, long seed) {
        if (codebooks < 1 || codeBits < 1 || codeBits > 12 || group < 1 || beam < 1 || rounds < 0) {
            throw new IllegalArgumentException("bad additive quantizer settings");
        }
        this.codebooks = codebooks;
        this.codeBits = codeBits;
        this.group = group;
        this.beam = beam;
        this.rounds = rounds;
        this.seed = seed;
    }

    /** @return bits per weight, codebooks not counted */
    public double bitsPerWeight() {
        return codebooks * codeBits / (double) group;
    }

    /**
     * Quantizes a weight and returns it dequantized, with codebooks of its own.
     *
     * @param weight {@code [in, out]}; {@code in} must be a multiple of the group size
     * @param calibration inputs {@code [samples, in]} for the channel weights, or {@code null}
     *     for plain mean-squared error
     * @return the reconstruction
     */
    public Tensor quantize(Tensor weight, Tensor calibration) {
        return quantizeAll(List.of(weight), java.util.Collections.singletonList(calibration)).getFirst();
    }

    /**
     * Quantizes several weights with one shared set of codebooks (for small models, where
     * per-layer codebooks would cost more bits than the codes; every column is scaled to unit
     * RMS first, so the groups of all layers share one distribution).
     *
     * @param weights the weights {@code [in, out]}; every {@code in} a multiple of the group size
     * @param calibrations per weight its inputs, or {@code null} entries for plain error
     * @return the reconstructions, in order
     */
    public List<Tensor> quantizeAll(List<Tensor> weights, List<Tensor> calibrations) {
        List<float[]> vecs = new ArrayList<>();
        List<double[]> wts = new ArrayList<>();
        List<float[]> scales = new ArrayList<>();
        for (int k = 0; k < weights.size(); k++) {
            Tensor weight = weights.get(k);
            Tensor calibration = calibrations.get(k);
            int in = weight.shape().dim(0);
            int out = weight.shape().dim(1);
            if (in % group != 0) {
                throw new IllegalArgumentException("input size " + in + " is not a multiple of the group " + group);
            }
            float[] w = weight.toArray();
            double[] h = new double[in];
            if (calibration != null) {
                float[] x = calibration.toArray();
                int rows = x.length / in;
                for (int n = 0; n < rows; n++) {
                    for (int i = 0; i < in; i++) {
                        h[i] += x[n * in + i] * x[n * in + i];
                    }
                }
                double mean = Arrays.stream(h).average().orElse(1);
                for (int i = 0; i < in; i++) {
                    h[i] = h[i] / Math.max(mean, 1e-12) + 1e-3;
                }
            } else {
                Arrays.fill(h, 1);
            }
            int perColumn = in / group;
            float[] scale = new float[out];
            for (int c = 0; c < out; c++) {
                double ss = 0;
                for (int i = 0; i < in; i++) {
                    ss += w[i * out + c] * w[i * out + c];
                }
                scale[c] = (float) Math.max(Math.sqrt(ss / in), 1e-12);
                for (int gi = 0; gi < perColumn; gi++) {
                    float[] v = new float[group];
                    double[] wt = new double[group];
                    for (int j = 0; j < group; j++) {
                        v[j] = w[(gi * group + j) * out + c] / scale[c];
                        wt[j] = h[gi * group + j];
                    }
                    vecs.add(v);
                    wts.add(wt);
                }
            }
            scales.add(scale);
        }
        int n = vecs.size();
        float[][] vec = vecs.toArray(new float[0][]);
        double[][] wt = wts.toArray(new double[0][]);
        int size = 1 << codeBits;
        float[][][] books = new float[codebooks][size][group];
        int[][] codes = new int[n][codebooks];
        float[][] residual = new float[n][];
        for (int v = 0; v < n; v++) {
            residual[v] = vec[v].clone();
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int m = 0; m < codebooks; m++) {
            kmeans(residual, wt, books[m], codes, m, r);
            for (int v = 0; v < n; v++) {
                float[] cw = books[m][codes[v][m]];
                for (int j = 0; j < group; j++) {
                    residual[v][j] -= cw[j];
                }
            }
        }
        for (int round = 0; round < rounds; round++) {
            for (int v = 0; v < n; v++) {
                codes[v] = beamSearch(vec[v], wt[v], books, codes[v]);
            }
            for (int m = 0; m < codebooks; m++) {
                update(vec, wt, books, codes, m);
            }
        }
        List<Tensor> outList = new ArrayList<>();
        int v = 0;
        for (int k = 0; k < weights.size(); k++) {
            Tensor weight = weights.get(k);
            int in = weight.shape().dim(0);
            int out = weight.shape().dim(1);
            int perColumn = in / group;
            float[] q = new float[in * out];
            float[] scale = scales.get(k);
            for (int c = 0; c < out; c++) {
                for (int gi = 0; gi < perColumn; gi++) {
                    float[] y = sum(books, codes[v++]);
                    for (int j = 0; j < group; j++) {
                        q[(gi * group + j) * out + c] = y[j] * scale[c];
                    }
                }
            }
            outList.add(Tensor.of(Shape.of(in, out), q, weight.device()));
        }
        return outList;
    }

    /**
     * @param weights the number of weights coded with one set of codebooks
     * @return bits per weight including the codebooks, stored as 16-bit values
     */
    public double effectiveBits(long weights) {
        return bitsPerWeight() + 16.0 * codebooks * (1 << codeBits) * group / weights;
    }

    /**
     * Quantizes a dense layer.
     *
     * @param dense the layer
     * @param calibration its inputs {@code [samples, in]}, or {@code null}
     * @return a copy with the dequantized weight
     */
    public Dense quantize(Dense dense, Tensor calibration) {
        return WeightPruning.replaceWeight(dense, quantize(dense.weight().value(), calibration));
    }

    private static double error(float[] x, double[] wt, float[] y) {
        double e = 0;
        for (int j = 0; j < x.length; j++) {
            double d = x[j] - y[j];
            e += wt[j] * d * d;
        }
        return e;
    }

    /** Weighted Lloyd iterations for codebook {@code m}, assigning {@code codes[v][m]}. */
    private void kmeans(float[][] data, double[][] wt, float[][] book, int[][] codes, int m, SplittableRandom r) {
        int n = data.length;
        int k = book.length;
        for (int c = 0; c < k; c++) {
            book[c] = data[r.nextInt(n)].clone();
        }
        for (int it = 0; it < 12; it++) {
            for (int v = 0; v < n; v++) {
                int best = 0;
                double bestErr = Double.MAX_VALUE;
                for (int c = 0; c < k; c++) {
                    double e = error(data[v], wt[v], book[c]);
                    if (e < bestErr) {
                        bestErr = e;
                        best = c;
                    }
                }
                codes[v][m] = best;
            }
            double[][] sum = new double[k][group];
            double[][] den = new double[k][group];
            for (int v = 0; v < n; v++) {
                int c = codes[v][m];
                for (int j = 0; j < group; j++) {
                    sum[c][j] += wt[v][j] * data[v][j];
                    den[c][j] += wt[v][j];
                }
            }
            for (int c = 0; c < k; c++) {
                for (int j = 0; j < group; j++) {
                    if (den[c][j] > 0) {
                        book[c][j] = (float) (sum[c][j] / den[c][j]);
                    }
                }
            }
        }
    }

    /** Beam search over code tuples, starting from the current tuple. */
    private int[] beamSearch(float[] x, double[] wt, float[][][] books, int[] start) {
        int size = books[0].length;
        int[][] tuples = {start.clone()};
        double[] errors = {error(x, wt, sum(books, start))};
        for (int pass = 0; pass < 2; pass++) {
            for (int m = 0; m < codebooks; m++) {
                int cand = tuples.length * size;
                int[][] next = new int[cand][];
                double[] err = new double[cand];
                int at = 0;
                for (int[] t : tuples) {
                    float[] rest = sum(books, t);
                    float[] old = books[m][t[m]];
                    for (int j = 0; j < group; j++) {
                        rest[j] -= old[j];
                    }
                    for (int c = 0; c < size; c++) {
                        double e = 0;
                        float[] cw = books[m][c];
                        for (int j = 0; j < group; j++) {
                            double d = x[j] - rest[j] - cw[j];
                            e += wt[j] * d * d;
                        }
                        int[] u = t.clone();
                        u[m] = c;
                        next[at] = u;
                        err[at++] = e;
                    }
                }
                Integer[] order = new Integer[cand];
                for (int i = 0; i < cand; i++) {
                    order[i] = i;
                }
                Arrays.sort(order, (a, b) -> Double.compare(err[a], err[b]));
                int keep = Math.min(beam, cand);
                tuples = new int[keep][];
                errors = new double[keep];
                for (int i = 0; i < keep; i++) {
                    tuples[i] = next[order[i]];
                    errors[i] = err[order[i]];
                }
            }
        }
        return tuples[0];
    }

    private static float[] sum(float[][][] books, int[] tuple) {
        float[] s = new float[books[0][0].length];
        for (int m = 0; m < tuple.length; m++) {
            float[] cw = books[m][tuple[m]];
            for (int j = 0; j < s.length; j++) {
                s[j] += cw[j];
            }
        }
        return s;
    }

    /** Sets every codeword of codebook {@code m} to the weighted mean of what its groups need. */
    private void update(float[][] vec, double[][] wt, float[][][] books, int[][] codes, int m) {
        int k = books[m].length;
        double[][] num = new double[k][group];
        double[][] den = new double[k][group];
        for (int v = 0; v < vec.length; v++) {
            float[] rest = sum(books, codes[v]);
            float[] own = books[m][codes[v][m]];
            int c = codes[v][m];
            for (int j = 0; j < group; j++) {
                double need = vec[v][j] - (rest[j] - own[j]);
                num[c][j] += wt[v][j] * need;
                den[c][j] += wt[v][j];
            }
        }
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < group; j++) {
                if (den[c][j] > 0) {
                    books[m][c][j] = (float) (num[c][j] / den[c][j]);
                }
            }
        }
    }

    @Override
    public String toString() {
        return "AdditiveQuant(M=" + codebooks + ", B=" + codeBits + ", g=" + group + ", " + bitsPerWeight()
                + " bits)";
    }
}
