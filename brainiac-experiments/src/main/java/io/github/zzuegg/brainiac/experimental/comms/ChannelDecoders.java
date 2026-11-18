package io.github.zzuegg.brainiac.experimental.comms;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes.LinearCode;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Soft and hard decoders for binary linear block codes on their Tanner graph, and the Viterbi
 * decoder for the rate-1/2 (7, 5) convolutional code. Inputs are channel log-likelihood ratios
 * {@code log P(0)/P(1)} (BPSK maps 0 to +1).
 *
 * <ul>
 *   <li>{@link Method#HARD_SYNDROME}: bounded-distance syndrome decoding with a coset-leader
 *       table of all error patterns up to weight 3.</li>
 *   <li>{@link Method#BP}: sum-product belief propagation (tanh rule), with early stopping on a
 *       zero syndrome.</li>
 *   <li>{@link Method#MIN_SUM} and {@link Method#NORMALIZED_MIN_SUM} (check messages scaled by
 *       0.8, Fossorier et al. 1999).</li>
 *   <li>{@link Method#NEURAL_BP} and {@link Method#NEURAL_MIN_SUM}: BP / min-sum unrolled for a
 *       fixed number of iterations with a learned weight per edge and iteration on the check
 *       messages and on the channel LLRs, trained with autograd on noisy all-zero codewords
 *       (valid for any codeword by symmetry) with the multi-loss of Nachmani, Be'ery and
 *       Burshtein, "Learning to decode linear codes using deep learning" (Allerton 2016),
 *       <a href="https://arxiv.org/abs/1607.04793">arXiv:1607.04793</a>.</li>
 *   <li>{@link Method#MAXIMUM_LIKELIHOOD}: exhaustive search over all codewords (small k).</li>
 * </ul>
 */
public final class ChannelDecoders {

    /** A block decoder. */
    public enum Method {
        /** coset-leader syndrome decoding of hard decisions. */
        HARD_SYNDROME,
        /** sum-product belief propagation. */
        BP,
        /** min-sum. */
        MIN_SUM,
        /** min-sum with check messages scaled by 0.8. */
        NORMALIZED_MIN_SUM,
        /** BP with learned edge weights. */
        NEURAL_BP,
        /** min-sum with learned edge weights. */
        NEURAL_MIN_SUM,
        /** exhaustive maximum likelihood. */
        MAXIMUM_LIKELIHOOD
    }

    private final LinearCode code;
    private final int edges;
    private final int[] edgeVar;
    private final int[] edgeChk;
    private final int[][] varEdges;
    private final int[][] chkEdges;
    private final int iterations;
    private Map<Long, int[]> cosetLeaders;
    private double[][] codewordSigns;
    // learned weights [iteration][edge] for check messages, [iteration][variable] for channel LLRs
    private double[][] bpWeights;
    private double[][] bpChannel;
    private double[][] msWeights;
    private double[][] msChannel;

    /**
     * Creates decoders for a code.
     *
     * @param code the code
     * @param iterations message-passing iterations
     */
    public ChannelDecoders(LinearCode code, int iterations) {
        this.code = code;
        this.iterations = iterations;
        List<int[]> e = new ArrayList<>();
        for (int c = 0; c < code.h().length; c++) {
            for (int v = 0; v < code.n(); v++) {
                if (code.h()[c][v] != 0) {
                    e.add(new int[] {v, c});
                }
            }
        }
        edges = e.size();
        edgeVar = new int[edges];
        edgeChk = new int[edges];
        List<List<Integer>> ve = new ArrayList<>();
        List<List<Integer>> ce = new ArrayList<>();
        for (int v = 0; v < code.n(); v++) {
            ve.add(new ArrayList<>());
        }
        for (int c = 0; c < code.h().length; c++) {
            ce.add(new ArrayList<>());
        }
        for (int i = 0; i < edges; i++) {
            edgeVar[i] = e.get(i)[0];
            edgeChk[i] = e.get(i)[1];
            ve.get(edgeVar[i]).add(i);
            ce.get(edgeChk[i]).add(i);
        }
        varEdges = ve.stream().map(l -> l.stream().mapToInt(Integer::intValue).toArray()).toArray(int[][]::new);
        chkEdges = ce.stream().map(l -> l.stream().mapToInt(Integer::intValue).toArray()).toArray(int[][]::new);
    }

    /** @return the code */
    public LinearCode code() {
        return code;
    }

    /** @return the number of Tanner-graph edges */
    public int edges() {
        return edges;
    }

    /**
     * Decodes channel LLRs.
     *
     * @param method the method
     * @param llr the channel LLRs
     * @return the decoded codeword bits
     */
    public int[] decode(Method method, double[] llr) {
        return switch (method) {
            case HARD_SYNDROME -> syndromeDecode(llr);
            case BP -> messagePassing(llr, false, 1.0, null, null);
            case MIN_SUM -> messagePassing(llr, true, 1.0, null, null);
            case NORMALIZED_MIN_SUM -> messagePassing(llr, true, 0.8, null, null);
            case NEURAL_BP -> messagePassing(llr, false, 1.0, require(bpWeights), bpChannel);
            case NEURAL_MIN_SUM -> messagePassing(llr, true, 1.0, require(msWeights), msChannel);
            case MAXIMUM_LIKELIHOOD -> maximumLikelihood(llr);
        };
    }

    private static double[][] require(double[][] w) {
        if (w == null) {
            throw new IllegalStateException("train the neural decoder first");
        }
        return w;
    }

    private static int[] hard(double[] llr) {
        int[] b = new int[llr.length];
        for (int i = 0; i < llr.length; i++) {
            b[i] = llr[i] < 0 ? 1 : 0;
        }
        return b;
    }

    private static long key(int[] s) {
        long k = 0;
        for (int b : s) {
            k = k * 2 + b;
        }
        return k;
    }

    private int[] syndromeDecode(double[] llr) {
        if (cosetLeaders == null) {
            cosetLeaders = new HashMap<>();
            int n = code.n();
            cosetLeaders.put(0L, new int[0]);
            for (int a = 0; a < n; a++) {
                addLeader(new int[] {a});
            }
            for (int a = 0; a < n; a++) {
                for (int b = a + 1; b < n; b++) {
                    addLeader(new int[] {a, b});
                }
            }
            if (n <= 64) {
                for (int a = 0; a < n; a++) {
                    for (int b = a + 1; b < n; b++) {
                        for (int c = b + 1; c < n; c++) {
                            addLeader(new int[] {a, b, c});
                        }
                    }
                }
            }
        }
        int[] word = hard(llr);
        int[] fix = cosetLeaders.get(key(code.syndrome(word)));
        if (fix != null) {
            for (int p : fix) {
                word[p] ^= 1;
            }
        }
        return word;
    }

    private void addLeader(int[] positions) {
        int[] e = new int[code.n()];
        for (int p : positions) {
            e[p] = 1;
        }
        cosetLeaders.putIfAbsent(key(code.syndrome(e)), positions);
    }

    private int[] maximumLikelihood(double[] llr) {
        if (codewordSigns == null) {
            int[][] words = code.codewords();
            codewordSigns = new double[words.length][];
            for (int i = 0; i < words.length; i++) {
                codewordSigns[i] = ChannelCodes.bpsk(words[i]);
            }
        }
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < codewordSigns.length; i++) {
            double s = 0;
            double[] x = codewordSigns[i];
            for (int j = 0; j < x.length; j++) {
                s += x[j] * llr[j];
            }
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        return hard(codewordSigns[best]);
    }

    private int[] messagePassing(double[] llr, boolean minSum, double scale, double[][] weights, double[][] channel) {
        int n = code.n();
        double[] c2v = new double[edges];
        double[] v2c = new double[edges];
        double[] total = new double[n];
        int iters = weights != null ? weights.length : iterations;
        for (int t = 0; t < iters; t++) {
            // variable to check
            for (int v = 0; v < n; v++) {
                double wl = channel != null ? channel[t][v] : 1;
                double s = wl * llr[v];
                for (int e : varEdges[v]) {
                    s += (weights != null && t > 0 ? weights[t - 1][e] : 1) * c2v[e];
                }
                for (int e : varEdges[v]) {
                    v2c[e] = s - (weights != null && t > 0 ? weights[t - 1][e] : 1) * c2v[e];
                }
            }
            // check to variable
            for (int[] ce : chkEdges) {
                if (minSum) {
                    double min1 = Double.POSITIVE_INFINITY;
                    double min2 = Double.POSITIVE_INFINITY;
                    int arg = -1;
                    int sign = 1;
                    for (int e : ce) {
                        double a = Math.abs(v2c[e]);
                        sign *= v2c[e] < 0 ? -1 : 1;
                        if (a < min1) {
                            min2 = min1;
                            min1 = a;
                            arg = e;
                        } else if (a < min2) {
                            min2 = a;
                        }
                    }
                    for (int e : ce) {
                        int s = sign * (v2c[e] < 0 ? -1 : 1);
                        c2v[e] = scale * s * (e == arg ? min2 : min1);
                    }
                } else {
                    double prod = 1;
                    int zeros = 0;
                    for (int e : ce) {
                        double t2 = Math.tanh(clip(v2c[e]) / 2);
                        if (t2 == 0) {
                            zeros++;
                        } else {
                            prod *= t2;
                        }
                    }
                    for (int e : ce) {
                        double t2 = Math.tanh(clip(v2c[e]) / 2);
                        double p = zeros > 1 || zeros == 1 && t2 != 0 ? 0 : (t2 == 0 ? prod : prod / t2);
                        p = Math.max(-0.999999999, Math.min(0.999999999, p));
                        c2v[e] = 2 * atanh(p);
                    }
                }
            }
            // marginals
            boolean ok = true;
            for (int v = 0; v < n; v++) {
                double s = llr[v];
                for (int e : varEdges[v]) {
                    s += (weights != null ? weights[t][e] : 1) * c2v[e];
                }
                total[v] = s;
            }
            int[] hardBits = hard(total);
            for (int s : code.syndrome(hardBits)) {
                ok &= s == 0;
            }
            if (ok && weights == null) {
                return hardBits;
            }
        }
        return hard(total);
    }

    private static double clip(double x) {
        return Math.max(-30, Math.min(30, x));
    }

    private static double atanh(double x) {
        return 0.5 * Math.log((1 + x) / (1 - x));
    }

    /**
     * Trains the neural BP or neural min-sum weights on all-zero codewords at random
     * {@code Eb/N0} in {@code [loDb, hiDb]}, by Adam on the multi-loss (binary cross-entropy of
     * every iteration's marginals).
     *
     * @param method {@link Method#NEURAL_BP} or {@link Method#NEURAL_MIN_SUM}
     * @param steps gradient steps
     * @param batch codewords per step
     * @param loDb the lowest training SNR
     * @param hiDb the highest training SNR
     * @param learningRate the Adam step size
     * @param seed the seed
     * @return the final loss
     */
    public double train(Method method, int steps, int batch, double loDb, double hiDb, float learningRate, long seed) {
        boolean minSum = method == Method.NEURAL_MIN_SUM;
        int n = code.n();
        int t = iterations;
        Var w = Var.param(Tensor.of(Shape.of(t, edges), filled(t * edges, 1f)));
        Var wl = Var.param(Tensor.of(Shape.of(t, n), filled(t * n, 1f)));
        Tensor varIdx = indices(edgeVar);
        Tensor chkIdx = indices(edgeChk);
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom rnd = new SplittableRandom(seed);
        double last = 0;
        int checks = code.h().length;
        for (int step = 0; step < steps; step++) {
            float[] l = new float[batch * n];
            for (int b = 0; b < batch; b++) {
                double sigma = ChannelCodes.sigma(loDb + (hiDb - loDb) * rnd.nextDouble(), code.rate());
                for (int v = 0; v < n; v++) {
                    l[b * n + v] = (float) (2 * (1 + sigma * ChannelCodes.gaussian(rnd)) / (sigma * sigma));
                }
            }
            Var llr = Var.of(Tensor.of(Shape.of(batch, n), l));
            Var c2v = Var.of(Tensor.zeros(Shape.of(batch, edges)));
            Var loss = null;
            for (int it = 0; it < t; it++) {
                Var wIt = w.slice(0, it, it + 1);
                Var wPrev = it == 0 ? null : w.slice(0, it - 1, it);
                Var weighted = wPrev == null ? c2v : c2v.mul(wPrev);
                Var chan = llr.mul(wl.slice(0, it, it + 1));
                Var totalV = chan.add(weighted.scatterAdd(1, varIdx, n));
                Var v2c = totalV.gather(1, varIdx).sub(weighted);
                c2v = minSum ? minSumCheck(v2c, checks) : bpCheck(v2c, chkIdx, checks);
                Var marg = llr.add(c2v.mul(wIt).scatterAdd(1, varIdx, n));
                Var term = marg.neg().softplus().mean();
                loss = loss == null ? term : loss.add(term);
            }
            loss.backward();
            last = loss.value().item() / t;
            opt.step(List.of(w, wl));
        }
        double[][] wOut = toRows(w.value().toArray(), t, edges);
        double[][] lOut = toRows(wl.value().toArray(), t, n);
        if (minSum) {
            msWeights = wOut;
            msChannel = lOut;
        } else {
            bpWeights = wOut;
            bpChannel = lOut;
        }
        return last;
    }

    private Var bpCheck(Var v2c, Tensor chkIdx, int checks) {
        float[] vals = v2c.value().toArray();
        int batch = vals.length / edges;
        // sign of the product over the other edges, computed outside the tape
        float[] sign = new float[vals.length];
        for (int b = 0; b < batch; b++) {
            for (int[] ce : chkEdges) {
                int neg = 0;
                for (int e : ce) {
                    neg += vals[b * edges + e] < 0 ? 1 : 0;
                }
                for (int e : ce) {
                    int own = vals[b * edges + e] < 0 ? 1 : 0;
                    sign[b * edges + e] = ((neg - own) % 2 == 0) ? 1 : -1;
                }
            }
        }
        Var logAbs = v2c.mul(0.5f).clamp(-15f, 15f).tanh().abs().clamp(1e-7f, 1f).log();
        Var excl = logAbs.scatterAdd(1, chkIdx, checks).gather(1, chkIdx).sub(logAbs);
        Var mag = excl.exp().clamp(0f, 0.999999f).mul(Tensor.of(v2c.shape(), sign));
        // 2 atanh(x) = log((1 + x) / (1 - x))
        return mag.add(1f).div(mag.neg().add(1f)).log();
    }

    private Var minSumCheck(Var v2c, int checks) {
        float[] vals = v2c.value().toArray();
        int batch = vals.length / edges;
        float[] sign = new float[vals.length];
        float[] pick = new float[vals.length];
        for (int b = 0; b < batch; b++) {
            for (int[] ce : chkEdges) {
                int neg = 0;
                int arg1 = -1;
                int arg2 = -1;
                for (int e : ce) {
                    float a = Math.abs(vals[b * edges + e]);
                    neg += vals[b * edges + e] < 0 ? 1 : 0;
                    if (arg1 < 0 || a < Math.abs(vals[b * edges + arg1])) {
                        arg2 = arg1;
                        arg1 = e;
                    } else if (arg2 < 0 || a < Math.abs(vals[b * edges + arg2])) {
                        arg2 = e;
                    }
                }
                for (int e : ce) {
                    int own = vals[b * edges + e] < 0 ? 1 : 0;
                    sign[b * edges + e] = ((neg - own) % 2 == 0) ? 1 : -1;
                    pick[b * edges + e] = b * edges + (e == arg1 ? arg2 : arg1);
                }
            }
        }
        Var flat = v2c.abs().reshape(batch * edges).gather(0, Tensor.of(Shape.of(batch * edges), pick));
        return flat.reshape(batch, edges).mul(Tensor.of(v2c.shape(), sign));
    }

    private static Tensor indices(int[] idx) {
        float[] f = new float[idx.length];
        for (int i = 0; i < idx.length; i++) {
            f[i] = idx[i];
        }
        return Tensor.of(Shape.of(idx.length), f);
    }

    private static float[] filled(int n, float v) {
        float[] f = new float[n];
        Arrays.fill(f, v);
        return f;
    }

    private static double[][] toRows(float[] v, int rows, int cols) {
        double[][] out = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[r][c] = v[r * cols + c];
            }
        }
        return out;
    }

    /**
     * Viterbi decoding of the terminated (7, 5) convolutional code on its 8-state edge trellis
     * {@code (u_t, u_(t-1), u_(t-2))}, reusing {@link HiddenMarkovModel#viterbi}: the soft
     * metric of each edge is {@code (llr_0 x_0 + llr_1 x_1) / 2} with {@code x} the edge's BPSK
     * output.
     *
     * @param llr channel LLRs of the {@code 2 (k + 2)} coded bits
     * @return the {@code k} decoded message bits
     */
    public static int[] viterbi(double[] llr) {
        int steps = llr.length / 2;
        int states = 8;
        double neg = -1e30;
        double[] start = new double[states];
        Arrays.fill(start, neg);
        double[][] trans = new double[states][states];
        for (double[] row : trans) {
            Arrays.fill(row, neg);
        }
        for (int s = 0; s < states; s++) {
            int u = s >> 2 & 1;
            int s1 = s >> 1 & 1;
            if ((s & 3) == 0) {
                start[s] = 0;
            }
            for (int next = 0; next < 2; next++) {
                int ns = next << 2 | u << 1 | s1;
                trans[s][ns] = 0;
            }
        }
        double[][] emit = new double[steps][states];
        for (int t = 0; t < steps; t++) {
            for (int s = 0; s < states; s++) {
                int u = s >> 2 & 1;
                int s1 = s >> 1 & 1;
                int s2 = s & 1;
                double x0 = 1 - 2 * (u ^ s1 ^ s2);
                double x1 = 1 - 2 * (u ^ s2);
                emit[t][s] = 0.5 * (llr[2 * t] * x0 + llr[2 * t + 1] * x1);
                if (t >= steps - 2 && u == 1) {
                    emit[t][s] = neg;
                }
            }
        }
        int[] path = HiddenMarkovModel.viterbi(start, trans, emit);
        int[] bits = new int[steps - 2];
        for (int t = 0; t < steps - 2; t++) {
            bits[t] = path[t] >> 2 & 1;
        }
        return bits;
    }
}
