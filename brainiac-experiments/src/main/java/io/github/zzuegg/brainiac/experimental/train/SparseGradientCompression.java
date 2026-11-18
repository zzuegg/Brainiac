package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Unstructured sparse gradient compression: each worker sends only its {@code k} largest-magnitude
 * entries (value and index, 8 bytes each) to the others (an all-gather), in three variants.
 *
 * <ul>
 *   <li>{@link #topK} with or without classic error feedback (Seide et al. 2014; Stich et al.
 *       2018, <a href="https://arxiv.org/abs/1809.07599">arXiv:1809.07599</a>): the unsent rest
 *       is accumulated locally and added to the next gradient.</li>
 *   <li>{@link #dgc} (deep gradient compression, Lin et al. 2018,
 *       <a href="https://arxiv.org/abs/1712.01887">arXiv:1712.01887</a>): error feedback on
 *       <em>momentum-corrected</em> velocities ({@code u = m u + g; v += u}, both zeroed where
 *       sent), local gradient-norm clipping, and a sparsity warm-up from 25% down to the target
 *       over the first steps. Pair it with an outer optimizer without momentum.</li>
 *   <li>{@link #ef21} (Richtarik, Sokolov and Fatkhullin 2021,
 *       <a href="https://arxiv.org/abs/2106.05203">arXiv:2106.05203</a>): each worker keeps a
 *       running estimate {@code g_i} and sends only the compressed <em>difference</em>
 *       {@code c_i = topk(grad_i - g_i)}, {@code g_i += c_i}; the aggregate is the mean
 *       {@code g_i}. A Markov compressor with clean convergence guarantees.</li>
 * </ul>
 */
public final class SparseGradientCompression implements GradientCompressor {

    private enum Mode { TOPK, DGC, EF21 }

    private final Mode mode;
    private final double fraction;
    private final boolean errorFeedback;
    private final double momentum;
    private final double clip;
    private final int warmup;
    private final Map<Var, float[][]> residual = new IdentityHashMap<>();
    private final Map<Var, float[][]> velocity = new IdentityHashMap<>();
    private final Map<Var, Integer> calls = new IdentityHashMap<>();
    private long bytes;

    private SparseGradientCompression(Mode mode, double fraction, boolean errorFeedback, double momentum, double clip,
            int warmup) {
        this.mode = mode;
        this.fraction = fraction;
        this.errorFeedback = errorFeedback;
        this.momentum = momentum;
        this.clip = clip;
        this.warmup = warmup;
    }

    /**
     * Top-k sparsification.
     *
     * @param fraction the share of entries sent, for example 0.01
     * @param errorFeedback whether to accumulate and re-send the unsent rest
     * @return the compressor
     */
    public static SparseGradientCompression topK(double fraction, boolean errorFeedback) {
        return new SparseGradientCompression(Mode.TOPK, fraction, errorFeedback, 0, 0, 0);
    }

    /**
     * Deep gradient compression.
     *
     * @param fraction the target share of entries sent
     * @param momentum the local momentum, 0.9
     * @param clip the local gradient-norm clip, 0 for none
     * @param warmup steps of the exponential sparsity warm-up
     * @return the compressor
     */
    public static SparseGradientCompression dgc(double fraction, double momentum, double clip, int warmup) {
        return new SparseGradientCompression(Mode.DGC, fraction, true, momentum, clip, warmup);
    }

    /**
     * EF21.
     *
     * @param fraction the share of entries of each difference sent
     * @return the compressor
     */
    public static SparseGradientCompression ef21(double fraction) {
        return new SparseGradientCompression(Mode.EF21, fraction, true, 0, 0, 0);
    }

    @Override
    public Tensor aggregate(Var parameter, List<Tensor> grads) {
        int workers = grads.size();
        int n = Math.toIntExact(grads.get(0).size());
        int step = calls.merge(parameter, 1, Integer::sum) - 1;
        double share = fraction;
        if (mode == Mode.DGC && step < warmup) {
            share = Math.max(fraction, 0.25 * Math.pow(fraction / 0.25, (step + 1.0) / warmup));
        }
        int k = Math.max(1, (int) Math.ceil(share * n));
        float[][] res = residual.computeIfAbsent(parameter, p -> new float[workers][n]);
        float[][] vel = mode == Mode.DGC ? velocity.computeIfAbsent(parameter, p -> new float[workers][n]) : null;
        float[] sum = new float[n];
        for (int w = 0; w < workers; w++) {
            float[] g = grads.get(w).toArray();
            if (mode == Mode.DGC && clip > 0) {
                double norm = 0;
                for (float v : g) {
                    norm += v * v;
                }
                double limit = clip / Math.sqrt(workers);
                norm = Math.sqrt(norm);
                if (norm > limit) {
                    for (int i = 0; i < n; i++) {
                        g[i] *= (float) (limit / norm);
                    }
                }
            }
            float[] acc = new float[n];
            switch (mode) {
                case TOPK -> {
                    for (int i = 0; i < n; i++) {
                        acc[i] = g[i] + (errorFeedback ? res[w][i] : 0);
                    }
                }
                case DGC -> {
                    for (int i = 0; i < n; i++) {
                        vel[w][i] = (float) (momentum * vel[w][i] + g[i]);
                        acc[i] = res[w][i] + vel[w][i];
                    }
                }
                case EF21 -> {
                    for (int i = 0; i < n; i++) {
                        acc[i] = g[i] - res[w][i];
                    }
                }
                default -> throw new IllegalStateException();
            }
            boolean[] keep = largest(acc, k);
            for (int i = 0; i < n; i++) {
                float sent = keep[i] ? acc[i] : 0;
                switch (mode) {
                    case TOPK -> {
                        if (errorFeedback) {
                            res[w][i] = acc[i] - sent;
                        }
                        sum[i] += sent;
                    }
                    case DGC -> {
                        res[w][i] = acc[i] - sent;
                        if (keep[i]) {
                            vel[w][i] = 0;  // momentum factor masking
                        }
                        sum[i] += sent;
                    }
                    case EF21 -> {
                        res[w][i] += sent;  // g_i += c_i
                        sum[i] += res[w][i];
                    }
                    default -> throw new IllegalStateException();
                }
            }
        }
        bytes += 8L * k * workers * workers;  // each sends k (value, index) pairs and receives the others'
        for (int i = 0; i < n; i++) {
            sum[i] /= workers;
        }
        return Tensor.of(grads.get(0).shape(), sum);
    }

    /** Marks the {@code k} entries of largest magnitude. */
    static boolean[] largest(float[] x, int k) {
        int n = x.length;
        boolean[] keep = new boolean[n];
        if (k >= n) {
            java.util.Arrays.fill(keep, true);
            return keep;
        }
        float[] mag = new float[n];
        for (int i = 0; i < n; i++) {
            mag[i] = Math.abs(x[i]);
        }
        float[] sorted = mag.clone();
        java.util.Arrays.sort(sorted);
        float threshold = sorted[n - k];
        int taken = 0;
        for (int i = 0; i < n && taken < k; i++) {
            if (mag[i] > threshold) {
                keep[i] = true;
                taken++;
            }
        }
        for (int i = 0; i < n && taken < k; i++) {
            if (!keep[i] && mag[i] == threshold) {
                keep[i] = true;
                taken++;
            }
        }
        return keep;
    }

    @Override
    public long bytes() {
        return bytes;
    }

    @Override
    public String toString() {
        return switch (mode) {
            case TOPK -> "top-k(" + fraction + (errorFeedback ? ", EF" : "") + ")";
            case DGC -> "DGC(" + fraction + ")";
            case EF21 -> "EF21(" + fraction + ")";
        };
    }
}
