package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * The sparsifying activations shared by sparse dictionaries ({@link SparseAutoencoder},
 * {@link Transcoder}, {@link CrossLayerTranscoder}, {@link Crosscoder}):
 *
 * <ul>
 *   <li>{@link #topK}: keep the {@code k} largest positive codes of each row (Gao et al.
 *       2024, <a href="https://arxiv.org/abs/2406.04093">arXiv:2406.04093</a>);</li>
 *   <li>{@link #batchTopK}: keep the {@code k * rows} largest positive codes of the whole
 *       batch, so busy samples may use more features and easy ones fewer (Bussmann et al.
 *       2024, <a href="https://arxiv.org/abs/2412.06410">arXiv:2412.06410</a>); at inference
 *       a single threshold replaces the batch rule ({@link #threshold});</li>
 *   <li>{@link #jumpRelu}: {@code z = pre * H(pre - theta)} with a learned per-feature
 *       threshold, trained by straight-through estimation with a rectangle kernel of width
 *       {@code epsilon}; {@link #jumpL0} is its differentiable L0 penalty (Rajamanoharan et
 *       al. 2024, <a href="https://arxiv.org/abs/2407.14435">arXiv:2407.14435</a>).</li>
 * </ul>
 *
 * <p>The straight-through gradients are built from ordinary autograd ops with the
 * {@code s - s.detach()} trick, so no custom op is needed: the forward value is exact and
 * only the threshold's gradient is the pseudo-derivative.
 */
public final class SparseCoding {

    private SparseCoding() {
    }

    /**
     * Keeps the {@code k} largest positive entries of each row.
     *
     * @param z non-negative codes {@code [rows, features]}
     * @param k entries kept per row
     * @return the masked codes
     */
    public static Var topK(Var z, int k) {
        int features = z.shape().dim(-1);
        float[] v = z.value().to(DType.FLOAT32).toArray();
        int rows = v.length / features;
        float[] mask = new float[v.length];
        float[] row = new float[features];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(v, r * features, row, 0, features);
            float[] sorted = row.clone();
            Arrays.sort(sorted);
            float cut = sorted[Math.max(0, features - Math.min(k, features))];
            int kept = 0;
            for (int f = 0; f < features && kept < k; f++) {
                if (row[f] >= cut && row[f] > 0) {
                    mask[r * features + f] = 1;
                    kept++;
                }
            }
        }
        return z.mul(Tensor.of(z.shape(), mask, z.value().device()).to(z.dtype()));
    }

    /**
     * Keeps the {@code k * rows} largest positive entries of the whole batch.
     *
     * @param z non-negative codes {@code [rows, features]}
     * @param k the mean number of entries kept per row
     * @return the masked codes
     */
    public static Var batchTopK(Var z, int k) {
        float[] v = z.value().to(DType.FLOAT32).toArray();
        int rows = v.length / z.shape().dim(-1);
        int keep = Math.min(v.length, k * rows);
        float[] sorted = v.clone();
        Arrays.sort(sorted);
        float cut = sorted[v.length - keep];
        float[] mask = new float[v.length];
        int kept = 0;
        for (int i = 0; i < v.length && kept < keep; i++) {
            if (v[i] >= cut && v[i] > 0) {
                mask[i] = 1;
                kept++;
            }
        }
        return z.mul(Tensor.of(z.shape(), mask, z.value().device()).to(z.dtype()));
    }

    /**
     * Keeps the entries above a fixed threshold: batch top-k at inference.
     *
     * @param z non-negative codes
     * @param threshold the cut
     * @return the masked codes
     */
    public static Var threshold(Var z, float threshold) {
        return z.mul(z.value().greater(Tensor.scalar(threshold).to(z.value().device())).to(z.dtype()));
    }

    /**
     * The smallest positive entry a batch top-k kept: running-average it to get the
     * inference threshold.
     *
     * @param kept codes after {@link #batchTopK}
     * @return the smallest positive value, or 0 if none
     */
    public static float smallestActive(Var kept) {
        float min = Float.POSITIVE_INFINITY;
        for (float x : kept.value().to(DType.FLOAT32).toArray()) {
            if (x > 0 && x < min) {
                min = x;
            }
        }
        return Float.isInfinite(min) ? 0 : min;
    }

    /**
     * JumpReLU: {@code pre * H(pre - theta)}. The gradient to {@code pre} is the mask; the
     * gradient to {@code theta} is the straight-through estimate
     * {@code -(theta / epsilon) K((pre - theta) / epsilon)} with a rectangle kernel.
     *
     * @param pre pre-activations {@code [rows, features]}
     * @param theta positive thresholds {@code [features]} (for example {@code exp(logTheta)})
     * @param epsilon the kernel bandwidth
     * @return the codes
     */
    public static Var jumpRelu(Var pre, Var theta, float epsilon) {
        float[] p = pre.value().to(DType.FLOAT32).toArray();
        float[] t = theta.value().to(DType.FLOAT32).toArray();
        int features = t.length;
        float[] mask = new float[p.length];
        float[] coef = new float[p.length];
        for (int i = 0; i < p.length; i++) {
            float th = t[i % features];
            mask[i] = p[i] > th ? 1 : 0;
            coef[i] = Math.abs((p[i] - th) / epsilon) < 0.5f ? -th / epsilon : 0;
        }
        Var surrogate = theta.mul(Tensor.of(pre.shape(), coef, pre.value().device()).to(pre.dtype()));
        return pre.mul(Tensor.of(pre.shape(), mask, pre.value().device()).to(pre.dtype()))
                .add(surrogate.sub(surrogate.detach()));
    }

    /**
     * The mean number of active features per row, {@code sum H(pre - theta) / rows}, with
     * the straight-through gradient {@code -(1 / epsilon) K((pre - theta) / epsilon)} to
     * {@code theta}.
     *
     * @param pre pre-activations {@code [rows, features]}
     * @param theta positive thresholds {@code [features]}
     * @param epsilon the kernel bandwidth
     * @return the scalar penalty
     */
    public static Var jumpL0(Var pre, Var theta, float epsilon) {
        float[] p = pre.value().to(DType.FLOAT32).toArray();
        float[] t = theta.value().to(DType.FLOAT32).toArray();
        int features = t.length;
        int rows = p.length / features;
        float[] coef = new float[p.length];
        double count = 0;
        for (int i = 0; i < p.length; i++) {
            float th = t[i % features];
            count += p[i] > th ? 1 : 0;
            coef[i] = Math.abs((p[i] - th) / epsilon) < 0.5f ? -1f / (epsilon * rows) : 0;
        }
        Var s = theta.mul(Tensor.of(pre.shape(), coef, pre.value().device()).to(pre.dtype())).sum();
        return s.sub(s.detach()).add((float) (count / rows));
    }

    /**
     * Reads a batch of rows out of a codes tensor.
     *
     * @param codes {@code [rows, features]}
     * @return {@code [row][feature]}
     */
    public static float[][] rows(Var codes) {
        int features = codes.shape().dim(-1);
        float[] z = codes.value().to(DType.FLOAT32).toArray();
        float[][] out = new float[z.length / features][features];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(z, i * features, out[i], 0, features);
        }
        return out;
    }
}
