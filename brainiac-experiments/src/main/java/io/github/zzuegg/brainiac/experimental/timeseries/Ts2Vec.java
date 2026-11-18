package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TS2Vec (Yue et al., AAAI 2022, <a href="https://arxiv.org/abs/2106.10466">arXiv:2106.10466</a>):
 * universal per-timestamp representations of series learned by hierarchical contrast. The
 * encoder is an input projection, random timestamp masking (training only), and residual
 * blocks of non-causal dilated convolutions (dilation {@code 2^i}). Two overlapping random
 * crops of each series are encoded; on their common segment, two InfoNCE losses are applied
 * at every level of a max-pooling hierarchy (halving the length until one step is left):
 * <b>instance-wise</b> (at a timestamp, the same instance's other view against all other
 * instances in the batch) and <b>temporal</b> (within an instance, the same timestamp's
 * other view against all other timestamps). Representations are max-pooled over time for
 * classification, or taken at the last step for forecasting with a linear head.
 *
 * <p>As a {@link Layer}: input {@code [batch, time]} (univariate), output the per-timestamp
 * representations {@code [batch, time, width]}.
 */
public final class Ts2Vec implements Layer {

    private final int hidden;
    private final int width;
    private final Var inProj;
    private final Var inBias;
    private final List<Var[]> blocks = new ArrayList<>();
    private final Var outW;
    private final Var outB;

    /**
     * Creates the encoder.
     *
     * @param hidden the convolution channels
     * @param width the representation width
     * @param depth residual dilated blocks
     * @param random the initialisation source
     */
    public Ts2Vec(int hidden, int width, int depth, SplittableRandom random) {
        this.hidden = hidden;
        this.width = width;
        inProj = Var.param(Tensor.normal(Shape.of(1, hidden), 0, 1f, random));
        inBias = Var.param(Tensor.zeros(Shape.of(hidden)));
        float s = (float) Math.sqrt(1.0 / (3 * hidden));
        for (int i = 0; i < depth; i++) {
            blocks.add(new Var[] {
                Var.param(Tensor.normal(Shape.of(hidden, hidden, 3), 0, s, random)),
                Var.param(Tensor.zeros(Shape.of(hidden))),
                Var.param(Tensor.normal(Shape.of(hidden, hidden, 3), 0, s, random)),
                Var.param(Tensor.zeros(Shape.of(hidden)))
            });
        }
        outW = Var.param(Tensor.normal(Shape.of(hidden, width), 0, (float) Math.sqrt(1.0 / hidden), random));
        outB = Var.param(Tensor.zeros(Shape.of(width)));
    }

    /**
     * Encodes series.
     *
     * @param x {@code [batch, time]}
     * @param mask {@code [batch, time]} with 1 for kept and 0 for masked timestamps, or {@code null}
     * @return {@code [batch, time, width]}
     */
    public Var encode(Var x, Tensor mask) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var h = x.reshape(b, t, 1).matmul(inProj).add(inBias);                // [b, t, hidden]
        if (mask != null) {
            h = h.mul(mask.reshape(b, t, 1));
        }
        h = h.permute(0, 2, 1);                                                // [b, hidden, t]
        for (int i = 0; i < blocks.size(); i++) {
            int d = 1 << i;
            Var[] w = blocks.get(i);
            Var y = Functional.conv(h.activation(ActivationOp.GELU, 0), w[0],
                    w[1], new int[] {1}, new int[] {d}, new int[] {d}, new int[] {d}, 1);
            y = Functional.conv(y.activation(ActivationOp.GELU, 0), w[2], w[3],
                    new int[] {1}, new int[] {d}, new int[] {d}, new int[] {d}, 1);
            h = h.add(y);
        }
        return h.permute(0, 2, 1).matmul(outW).add(outB);
    }

    @Override
    public Var forward(Var input) {
        return encode(input, null);
    }

    private static Var logSoftmaxExcludingSelf(Var sim, int n) {
        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) {
            eye[i * n + i] = -1e9f;
        }
        Var masked = sim.add(Tensor.of(Shape.of(n, n), eye));
        return masked.sub(masked.logsumexp(-1, true));
    }

    /** The mean log-probability of the partner at offset {@code half} in a {@code [*, 2half, 2half]} block. */
    private static Var partnerLoss(Var logp, int half) {
        int n = 2 * half;
        float[] pick = new float[n * n];
        for (int i = 0; i < n; i++) {
            pick[i * n + (i + half) % n] = 1f / n;
        }
        return logp.mul(Tensor.of(Shape.of(n, n), pick)).sum(-1, false).sum(-1, false).mean().neg();
    }

    /**
     * The hierarchical contrastive loss of two views' representations on their common segment.
     *
     * @param r1 {@code [batch, time, width]}
     * @param r2 {@code [batch, time, width]}
     * @return the loss
     */
    public static Var hierarchicalLoss(Var r1, Var r2) {
        Var total = null;
        int levels = 0;
        while (true) {
            int b = r1.shape().dim(0);
            int t = r1.shape().dim(1);
            Var loss = null;
            if (b > 1) {
                Var z = Var.concat(0, r1, r2).permute(1, 0, 2);                // [t, 2b, w]
                loss = partnerLoss(logSoftmaxExcludingSelf(z.matmul(z.permute(0, 2, 1)), 2 * b), b);
            }
            if (t > 1) {
                Var z = Var.concat(1, r1, r2);                                 // [b, 2t, w]
                Var temporal = partnerLoss(logSoftmaxExcludingSelf(z.matmul(z.permute(0, 2, 1)), 2 * t), t);
                loss = loss == null ? temporal : loss.add(temporal);
            }
            if (loss != null) {
                total = total == null ? loss : total.add(loss);
                levels++;
            }
            if (t <= 1) {
                break;
            }
            int half = t / 2;
            int w = r1.shape().dim(2);
            r1 = r1.slice(1, 0, 2 * half).reshape(b, half, 2, w).max(2, false);
            r2 = r2.slice(1, 0, 2 * half).reshape(b, half, 2, w).max(2, false);
        }
        return total.mul(1f / Math.max(1, levels));
    }

    /**
     * Self-supervised training on unlabelled series.
     *
     * @param series {@code [n][length]}
     * @param steps Adam steps on batches of {@code batch} series
     * @param batch the batch size
     * @param seed the cropping and masking seed
     */
    public void fit(float[][] series, int steps, int batch, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adamW(1e-3f, 1e-4f);
        int length = series[0].length;
        for (int s = 0; s < steps; s++) {
            // overlapping crops [a1, b1) and [a2, b2) with a1 <= a2 < b1 <= b2
            int overlap = Math.max(4, r.nextInt(length / 4, length + 1));
            int a2 = r.nextInt(0, length - overlap + 1);
            int b1 = a2 + overlap;
            int a1 = r.nextInt(0, a2 + 1);
            int b2 = r.nextInt(b1, length + 1);
            float[] x1 = new float[batch * (b1 - a1)];
            float[] x2 = new float[batch * (b2 - a2)];
            float[] m1 = new float[x1.length];
            float[] m2 = new float[x2.length];
            for (int i = 0; i < batch; i++) {
                float[] row = series[r.nextInt(series.length)];
                System.arraycopy(row, a1, x1, i * (b1 - a1), b1 - a1);
                System.arraycopy(row, a2, x2, i * (b2 - a2), b2 - a2);
            }
            for (int i = 0; i < m1.length; i++) {
                m1[i] = r.nextDouble() < 0.5 ? 0 : 1;   // binomial timestamp masking, p = 0.5
            }
            for (int i = 0; i < m2.length; i++) {
                m2[i] = r.nextDouble() < 0.5 ? 0 : 1;
            }
            Var e1 = encode(Var.of(Tensor.of(Shape.of(batch, b1 - a1), x1)), Tensor.of(Shape.of(batch, b1 - a1), m1));
            Var e2 = encode(Var.of(Tensor.of(Shape.of(batch, b2 - a2), x2)), Tensor.of(Shape.of(batch, b2 - a2), m2));
            hierarchicalLoss(e1.slice(1, a2 - a1, b1 - a1), e2.slice(1, 0, overlap)).backward();
            opt.step(parameters());
        }
    }

    /**
     * Instance representations: per-timestamp encodings max-pooled over time.
     *
     * @param series {@code [n][length]}
     * @return {@code [n][width]}
     */
    public double[][] instanceRepresentations(float[][] series) {
        int n = series.length;
        int t = series[0].length;
        float[] x = new float[n * t];
        for (int i = 0; i < n; i++) {
            System.arraycopy(series[i], 0, x, i * t, t);
        }
        float[] v = Var.noGrad(() -> encode(Var.of(Tensor.of(Shape.of(n, t), x)), null).max(1, false)).value()
                .toArray();
        double[][] out = new double[n][width];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < width; j++) {
                out[i][j] = v[i * width + j];
            }
        }
        return out;
    }

    /**
     * Last-timestamp representations, for forecasting with a linear head.
     *
     * @param windows {@code [n][length]}
     * @return {@code [n][width]}
     */
    public double[][] lastRepresentations(float[][] windows) {
        int n = windows.length;
        int t = windows[0].length;
        float[] x = new float[n * t];
        for (int i = 0; i < n; i++) {
            System.arraycopy(windows[i], 0, x, i * t, t);
        }
        float[] v = Var.noGrad(() -> encode(Var.of(Tensor.of(Shape.of(n, t), x)), null).slice(1, t - 1, t))
                .value().toArray();
        double[][] out = new double[n][width];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < width; j++) {
                out[i][j] = v[i * width + j];
            }
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inProj, inBias));
        for (Var[] b : blocks) {
            p.addAll(List.of(b));
        }
        p.add(outW);
        p.add(outB);
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(-1), width);
    }

    @Override
    public String toString() {
        return "Ts2Vec(hidden=" + hidden + ", width=" + width + ", depth=" + blocks.size() + ")";
    }
}
