package io.github.zzuegg.brainiac.experimental.events;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep imputers for multichannel series with missing values on a grid:
 * <ul>
 *   <li>{@link Kind#BRITS} (Cao et al., NeurIPS 2018,
 *       <a href="https://arxiv.org/abs/1805.10572">arXiv:1805.10572</a>): in each direction a
 *       recurrent imputer whose hidden state decays with the time since the last observation; at
 *       every step a history estimate {@code W h} and a feature estimate from the other channels
 *       (a zero-diagonal regression) are blended by learned weights, the complement feeds a GRU,
 *       and all estimates are scored against the observed values; the two directions are averaged
 *       and pushed to agree;</li>
 *   <li>{@link Kind#SAITS} (Du et al., ESWA 2023,
 *       <a href="https://arxiv.org/abs/2202.08516">arXiv:2202.08516</a>): two diagonally-masked
 *       self-attention blocks (a step cannot attend to itself), the second refining the first's
 *       imputation, blended by weights computed from the attention map and the mask; trained with
 *       the observed-reconstruction plus masked-imputation losses;</li>
 *   <li>{@link Kind#CSDI} (Tashiro et al., NeurIPS 2021,
 *       <a href="https://arxiv.org/abs/2107.03502">arXiv:2107.03502</a>): a conditional DDPM over
 *       the missing entries; the denoiser sees the noisy targets, the observed values and the mask
 *       per (time, channel) token, with diffusion-step, time and channel embeddings, and alternates
 *       temporal and feature attention; trained by self-supervised random masking of observed
 *       values; gives samples, so probabilistic imputations.</li>
 * </ul>
 */
public final class TimeSeriesImputer {

    /** The imputer type. */
    public enum Kind {
        /** Bidirectional recurrent imputation. */
        BRITS,
        /** Diagonally-masked self-attention. */
        SAITS,
        /** Conditional score-based diffusion. */
        CSDI
    }

    private final Kind kind;
    private final int channels;
    private final int dim;
    private final List<Var> params = new ArrayList<>();
    private final List<Var> weights = new ArrayList<>();
    private final GruCell forwardGru;
    private final GruCell backwardGru;
    private final int diffusionSteps;
    private final double[] beta;
    private final double[] alphaBar;
    private final SplittableRandom random;

    private TimeSeriesImputer(Kind kind, int channels, int length, int dim, int diffusionSteps, long seed) {
        this.kind = kind;
        this.channels = channels;
        this.dim = dim;
        this.diffusionSteps = diffusionSteps;
        this.random = new SplittableRandom(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int c = channels;
        switch (kind) {
            case BRITS -> {
                forwardGru = new GruCell(2 * c, dim, r);
                backwardGru = new GruCell(2 * c, dim, r);
                params.addAll(forwardGru.parameters());
                params.addAll(backwardGru.parameters());
                for (int dir = 0; dir < 2; dir++) {
                    add(matrix(r, c, dim));                 // hidden decay
                    add(zeros(dim));
                    add(matrix(r, dim, c));                 // history regression
                    add(zeros(c));
                    add(matrix(r, c, c));                   // feature regression (diagonal masked)
                    add(zeros(c));
                    add(Var.param(Tensor.full(Shape.of(c), 1f)));   // input decay
                    add(zeros(c));
                    add(matrix(r, 2 * c, c));               // blend weights
                    add(zeros(c));
                }
            }
            case SAITS -> {
                forwardGru = null;
                backwardGru = null;
                for (int block = 0; block < 2; block++) {
                    add(matrix(r, 2 * c, dim));             // embed
                    add(zeros(dim));
                    add(matrix(r, dim, dim));               // q
                    add(matrix(r, dim, dim));               // k
                    add(matrix(r, dim, dim));               // v
                    add(matrix(r, dim, dim));               // out
                    add(matrix(r, dim, 2 * dim));           // ffn
                    add(matrix(r, 2 * dim, dim));
                    add(matrix(r, dim, c));                 // read-out
                    add(zeros(c));
                }
                add(matrix(r, length + c, c));              // blend weights from mask and attention
            }
            default -> {
                forwardGru = null;
                backwardGru = null;
                add(matrix(r, 3, dim));                     // token input
                add(zeros(dim));
                add(matrix(r, dim, dim));                   // step embedding MLP
                add(Var.param(Tensor.uniform(Shape.of(c, dim), -0.5f, 0.5f, r)));   // channel embedding
                for (int layer = 0; layer < 2; layer++) {
                    for (int a = 0; a < 2; a++) {
                        add(matrix(r, dim, dim));
                        add(matrix(r, dim, dim));
                        add(matrix(r, dim, dim));
                        add(matrix(r, dim, dim));
                    }
                    add(matrix(r, dim, dim));
                }
                add(matrix(r, dim, dim));
                add(matrix(r, dim, 1));
            }
        }
        beta = new double[diffusionSteps + 1];
        alphaBar = new double[diffusionSteps + 1];
        alphaBar[0] = 1;
        for (int k = 1; k <= diffusionSteps; k++) {
            double s = Math.sqrt(1e-4) + (Math.sqrt(0.5) - Math.sqrt(1e-4)) * (k - 1) / Math.max(1, diffusionSteps - 1);
            beta[k] = s * s;
            alphaBar[k] = alphaBar[k - 1] * (1 - beta[k]);
        }
    }

    private void add(Var v) {
        weights.add(v);
        params.add(v);
    }

    private static Var matrix(SplittableRandom r, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, r));
    }

    private static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /**
     * BRITS.
     *
     * @param channels channels
     * @param units hidden units per direction
     * @param seed the seed
     * @return the imputer
     */
    public static TimeSeriesImputer brits(int channels, int units, long seed) {
        return new TimeSeriesImputer(Kind.BRITS, channels, 0, units, 0, seed);
    }

    /**
     * SAITS.
     *
     * @param channels channels
     * @param length the grid length
     * @param dim the model width
     * @param seed the seed
     * @return the imputer
     */
    public static TimeSeriesImputer saits(int channels, int length, int dim, long seed) {
        return new TimeSeriesImputer(Kind.SAITS, channels, length, dim, 0, seed);
    }

    /**
     * CSDI.
     *
     * @param channels channels
     * @param dim the token width
     * @param diffusionSteps diffusion steps (the paper uses 50)
     * @param seed the seed
     * @return the imputer
     */
    public static TimeSeriesImputer csdi(int channels, int dim, int diffusionSteps, long seed) {
        return new TimeSeriesImputer(Kind.CSDI, channels, 0, dim, diffusionSteps, seed);
    }

    /** @return the imputer type */
    public Kind kind() {
        return kind;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    // ---------------------------------------------------------------- BRITS

    private static float[][][] reverse(float[][][] v) {
        float[][][] out = new float[v.length][][];
        for (int i = 0; i < v.length; i++) {
            out[i] = new float[v[i].length][];
            for (int t = 0; t < v[i].length; t++) {
                out[i][t] = v[i][v[i].length - 1 - t];
            }
        }
        return out;
    }

    private static boolean[][][] reverse(boolean[][][] v) {
        boolean[][][] out = new boolean[v.length][][];
        for (int i = 0; i < v.length; i++) {
            out[i] = new boolean[v[i].length][];
            for (int t = 0; t < v[i].length; t++) {
                out[i][t] = v[i][v[i].length - 1 - t];
            }
        }
        return out;
    }

    /** One RITS direction: {@code {final estimates [B,T,C], loss}}. */
    private Var[] rits(IrregularBatch b, int dir) {
        GruCell gru = dir == 0 ? forwardGru : backwardGru;
        int o = dir * 10;
        Var wGh = weights.get(o);
        Var bGh = weights.get(o + 1);
        Var wX = weights.get(o + 2);
        Var bX = weights.get(o + 3);
        float[] off = new float[channels * channels];
        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < channels; j++) {
                off[i * channels + j] = i == j ? 0 : 1;
            }
        }
        Var wZ = weights.get(o + 4).mul(Tensor.of(Shape.of(channels, channels), off));
        Var bZ = weights.get(o + 5);
        Var gx = weights.get(o + 6);
        Var gxb = weights.get(o + 7);
        Var wB = weights.get(o + 8);
        Var bB = weights.get(o + 9);
        Var h = gru.zeroState(b.batch());
        List<Var> estimates = new ArrayList<>();
        Var loss = null;
        float scale = b.steps() - 1;
        float count = Math.max(1, b.mask().sum().item());
        for (int t = 0; t < b.steps(); t++) {
            Var x = IrregularSequenceModels.step(b.values(), t);
            Var m = IrregularSequenceModels.step(b.mask(), t);
            Var d = IrregularSequenceModels.step(b.delta(), t).mul(scale);
            Var notM = Var.of(Tensor.full(m.shape(), 1f)).sub(m);
            h = h.mul(d.matmul(wGh).add(bGh).relu().neg().exp());
            Var xh = h.matmul(wX).add(bX);
            Var xc = x.add(notM.mul(xh));
            Var zh = xc.matmul(wZ).add(bZ);
            Var gammaX = d.mul(gx).add(gxb).relu().neg().exp();
            Var blend = Var.concat(-1, gammaX, m).matmul(wB).add(bB).sigmoid();
            Var ch = blend.mul(zh).add(Var.of(Tensor.full(m.shape(), 1f)).sub(blend).mul(xh));
            Var cc = x.add(notM.mul(ch));
            Var err = xh.sub(x).abs().add(zh.sub(x).abs()).add(ch.sub(x).abs()).mul(m).sum();
            loss = loss == null ? err : loss.add(err);
            estimates.add(ch);
            h = gru.step(Var.concat(-1, cc, m), h);
        }
        return new Var[] {IrregularSequenceModels.stack(estimates), loss.div(count)};
    }

    private Var[] brits(float[][][] values, boolean[][][] observed, int[] rows, double[] times) {
        IrregularBatch f = IrregularBatch.of(values, observed, rows, times);
        double[] rt = new double[times.length];
        for (int i = 0; i < rt.length; i++) {
            rt[i] = times[times.length - 1] - times[times.length - 1 - i];
        }
        IrregularBatch bw = IrregularBatch.of(reverse(values), reverse(observed), rows, rt);
        Var[] a = rits(f, 0);
        Var[] b = rits(bw, 1);
        Var back = b[0].flip(1);
        Var est = a[0].add(back).mul(0.5f);
        Var consistency = a[0].sub(back).abs().mean().mul(0.1f);
        return new Var[] {est, a[1].add(b[1]).add(consistency)};
    }

    // ---------------------------------------------------------------- SAITS

    private Var[] dmsa(Var input, int block, int steps) {
        int o = block * 10;
        int b = input.shape().dim(0);
        Var x = input.matmul(weights.get(o)).add(weights.get(o + 1))
                .add(Var.of(positions(steps, dim)).reshape(1, steps, dim));
        float[] diag = new float[steps * steps];
        for (int i = 0; i < steps; i++) {
            diag[i * steps + i] = -1e9f;
        }
        Var q = x.matmul(weights.get(o + 2));
        Var k = x.matmul(weights.get(o + 3));
        Var v = x.matmul(weights.get(o + 4));
        Var s = q.matmul(k.permute(0, 2, 1)).mul((float) (1 / Math.sqrt(dim)))
                .add(Tensor.of(Shape.of(1, steps, steps), diag));
        Var a = s.sub(s.logsumexp(-1, true)).exp();
        x = norm(x.add(a.matmul(v).matmul(weights.get(o + 5))));
        x = norm(x.add(x.matmul(weights.get(o + 6)).relu().matmul(weights.get(o + 7))));
        Var out = x.matmul(weights.get(o + 8)).add(weights.get(o + 9));
        return new Var[] {out, a.reshape(b, steps, steps)};
    }

    private static Var norm(Var x) {
        return x.div(x.square().mean(-1, true).add(1e-5f).sqrt());
    }

    private static Tensor positions(int steps, int dim) {
        float[] f = new float[steps * dim];
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < dim / 2; i++) {
                double a = t / Math.pow(100, 2.0 * i / dim);
                f[t * dim + 2 * i] = (float) Math.sin(a);
                f[t * dim + 2 * i + 1] = (float) Math.cos(a);
            }
        }
        return Tensor.of(Shape.of(steps, dim), f);
    }

    /** {@code {X1, X2, X3}} for inputs X (0 where missing) and mask M. */
    private Var[] saits(Var x, Var m, int steps) {
        Var etaWeight = weights.get(20);
        Var notM = Var.of(Tensor.full(m.shape(), 1f)).sub(m);
        Var[] first = dmsa(Var.concat(-1, x, m), 0, steps);
        Var x1 = first[0];
        Var xr = x.add(notM.mul(x1));
        Var[] second = dmsa(Var.concat(-1, xr, m), 1, steps);
        Var x2 = second[0];
        Var eta = Var.concat(-1, m, second[1]).matmul(etaWeight).sigmoid();
        Var x3 = x1.mul(Var.of(Tensor.full(eta.shape(), 1f)).sub(eta)).add(x2.mul(eta));
        return new Var[] {x1, x2, x3};
    }

    // ---------------------------------------------------------------- CSDI

    private Var attend(Var x, int base) {
        // x [groups, length, d]
        int g = x.shape().dim(0);
        int len = x.shape().dim(1);
        Var q = x.matmul(weights.get(base)).reshape(g, 1, len, dim);
        Var k = x.matmul(weights.get(base + 1)).reshape(g, 1, len, dim);
        Var v = x.matmul(weights.get(base + 2)).reshape(g, 1, len, dim);
        Var a = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(dim))).reshape(g, len, dim);
        return norm(x.add(a.matmul(weights.get(base + 3))));
    }

    /** Predicted noise {@code [B, T, C]}. */
    private Var denoise(Var noisy, Tensor cond, Tensor condMask, int[] step, int steps) {
        int b = noisy.shape().dim(0);
        int c = channels;
        Var tokens = Var.concat(-1, noisy.reshape(b, steps, c, 1), Var.of(cond).reshape(b, steps, c, 1),
                Var.of(condMask).reshape(b, steps, c, 1));
        Var x = tokens.matmul(weights.get(0)).add(weights.get(1));                     // [B, T, C, d]
        float[] se = new float[b * dim];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < dim / 2; j++) {
                double a = step[i] / Math.pow(100, 2.0 * j / dim);
                se[i * dim + 2 * j] = (float) Math.sin(a);
                se[i * dim + 2 * j + 1] = (float) Math.cos(a);
            }
        }
        Var stepEmb = Var.of(Tensor.of(Shape.of(b, dim), se)).matmul(weights.get(2)).relu().reshape(b, 1, 1, dim);
        Var pos = Var.of(positions(steps, dim)).reshape(1, steps, 1, dim);
        Var chan = weights.get(3).reshape(1, 1, c, dim);
        x = x.add(stepEmb).add(pos).add(chan);
        int w = 4;
        for (int layer = 0; layer < 2; layer++) {
            Var temporal = attend(x.permute(0, 2, 1, 3).reshape(b * c, steps, dim), w);
            x = temporal.reshape(b, c, steps, dim).permute(0, 2, 1, 3);
            Var feature = attend(x.reshape(b * steps, c, dim), w + 4);
            x = feature.reshape(b, steps, c, dim);
            x = x.add(x.matmul(weights.get(w + 8)).relu());
            w += 9;
        }
        return x.matmul(weights.get(w)).relu().matmul(weights.get(w + 1)).reshape(b, steps, c);
    }

    // ---------------------------------------------------------------- training and imputation

    private static Tensor tensor(float[][][] v, int[] rows) {
        int steps = v[0].length;
        int c = v[0][0].length;
        float[] f = new float[rows.length * steps * c];
        for (int i = 0; i < rows.length; i++) {
            for (int t = 0; t < steps; t++) {
                System.arraycopy(v[rows[i]][t], 0, f, (i * steps + t) * c, c);
            }
        }
        return Tensor.of(Shape.of(rows.length, steps, c), f);
    }

    private static Tensor tensor(boolean[][][] v, int[] rows) {
        int steps = v[rows[0]].length;
        int c = v[rows[0]][0].length;
        float[] f = new float[rows.length * steps * c];
        for (int i = 0; i < rows.length; i++) {
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    f[(i * steps + t) * c + ch] = v[rows[i]][t][ch] ? 1 : 0;
                }
            }
        }
        return Tensor.of(Shape.of(rows.length, steps, c), f);
    }

    /**
     * Trains on series with missing values, using only observed entries.
     *
     * @param values values (unobserved entries are ignored)
     * @param observed the observation mask
     * @param times the grid
     * @param steps optimiser steps
     * @param batch series per step
     * @param lr the learning rate
     * @param seed the sampling seed
     * @return the final loss
     */
    public double fit(float[][][] values, boolean[][][] observed, double[] times, int steps, int batch, float lr,
            long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int len = times.length;
        int c = channels;
        double last = 0;
        for (int s = 0; s < steps; s++) {
            int[] rows = new int[batch];
            for (int i = 0; i < batch; i++) {
                rows[i] = r.nextInt(values.length);
            }
            // self-supervised hiding of observed values (SAITS: 20%; CSDI: a random share per series)
            boolean[][][] input = new boolean[values.length][][];
            boolean[][][] hidden = new boolean[values.length][][];
            for (int i = 0; i < batch; i++) {
                int row = rows[i];
                double ratio = kind == Kind.CSDI ? 0.1 + 0.8 * r.nextDouble() : 0.2;
                input[row] = new boolean[len][c];
                hidden[row] = new boolean[len][c];
                for (int t = 0; t < len; t++) {
                    for (int ch = 0; ch < c; ch++) {
                        boolean hide = observed[row][t][ch] && r.nextDouble() < ratio;
                        input[row][t][ch] = observed[row][t][ch] && !hide;
                        hidden[row][t][ch] = hide;
                    }
                }
            }
            Tensor x = tensor(values, rows);
            Tensor inM = tensor(input, rows);
            Tensor hidM = tensor(hidden, rows);
            Var loss;
            switch (kind) {
                case BRITS -> loss = brits(values, observed, rows, times)[1];
                case SAITS -> {
                    Var[] out = saits(Var.of(x.mul(inM)), Var.of(inM), len);
                    float ic = Math.max(1, inM.sum().item());
                    float hc = Math.max(1, hidM.sum().item());
                    Var ort = null;
                    for (Var o : out) {
                        Var e = o.sub(x).abs().mul(inM).sum().div(ic);
                        ort = ort == null ? e : ort.add(e);
                    }
                    Var mit = out[2].sub(x).abs().mul(hidM).sum().div(hc);
                    loss = ort.add(mit);
                }
                default -> {
                    int[] k = new int[batch];
                    float[] scaleX = new float[batch];
                    float[] scaleE = new float[batch];
                    for (int i = 0; i < batch; i++) {
                        k[i] = 1 + r.nextInt(diffusionSteps);
                        scaleX[i] = (float) Math.sqrt(alphaBar[k[i]]);
                        scaleE[i] = (float) Math.sqrt(1 - alphaBar[k[i]]);
                    }
                    Tensor eps = Tensor.normal(x.shape(), 0, 1, random);
                    Tensor sx = Tensor.of(Shape.of(batch, 1, 1), scaleX);
                    Tensor se = Tensor.of(Shape.of(batch, 1, 1), scaleE);
                    Tensor noisy = x.mul(sx).add(eps.mul(se)).mul(hidM);
                    Var pred = denoise(Var.of(noisy), x.mul(inM), inM, k, len);
                    float hc = Math.max(1, hidM.sum().item());
                    loss = pred.sub(eps).square().mul(hidM).sum().div(hc);
                }
            }
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Point imputations (CSDI: the median of {@code samples} draws).
     *
     * @param values values (unobserved entries ignored)
     * @param observed what may be used
     * @param times the grid
     * @param samples CSDI samples
     * @return completed series: observed values kept, the rest imputed
     */
    public float[][][] impute(float[][][] values, boolean[][][] observed, double[] times, int samples) {
        if (kind == Kind.CSDI) {
            float[][][][] s = sample(values, observed, times, samples);
            int n = values.length;
            int len = times.length;
            float[][][] out = new float[n][len][channels];
            float[] buf = new float[samples];
            for (int i = 0; i < n; i++) {
                for (int t = 0; t < len; t++) {
                    for (int ch = 0; ch < channels; ch++) {
                        for (int k = 0; k < samples; k++) {
                            buf[k] = s[k][i][t][ch];
                        }
                        java.util.Arrays.sort(buf);
                        out[i][t][ch] = samples % 2 == 1 ? buf[samples / 2] : (buf[samples / 2 - 1] + buf[samples / 2]) / 2;
                    }
                }
            }
            return out;
        }
        int n = values.length;
        int len = times.length;
        float[][][] out = new float[n][len][channels];
        for (int start = 0; start < n; start += 64) {
            int end = Math.min(n, start + 64);
            int[] rows = new int[end - start];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = start + i;
            }
            Tensor x = tensor(values, rows);
            Tensor m = tensor(observed, rows);
            float[] p = Var.noGrad(() -> {
                Var est = kind == Kind.BRITS ? brits(values, observed, rows, times)[0]
                        : saits(Var.of(x.mul(m)), Var.of(m), len)[2];
                return est.value().toArray();
            });
            float[] xv = x.toArray();
            float[] mv = m.toArray();
            for (int i = 0; i < rows.length; i++) {
                for (int t = 0; t < len; t++) {
                    for (int ch = 0; ch < channels; ch++) {
                        int idx = (i * len + t) * channels + ch;
                        out[start + i][t][ch] = mv[idx] > 0 ? xv[idx] : p[idx];
                    }
                }
            }
        }
        return out;
    }

    /**
     * CSDI samples of the completed series by ancestral DDPM sampling.
     *
     * @param values values (unobserved entries ignored)
     * @param observed what may be used
     * @param times the grid
     * @param samples draws
     * @return {@code [samples][series][steps][channels]}
     */
    public float[][][][] sample(float[][][] values, boolean[][][] observed, double[] times, int samples) {
        if (kind != Kind.CSDI) {
            float[][][] p = impute(values, observed, times, 1);
            float[][][][] out = new float[samples][][][];
            java.util.Arrays.fill(out, p);
            return out;
        }
        int n = values.length;
        int len = times.length;
        int c = channels;
        float[][][][] out = new float[samples][n][len][c];
        int chunk = Math.max(1, 256 / samples);
        for (int start = 0; start < n; start += chunk) {
            int end = Math.min(n, start + chunk);
            int rowsN = (end - start) * samples;
            int[] rows = new int[rowsN];
            for (int s = 0; s < samples; s++) {
                for (int i = start; i < end; i++) {
                    rows[s * (end - start) + i - start] = i;
                }
            }
            Tensor x = tensor(values, rows);
            Tensor m = tensor(observed, rows);
            Tensor target = Tensor.full(m.shape(), 1f).sub(m);
            Tensor cond = x.mul(m);
            Tensor z = Tensor.normal(x.shape(), 0, 1, random).mul(target);
            for (int k = diffusionSteps; k >= 1; k--) {
                int[] ks = new int[rowsN];
                java.util.Arrays.fill(ks, k);
                Tensor zk = z;
                Tensor eps = Var.noGrad(() -> denoise(Var.of(zk), cond, m, ks, len).value());
                double a = 1 - beta[k];
                Tensor mean = z.sub(eps.mul((float) (beta[k] / Math.sqrt(1 - alphaBar[k])))).mul((float) (1 / Math.sqrt(a)));
                if (k > 1) {
                    double sigma = Math.sqrt((1 - alphaBar[k - 1]) / (1 - alphaBar[k]) * beta[k]);
                    mean = mean.add(Tensor.normal(x.shape(), 0, 1, random).mul((float) sigma));
                }
                z = mean.mul(target);
            }
            float[] zv = z.add(cond).toArray();
            for (int s = 0; s < samples; s++) {
                for (int i = start; i < end; i++) {
                    int row = s * (end - start) + i - start;
                    for (int t = 0; t < len; t++) {
                        System.arraycopy(zv, (row * len + t) * c, out[s][i][t], 0, c);
                    }
                }
            }
        }
        return out;
    }
}
