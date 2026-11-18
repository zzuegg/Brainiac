package io.github.zzuegg.brainiac.experimental.events;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Sequence models for irregularly sampled multichannel series on a time grid with missing
 * values ({@link IrregularBatch}). Each model returns a prediction of every channel at every grid
 * time (used for interpolation and extrapolation) and a fixed-size summary (for classification):
 * <ul>
 *   <li>{@link #gruD}: GRU-D (Che et al., Scientific Reports 2018,
 *       <a href="https://arxiv.org/abs/1606.01865">arXiv:1606.01865</a>): a GRU fed
 *       {@code [x_hat, mask]}, where a missing input decays from its last observation towards the
 *       channel mean, {@code x_hat = m x + (1 - m)(g x_last + (1 - g) mean)},
 *       {@code g = exp(-relu(w delta + b))}, and the hidden state decays by
 *       {@code exp(-relu(W delta + b))};</li>
 *   <li>{@link #odeRnn}: ODE-RNN (Rubanova et al., NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1907.03907">arXiv:1907.03907</a>): the state follows a
 *       neural ODE between grid times ({@link OdeSolver}, RK4) and is updated by a GRU where
 *       anything is observed;</li>
 *   <li>{@link #latentOde}: the latent ODE: an ODE-RNN run backwards in time encodes
 *       {@code q(z_0)}, one ODE solution from {@code z_0} decodes every time (a VAE with a KL
 *       term; predictions use the posterior mean);</li>
 *   <li>{@link #mtan}: mTAN (Shukla and Marlin, ICLR 2021,
 *       <a href="https://arxiv.org/abs/2101.10318">arXiv:2101.10318</a>): learned sinusoidal time
 *       embeddings, attention from fixed reference times to each channel's observed times
 *       (softmax only over that channel's observations), an MLP at each reference point, and
 *       attention from the query times back to the reference points.</li>
 * </ul>
 * {@link #fit} trains any of them by masked reconstruction: for interpolation a random 20% of the
 * observed values are hidden from the input and scored; for extrapolation the input stops at 75%
 * of the window and the observed values after it are scored.
 */
public final class IrregularSequenceModels {

    private IrregularSequenceModels() {
    }

    /** A model over irregular batches. */
    public interface SequenceModel {

        /**
         * Runs the model.
         *
         * @param batch the inputs
         * @return {@code {predictions [batch, steps, channels], summary [batch, width], auxiliary loss
         *     (scalar) or null}}
         */
        Var[] forward(IrregularBatch batch);

        /** @return the summary width */
        int summaryWidth();

        /** @return the trainable parameters */
        List<Var> parameters();
    }

    /** A reconstruction task. */
    public enum Task {
        /** Fill in held-out observation times. */
        INTERPOLATION,
        /** Forecast the last quarter of the window. */
        EXTRAPOLATION
    }

    static Var matrix(SplittableRandom r, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, r));
    }

    /** Slices step {@code t} of {@code [B, T, C]} to {@code [B, C]}. */
    static Var step(Var x, int t) {
        int b = x.shape().dim(0);
        int c = x.shape().dim(2);
        return x.slice(1, t, t + 1).reshape(b, c);
    }

    static Var step(Tensor x, int t) {
        return step(Var.of(x), t);
    }

    /** Stacks {@code T} tensors {@code [B, C]} into {@code [B, T, C]}. */
    static Var stack(List<Var> steps) {
        int b = steps.get(0).shape().dim(0);
        int c = steps.get(0).shape().dim(1);
        Var[] parts = new Var[steps.size()];
        for (int i = 0; i < parts.length; i++) {
            parts[i] = steps.get(i).reshape(b, 1, c);
        }
        return Var.concat(1, parts);
    }

    /**
     * GRU-D.
     *
     * @param channels channels
     * @param units hidden units
     * @param seed the initialisation seed
     * @return the model
     */
    public static SequenceModel gruD(int channels, int units, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        GruCell gru = new GruCell(2 * channels, units, r);
        Var gx = Var.param(Tensor.full(Shape.of(channels), 1f));
        Var gxb = Var.param(Tensor.zeros(Shape.of(channels)));
        Var gh = matrix(r, channels, units);
        Var ghb = Var.param(Tensor.zeros(Shape.of(units)));
        Var out = matrix(r, units, channels);
        Var outB = Var.param(Tensor.zeros(Shape.of(channels)));
        List<Var> params = new ArrayList<>(List.of(gx, gxb, gh, ghb, out, outB));
        params.addAll(gru.parameters());
        return new SequenceModel() {
            @Override
            public Var[] forward(IrregularBatch b) {
                Var h = gru.zeroState(b.batch());
                List<Var> preds = new ArrayList<>();
                // delta in units of grid steps, so the learned rates start in a sensible range
                float scale = b.steps() - 1;
                for (int t = 0; t < b.steps(); t++) {
                    Var m = step(b.mask(), t);
                    Var d = step(b.delta(), t).mul(scale);
                    Var x = step(b.values(), t);
                    Var gamma = d.mul(gx).add(gxb).relu().neg().exp();
                    Var decayed = step(b.last(), t).mul(gamma);
                    Var xhat = x.add(Var.of(Tensor.full(m.shape(), 1f)).sub(m).mul(decayed));
                    Var hDecay = d.matmul(gh).add(ghb).relu().neg().exp();
                    h = gru.step(Var.concat(-1, xhat, m), h.mul(hDecay));
                    preds.add(h.matmul(out).add(outB));
                }
                return new Var[] {stack(preds), h, null};
            }

            @Override
            public int summaryWidth() {
                return units;
            }

            @Override
            public List<Var> parameters() {
                return params;
            }
        };
    }

    /** An MLP vector field {@code tanh(z W1 + b1) W2}. */
    private static OdeSolver.Dynamics field(Var w1, Var b1, Var w2) {
        return (t, z) -> z.matmul(w1).add(b1).tanh().matmul(w2);
    }

    /** Runs an ODE-RNN over the grid (optionally backwards); returns the state after each step. */
    private static List<Var> odeRnn(IrregularBatch b, GruCell gru, OdeSolver.Dynamics f, boolean backwards) {
        int steps = b.steps();
        Var h = gru.zeroState(b.batch());
        OdeSolver solver = OdeSolver.rk4(1);
        List<Var> states = new ArrayList<>();
        for (int i = 0; i < steps; i++) {
            int t = backwards ? steps - 1 - i : i;
            if (i > 0) {
                int prev = backwards ? t + 1 : t - 1;
                h = solver.integrate(f, h, (float) b.times()[prev], (float) b.times()[t]);
            }
            Var m = step(b.mask(), t);
            Var any = m.max(-1, true);
            Var updated = gru.step(Var.concat(-1, step(b.values(), t), m), h);
            h = updated.mul(any).add(h.mul(Var.of(Tensor.full(any.shape(), 1f)).sub(any)));
            states.add(h);
        }
        return states;
    }

    /**
     * ODE-RNN.
     *
     * @param channels channels
     * @param units hidden units
     * @param seed the initialisation seed
     * @return the model
     */
    public static SequenceModel odeRnn(int channels, int units, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        GruCell gru = new GruCell(2 * channels, units, r);
        Var w1 = matrix(r, units, units);
        Var b1 = Var.param(Tensor.zeros(Shape.of(units)));
        Var w2 = Var.param(Tensor.zeros(Shape.of(units, units)));
        Var out = matrix(r, units, channels);
        Var outB = Var.param(Tensor.zeros(Shape.of(channels)));
        List<Var> params = new ArrayList<>(List.of(w1, b1, w2, out, outB));
        params.addAll(gru.parameters());
        OdeSolver.Dynamics f = field(w1, b1, w2);
        return new SequenceModel() {
            @Override
            public Var[] forward(IrregularBatch b) {
                List<Var> states = odeRnn(b, gru, f, false);
                List<Var> preds = new ArrayList<>();
                for (Var h : states) {
                    preds.add(h.matmul(out).add(outB));
                }
                return new Var[] {stack(preds), states.get(states.size() - 1), null};
            }

            @Override
            public int summaryWidth() {
                return units;
            }

            @Override
            public List<Var> parameters() {
                return params;
            }
        };
    }

    /**
     * The latent ODE with an ODE-RNN encoder.
     *
     * @param channels channels
     * @param units encoder units
     * @param latent latent width
     * @param beta the KL weight
     * @param seed the initialisation and sampling seed
     * @return the model
     */
    public static SequenceModel latentOde(int channels, int units, int latent, float beta, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        GruCell gru = new GruCell(2 * channels, units, r);
        Var e1 = matrix(r, units, units);
        Var eb = Var.param(Tensor.zeros(Shape.of(units)));
        Var e2 = Var.param(Tensor.zeros(Shape.of(units, units)));
        Var toZ = matrix(r, units, 2 * latent);
        Var d1 = matrix(r, latent, 2 * latent);
        Var db = Var.param(Tensor.zeros(Shape.of(2 * latent)));
        Var d2 = matrix(r, 2 * latent, latent);
        Var out1 = matrix(r, latent, units);
        Var out2 = matrix(r, units, channels);
        List<Var> params = new ArrayList<>(List.of(e1, eb, e2, toZ, d1, db, d2, out1, out2));
        params.addAll(gru.parameters());
        OdeSolver.Dynamics encoderField = field(e1, eb, e2);
        OdeSolver.Dynamics decoderField = field(d1, db, d2);
        return new SequenceModel() {
            @Override
            public Var[] forward(IrregularBatch b) {
                List<Var> enc = odeRnn(b, gru, encoderField, true);
                Var stats = enc.get(enc.size() - 1).matmul(toZ);
                Var mu = stats.slice(1, 0, latent);
                Var logSigma = stats.slice(1, latent, 2 * latent).clamp(-5, 2);
                Var z = mu;
                if (Var.gradEnabled()) {
                    z = mu.add(logSigma.exp().mul(Tensor.normal(mu.shape(), 0, 1, r)));
                }
                OdeSolver solver = OdeSolver.rk4(1);
                List<Var> preds = new ArrayList<>();
                for (int t = 0; t < b.steps(); t++) {
                    if (t > 0) {
                        z = solver.integrate(decoderField, z, (float) b.times()[t - 1], (float) b.times()[t]);
                    }
                    preds.add(z.matmul(out1).tanh().matmul(out2));
                }
                Var kl = mu.square().add(logSigma.mul(2).exp()).sub(1).sub(logSigma.mul(2)).mul(0.5f).sum(-1, false)
                        .mean().mul(beta / (b.steps() * b.channels()));
                return new Var[] {stack(preds), mu, kl};
            }

            @Override
            public int summaryWidth() {
                return latent;
            }

            @Override
            public List<Var> parameters() {
                return params;
            }
        };
    }

    /**
     * mTAN encoder-decoder.
     *
     * @param channels channels
     * @param units hidden units
     * @param references reference time points
     * @param embed time embedding width
     * @param seed the initialisation seed
     * @return the model
     */
    public static SequenceModel mtan(int channels, int units, int references, int embed, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var lin = Var.param(Tensor.uniform(Shape.of(1, 1), -1, 1, r));
        Var linB = Var.param(Tensor.zeros(Shape.of(1)));
        Var freq = Var.param(Tensor.uniform(Shape.of(1, embed - 1), 0, 20, r));
        Var phase = Var.param(Tensor.uniform(Shape.of(embed - 1), 0, (float) (2 * Math.PI), r));
        Var wq = matrix(r, embed, embed);
        Var wk = matrix(r, embed, embed);
        Var m1 = matrix(r, 2 * channels, units);
        Var m1b = Var.param(Tensor.zeros(Shape.of(units)));
        Var m2 = matrix(r, units, units);
        Var dq = matrix(r, embed, embed);
        Var dk = matrix(r, embed, embed);
        Var o1 = matrix(r, units, units);
        Var o2 = matrix(r, units, channels);
        List<Var> params = new ArrayList<>(List.of(lin, linB, freq, phase, wq, wk, m1, m1b, m2, dq, dk, o1, o2));
        double[] refs = new double[references];
        for (int i = 0; i < references; i++) {
            refs[i] = references == 1 ? 0 : (double) i / (references - 1);
        }
        return new SequenceModel() {
            private Var timeEmbedding(double[] t) {
                float[] f = new float[t.length];
                for (int i = 0; i < t.length; i++) {
                    f[i] = (float) t[i];
                }
                Var x = Var.of(Tensor.of(Shape.of(t.length, 1), f));
                return Var.concat(-1, x.matmul(lin).add(linB), x.matmul(freq).add(phase).sin());
            }

            @Override
            public Var[] forward(IrregularBatch b) {
                int bs = b.batch();
                int steps = b.steps();
                int c = b.channels();
                float scale = (float) (1 / Math.sqrt(embed));
                Var et = timeEmbedding(b.times());
                Var er = timeEmbedding(refs);
                Var scores = er.matmul(wq).matmul(et.matmul(wk).transpose()).mul(scale);       // [R, T]
                // per channel masked softmax over observed times: [B, C, R, T]
                Var mask = Var.of(b.mask()).permute(0, 2, 1).reshape(bs, c, 1, steps);
                Var bias = mask.sub(1).mul(1e9f);
                Var s = scores.reshape(1, 1, references, steps).add(bias);
                Var w = s.sub(s.logsumexp(-1, true)).exp().mul(mask.max(-1, true));
                Var vals = Var.of(b.values()).permute(0, 2, 1).reshape(bs, c, steps, 1);
                Var interp = w.matmul(vals).reshape(bs, c, references).permute(0, 2, 1);       // [B, R, C]
                Var all = scores.sub(scores.logsumexp(-1, true)).exp();
                Var density = all.reshape(1, 1, references, steps).mul(mask).sum(-1, false).permute(0, 2, 1);
                Var hidden = Var.concat(-1, interp, density).matmul(m1).add(m1b).relu().matmul(m2).relu();  // [B, R, U]
                Var back = et.matmul(dq).matmul(er.matmul(dk).transpose()).mul(scale);          // [T, R]
                Var a = back.sub(back.logsumexp(-1, true)).exp();
                Var dec = a.reshape(1, steps, references).matmul(hidden);                      // [B, T, U]
                Var pred = dec.matmul(o1).relu().matmul(o2);
                return new Var[] {pred, hidden.mean(1, false), null};
            }

            @Override
            public int summaryWidth() {
                return units;
            }

            @Override
            public List<Var> parameters() {
                return params;
            }
        };
    }

    /**
     * Input and target masks for a task.
     *
     * @param observed the observation mask
     * @param task the task
     * @param r the random source (interpolation hides a random 20% of observed values)
     * @return {@code {input mask, target mask}}
     */
    public static boolean[][][][] split(boolean[][][] observed, Task task, SplittableRandom r) {
        int n = observed.length;
        int steps = observed[0].length;
        int c = observed[0][0].length;
        boolean[][][] in = new boolean[n][steps][c];
        boolean[][][] target = new boolean[n][steps][c];
        int cut = (int) (steps * 0.75);
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    if (!observed[i][t][ch]) {
                        continue;
                    }
                    boolean hide = task == Task.INTERPOLATION ? r.nextDouble() < 0.2 : t >= cut;
                    in[i][t][ch] = !hide;
                    target[i][t][ch] = hide;
                }
            }
        }
        return new boolean[][][][] {in, target};
    }

    private static Tensor maskTensor(boolean[][][] m, int[] rows) {
        int steps = m[0].length;
        int c = m[0][0].length;
        float[] f = new float[rows.length * steps * c];
        for (int i = 0; i < rows.length; i++) {
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    f[(i * steps + t) * c + ch] = m[rows[i]][t][ch] ? 1 : 0;
                }
            }
        }
        return Tensor.of(Shape.of(rows.length, steps, c), f);
    }

    private static Tensor valueTensor(float[][][] v, int[] rows) {
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

    /**
     * Trains by masked reconstruction (squared error on the hidden values, plus any auxiliary
     * loss); a fresh split is drawn every step.
     *
     * @param model the model
     * @param values series values (only observed entries are used)
     * @param observed the observation mask
     * @param times the grid
     * @param task the task
     * @param steps optimiser steps
     * @param batch series per step
     * @param lr the learning rate
     * @param seed the sampling seed
     * @return the final loss
     */
    public static double fit(SequenceModel model, float[][][] values, boolean[][][] observed, double[] times, Task task,
            int steps, int batch, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        double last = 0;
        for (int s = 0; s < steps; s++) {
            int[] rows = new int[batch];
            for (int i = 0; i < batch; i++) {
                rows[i] = r.nextInt(values.length);
            }
            boolean[][][] sub = new boolean[batch][][];
            for (int i = 0; i < batch; i++) {
                sub[i] = observed[rows[i]];
            }
            boolean[][][][] masks = split(sub, task, r);
            int[] local = new int[batch];
            for (int i = 0; i < batch; i++) {
                local[i] = i;
            }
            float[][][] v = new float[batch][][];
            for (int i = 0; i < batch; i++) {
                v[i] = values[rows[i]];
            }
            IrregularBatch in = IrregularBatch.of(v, masks[0], local, times);
            Tensor target = maskTensor(masks[1], local);
            Var[] out = model.forward(in);
            float count = Math.max(1, target.sum().item());
            Var loss = out[0].sub(valueTensor(v, local)).square().mul(target).sum().div(count);
            if (out[2] != null) {
                loss = loss.add(out[2]);
            }
            last = loss.value().item();
            loss.backward();
            opt.step(model.parameters());
        }
        return last;
    }

    /**
     * Predicts every grid value from the given inputs.
     *
     * @param model the model
     * @param values series values
     * @param input what the model may see
     * @param times the grid
     * @return predictions {@code [series][steps][channels]}
     */
    public static float[][][] predict(SequenceModel model, float[][][] values, boolean[][][] input, double[] times) {
        int n = values.length;
        int steps = times.length;
        int c = values[0][0].length;
        float[][][] out = new float[n][steps][c];
        for (int start = 0; start < n; start += 64) {
            int end = Math.min(n, start + 64);
            int[] rows = new int[end - start];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = start + i;
            }
            float[] p = Var.noGrad(() -> model.forward(IrregularBatch.of(values, input, rows, times))[0].value()
                    .toArray());
            for (int i = 0; i < rows.length; i++) {
                for (int t = 0; t < steps; t++) {
                    System.arraycopy(p, (i * steps + t) * c, out[start + i][t], 0, c);
                }
            }
        }
        return out;
    }

    /**
     * Trains the summary with a linear softmax head.
     *
     * @param model the model
     * @param values series
     * @param observed the observation mask
     * @param labels class labels
     * @param classes the class count
     * @param times the grid
     * @param steps optimiser steps
     * @param batch series per step
     * @param lr the learning rate
     * @param seed the seed
     * @return the head {@code [width, classes]}
     */
    public static Var fitClassifier(SequenceModel model, float[][][] values, boolean[][][] observed, int[] labels,
            int classes, double[] times, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var head = matrix(r, model.summaryWidth(), classes);
        List<Var> params = new ArrayList<>(model.parameters());
        params.add(head);
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            int[] rows = new int[batch];
            float[] y = new float[batch];
            for (int i = 0; i < batch; i++) {
                rows[i] = r.nextInt(values.length);
                y[i] = labels[rows[i]];
            }
            Var logits = model.forward(IrregularBatch.of(values, observed, rows, times))[1].matmul(head);
            Var loss = Functional.crossEntropy(logits, Tensor.of(Shape.of(batch), y));
            loss.backward();
            opt.step(params);
        }
        return head;
    }

    /**
     * Classifies with a trained head.
     *
     * @param model the model
     * @param head the head from {@link #fitClassifier}
     * @param values series
     * @param observed the observation mask
     * @param times the grid
     * @return predicted classes
     */
    public static int[] classify(SequenceModel model, Var head, float[][][] values, boolean[][][] observed,
            double[] times) {
        int n = values.length;
        int classes = head.shape().dim(1);
        int[] out = new int[n];
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = i;
        }
        float[] p = Var.noGrad(() -> model.forward(IrregularBatch.of(values, observed, rows, times))[1].matmul(head)
                .value().toArray());
        for (int i = 0; i < n; i++) {
            for (int k = 1; k < classes; k++) {
                if (p[i * classes + k] > p[i * classes + out[i]]) {
                    out[i] = k;
                }
            }
        }
        return out;
    }
}
