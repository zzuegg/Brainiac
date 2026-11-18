package io.github.zzuegg.brainiac.experimental.tabular;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Self-supervised pretraining of a tabular MLP encoder, with the four objectives compared under
 * identical encoders by the 2026 study (<a href="https://arxiv.org/abs/2608.24381">arXiv:2608.24381</a>):
 * <ul>
 *   <li>{@link Objective#VIME} (Yoon et al., NeurIPS 2020): a row is corrupted by replacing a
 *       random share of its features with values drawn from their marginals (another row's
 *       value); the encoder feeds a mask head (which features were replaced, binary
 *       cross-entropy) and a value head (the original row, squared error);</li>
 *   <li>{@link Objective#SCARF} (Bahri et al., ICLR 2022,
 *       <a href="https://arxiv.org/abs/2106.15147">arXiv:2106.15147</a>):
 *       InfoNCE ({@link SelfSupervisedLosses#ntXent}) between the projections of a row and its
 *       corrupted view;</li>
 *   <li>{@link Objective#SUBTAB} (Ucar, Hajiramezanali and Edwards, NeurIPS 2021): the features
 *       are split into overlapping subsets; the encoder sees one subset (the other features set to
 *       zero) and reconstructs the whole row;</li>
 *   <li>{@link Objective#MASK_RECOVER}: the corrupted entries are zeroed and the encoder
 *       reconstructs them (a masked autoencoder on rows).</li>
 * </ul>
 * The corruption is pluggable ({@link Corruptor}); {@link #marginal} is the random per-feature
 * resampling SCARF and VIME use, and {@link StructuredCorruption} corrupts groups of dependent
 * features together. After pretraining, {@link #embed} gives frozen features for a linear probe
 * and {@link #fineTune} trains the encoder with a linear head on the labelled rows. Rows are
 * expected standardised (zero mean, unit variance per feature), so a missing value is imputed by
 * zero.
 */
public final class TabularPretraining {

    /** The pretext objective. */
    public enum Objective {
        /** Mask estimation and value reconstruction. */
        VIME,
        /** Contrastive learning between a row and its corrupted view. */
        SCARF,
        /** Reconstruction of the row from feature subsets. */
        SUBTAB,
        /** Reconstruction of the corrupted (zeroed) entries. */
        MASK_RECOVER
    }

    /**
     * A corrupted batch.
     *
     * @param values the corrupted rows {@code [b][d]}
     * @param mask 1 where a value was replaced, else 0
     */
    public record Corrupted(float[][] values, float[][] mask) {
    }

    /** Corrupts a batch of rows. */
    @FunctionalInterface
    public interface Corruptor {
        /**
         * Corrupts rows.
         *
         * @param rows the clean rows {@code [b][d]}
         * @param random the random source
         * @return the corrupted rows and the mask
         */
        Corrupted corrupt(float[][] rows, SplittableRandom random);
    }

    /**
     * Random marginal corruption: every feature independently, with probability {@code rate},
     * takes the value of a uniformly drawn row of {@code pool}.
     *
     * @param pool the rows to draw replacement values from (the training set)
     * @param rate the corruption rate, such as 0.3 to 0.6
     * @return the corruptor
     */
    public static Corruptor marginal(float[][] pool, double rate) {
        return (rows, r) -> {
            int d = rows[0].length;
            float[][] v = new float[rows.length][d];
            float[][] m = new float[rows.length][d];
            for (int i = 0; i < rows.length; i++) {
                for (int j = 0; j < d; j++) {
                    if (r.nextDouble() < rate) {
                        v[i][j] = pool[r.nextInt(pool.length)][j];
                        m[i][j] = 1;
                    } else {
                        v[i][j] = rows[i][j];
                    }
                }
            }
            return new Corrupted(v, m);
        };
    }

    private final Objective objective;
    private final int features;
    private final int hidden;
    private final Model encoder;
    private final Model headA;
    private final Model headB;
    private final int subsets;
    private Corruptor corruptor;

    /**
     * Creates an encoder ({@code features -> hidden -> hidden}, ReLU) with the objective's heads.
     *
     * @param objective the pretext objective
     * @param features input features
     * @param hidden the representation width
     * @param seed the initialisation seed
     */
    public TabularPretraining(Objective objective, int features, int hidden, long seed) {
        this.objective = objective;
        this.features = features;
        this.hidden = hidden;
        this.subsets = 4;
        this.encoder = Model.sequential().input(features).dense(hidden, Activation.RELU).dense(hidden, Activation.RELU)
                .seed(seed).build();
        this.headA = switch (objective) {
            case SCARF -> Model.sequential().input(hidden).dense(hidden, Activation.RELU).dense(hidden)
                    .seed(seed + 1).build();
            default -> Model.sequential().input(hidden).dense(features).seed(seed + 1).build();
        };
        this.headB = Model.sequential().input(hidden).dense(features).seed(seed + 2).build();
    }

    private TabularPretraining(TabularPretraining source) {
        this.objective = source.objective;
        this.features = source.features;
        this.hidden = source.hidden;
        this.subsets = source.subsets;
        this.encoder = source.encoder.copy(source.encoder.device());
        this.headA = source.headA.copy(source.headA.device());
        this.headB = source.headB.copy(source.headB.device());
        this.corruptor = source.corruptor;
    }

    /**
     * An independent copy (weights copied), so one pretrained encoder can be fine-tuned several
     * times.
     *
     * @return the copy
     */
    public TabularPretraining copy() {
        return new TabularPretraining(this);
    }

    /**
     * Sets the corruption (by default {@link #marginal} on the pretraining rows at rate 0.4).
     *
     * @param c the corruptor
     * @return this
     */
    public TabularPretraining corruption(Corruptor c) {
        this.corruptor = c;
        return this;
    }

    /** @return the encoder, {@code [b, features] -> [b, hidden]} */
    public Model encoder() {
        return encoder;
    }

    /** @return the objective */
    public Objective objective() {
        return objective;
    }

    private List<Var> pretextParameters() {
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(headA.parameters());
        if (objective == Objective.VIME) {
            p.addAll(headB.parameters());
        }
        return p;
    }

    /** The pretext loss of a batch. */
    private Var pretext(float[][] rows, SplittableRandom r, boolean training) {
        int b = rows.length;
        Tensor clean = tensor(rows);
        switch (objective) {
            case VIME -> {
                Corrupted c = corruptor.corrupt(rows, r);
                Var h = encoder.forward(Var.of(tensor(c.values())), training);
                Var maskLogits = headA.forward(h, training);
                Tensor m = tensor(c.mask());
                // binary cross-entropy with logits: softplus(z) - m z
                Var bce = maskLogits.softplus().sub(maskLogits.mul(m)).mean();
                Var mse = headB.forward(h, training).sub(clean).square().mean();
                return bce.add(mse.mul(2f));
            }
            case SCARF -> {
                Corrupted c = corruptor.corrupt(rows, r);
                Var z1 = headA.forward(encoder.forward(Var.of(clean), training), training);
                Var z2 = headA.forward(encoder.forward(Var.of(tensor(c.values())), training), training);
                return SelfSupervisedLosses.ntXent(z1, z2, 0.5f);
            }
            case SUBTAB -> {
                Var total = null;
                for (int s = 0; s < subsets; s++) {
                    float[][] sub = new float[b][features];
                    for (int i = 0; i < b; i++) {
                        for (int j = 0; j < features; j++) {
                            if (inSubset(j, s)) {
                                sub[i][j] = rows[i][j];
                            }
                        }
                    }
                    Var rec = headA.forward(encoder.forward(Var.of(tensor(sub)), training), training);
                    Var l = rec.sub(clean).square().mean();
                    total = total == null ? l : total.add(l);
                }
                return total.mul(1f / subsets);
            }
            case MASK_RECOVER -> {
                Corrupted c = corruptor.corrupt(rows, r);
                float[][] zeroed = new float[b][features];
                for (int i = 0; i < b; i++) {
                    for (int j = 0; j < features; j++) {
                        zeroed[i][j] = c.mask()[i][j] > 0 ? 0 : rows[i][j];
                    }
                }
                Tensor m = tensor(c.mask());
                Var rec = headA.forward(encoder.forward(Var.of(tensor(zeroed)), training), training);
                return rec.sub(clean).square().mul(m).sum().mul(1f / Math.max(1f, sum(c.mask())));
            }
            default -> throw new IllegalStateException(objective.toString());
        }
    }

    /** SubTab's subsets: four contiguous blocks of the columns, each widened by a quarter overlap. */
    private boolean inSubset(int j, int s) {
        double width = features / (double) subsets;
        double lo = s * width - 0.25 * width;
        double hi = (s + 1) * width + 0.25 * width;
        return j >= lo && j < hi;
    }

    /**
     * Pretrains on unlabelled rows with Adam.
     *
     * @param rows the rows {@code [n][features]}
     * @param steps steps
     * @param batch batch size
     * @param learningRate Adam's step
     * @param seed the seed
     * @return this
     */
    public TabularPretraining pretrain(float[][] rows, int steps, int batch, float learningRate, long seed) {
        if (corruptor == null) {
            corruptor = marginal(rows, 0.4);
        }
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = pretextParameters();
        for (int s = 0; s < steps; s++) {
            float[][] b = sample(rows, batch, r);
            long stepSeed = r.nextLong();
            opt.minimize(params, () -> pretext(b, new SplittableRandom(stepSeed), true));
        }
        return this;
    }

    /**
     * The pretext loss on rows, without gradients (to compare corruptions).
     *
     * @param rows the rows
     * @param seed the corruption seed
     * @return the loss
     */
    public double pretextLoss(float[][] rows, long seed) {
        if (corruptor == null) {
            corruptor = marginal(rows, 0.4);
        }
        return Var.noGrad(() -> pretext(rows, new SplittableRandom(seed), false)).value().item();
    }

    /**
     * Frozen representations.
     *
     * @param rows the rows
     * @return {@code [n][hidden]}
     */
    public double[][] embed(float[][] rows) {
        float[] h = Var.noGrad(() -> encoder.forward(Var.of(tensor(rows)), false)).value().toArray();
        double[][] out = new double[rows.length][hidden];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < hidden; j++) {
                out[i][j] = h[i * hidden + j];
            }
        }
        return out;
    }

    /**
     * Fine-tunes the encoder with a new linear head on labelled rows and predicts the test rows.
     * Called on a fresh (not pretrained) instance it is the scratch-MLP baseline.
     *
     * @param x labelled rows
     * @param y their classes
     * @param classes the number of classes
     * @param test rows to predict
     * @param steps training steps
     * @param learningRate Adam's step
     * @param seed the seed
     * @return class probabilities of the test rows {@code [m][classes]}
     */
    public double[][] fineTune(float[][] x, int[] y, int classes, float[][] test, int steps, float learningRate,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var head = Var.param(Tensor.normal(Shape.of(hidden, classes), 0, (float) Math.sqrt(1.0 / hidden), r));
        Var bias = Var.param(Tensor.zeros(Shape.of(classes)));
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.add(head);
        params.add(bias);
        Optimizer opt = Optimizer.adam(learningRate);
        int b = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] xb = new float[b][];
            float[] yb = new float[b];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(x.length);
                xb[i] = x[k];
                yb[i] = y[k];
            }
            Tensor ids = Tensor.of(Shape.of(b), yb);
            opt.minimize(params, () -> Functional.crossEntropy(
                    encoder.forward(Var.of(tensor(xb)), true).matmul(head).add(bias), ids));
        }
        float[] p = Var.noGrad(() -> encoder.forward(Var.of(tensor(test)), false).matmul(head).add(bias)
                .softmax(1)).value().toArray();
        double[][] out = new double[test.length][classes];
        for (int i = 0; i < test.length; i++) {
            for (int c = 0; c < classes; c++) {
                out[i][c] = p[i * classes + c];
            }
        }
        return out;
    }

    private static float[][] sample(float[][] rows, int batch, SplittableRandom r) {
        int b = Math.min(batch, rows.length);
        float[][] out = new float[b][];
        for (int i = 0; i < b; i++) {
            out[i] = rows[r.nextInt(rows.length)];
        }
        return out;
    }

    private static float sum(float[][] m) {
        float s = 0;
        for (float[] row : m) {
            for (float v : row) {
                s += v;
            }
        }
        return s;
    }

    /**
     * Stacks rows into a tensor.
     *
     * @param rows {@code [n][d]}
     * @return {@code [n, d]}
     */
    public static Tensor tensor(float[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] v = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows[i], 0, v, i * d, d);
        }
        return Tensor.of(Shape.of(n, d), v);
    }
}
