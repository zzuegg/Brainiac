package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * DeepHit (Lee, Zame, Yoon and van der Schaar, AAAI 2018,
 * <a href="https://ojs.aaai.org/index.php/AAAI/article/view/11842">AAAI 11842</a>): a network that
 * predicts the joint distribution of the first hitting time and its cause directly, as one softmax
 * over {@code causes x bins} cells plus a final "beyond the horizon" cell. A shared trunk feeds one
 * cause-specific head per competing risk, as in the paper.
 *
 * <p>The loss has two parts:
 * <ul>
 *   <li>the log-likelihood: the probability of the observed (cause, bin) cell for an event, or of
 *       every cell after the censoring bin for a censored unit;</li>
 *   <li>a ranking loss per cause: for comparable pairs (unit {@code i} had cause {@code c} at
 *       {@code t_i < t_j}), {@code exp(-(F_c(t_i | x_i) - F_c(t_i | x_j)) / sigma)}, which pushes
 *       the cumulative incidence of the unit that failed first above the other's.</li>
 * </ul>
 * As a {@link Layer} it maps {@code [batch, features]} to the {@code causes * bins + 1} logits.
 */
public final class DeepHit implements Layer {

    private final double[] cuts;
    private final int causes;
    private final Model trunk;
    private final List<Model> heads = new ArrayList<>();
    private final Var beyond;

    /**
     * Creates the network.
     *
     * @param features input covariates
     * @param hidden hidden width of the trunk and heads
     * @param causes competing causes, at least 1
     * @param cuts bin edges, as for {@link DiscreteHazardNet}
     * @param seed the seed
     */
    public DeepHit(int features, int hidden, int causes, double[] cuts, long seed) {
        this.cuts = cuts.clone();
        this.causes = causes;
        int bins = cuts.length - 1;
        this.trunk = Model.sequential().input(features).dense(hidden, Activation.RELU).seed(seed).build();
        for (int c = 0; c < causes; c++) {
            heads.add(Model.sequential().input(hidden + features).dense(hidden, Activation.RELU).dense(bins)
                    .seed(seed + 1 + c).build());
        }
        this.beyond = Var.param(Tensor.zeros(Shape.of(1, 1)));
    }

    /**
     * @param hidden hidden width
     * @param causes competing causes
     * @param cuts bin edges
     * @return a spec for model builders
     */
    public static LayerSpec spec(int hidden, int causes, double[] cuts) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(causes * (cuts.length - 1) + 1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DeepHit(input.dim(-1), hidden, causes, cuts, random.nextLong());
            }
        };
    }

    private int bins() {
        return cuts.length - 1;
    }

    private int bin(double t) {
        for (int k = 0; k < bins(); k++) {
            if (t < cuts[k + 1]) {
                return k;
            }
        }
        return bins() - 1;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        int b = input.shape().dim(0);
        Var h = Var.concat(1, trunk.forward(input, training), input);
        List<Var> parts = new ArrayList<>();
        for (Model head : heads) {
            parts.add(head.forward(h, training));
        }
        parts.add(beyond.expand(Shape.of(b, 1)));
        return Var.concat(1, parts.toArray(Var[]::new));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(trunk.parameters());
        for (Model head : heads) {
            p.addAll(head.parameters());
        }
        p.add(beyond);
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return causes * bins() + 1;
    }

    /**
     * The DeepHit loss of a batch.
     *
     * @param logits {@code [batch, causes * bins + 1]} from {@link #forward}
     * @param time observed times
     * @param event 0 if censored, else the cause {@code 1..causes}
     * @param alpha the weight of the ranking loss
     * @param sigma the ranking loss temperature, such as 0.1
     * @return the scalar loss
     */
    public Var loss(Var logits, double[] time, int[] event, float alpha, float sigma) {
        int n = time.length;
        int bins = bins();
        int cells = causes * bins + 1;
        float[] mask = new float[n * cells];
        java.util.Arrays.fill(mask, -1e9f);
        int[] binOf = new int[n];
        for (int i = 0; i < n; i++) {
            int j = bin(time[i]);
            binOf[i] = j;
            if (event[i] != 0) {
                mask[i * cells + (event[i] - 1) * bins + j] = 0;
            } else {
                for (int c = 0; c < causes; c++) {
                    for (int k = j + 1; k < bins; k++) {
                        mask[i * cells + c * bins + k] = 0;
                    }
                }
                mask[i * cells + cells - 1] = 0;
            }
        }
        Var logp = logits.sub(logits.logsumexp(1, true));
        Var nll = logp.add(Tensor.of(Shape.of(n, cells), mask)).logsumexp(1, true).mean().neg();
        if (alpha == 0) {
            return nll;
        }
        Var p = logp.exp();
        float[] upper = new float[bins * bins];
        for (int k = 0; k < bins; k++) {
            for (int m = k; m < bins; m++) {
                upper[k * bins + m] = 1;
            }
        }
        float[] pick = new float[bins * n];
        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) {
            pick[binOf[i] * n + i] = 1;
            eye[i * n + i] = 1;
        }
        Tensor cumulate = Tensor.of(Shape.of(bins, bins), upper);
        Tensor select = Tensor.of(Shape.of(bins, n), pick);
        Var rank = null;
        for (int c = 0; c < causes; c++) {
            float[] pairs = new float[n * n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (event[i] != c + 1) {
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    if (time[i] < time[j]) {
                        pairs[j * n + i] = 1;
                        count++;
                    }
                }
            }
            if (count == 0) {
                continue;
            }
            // g[j, i] = F_c(t_i | x_j)
            Var g = p.slice(1, c * bins, (c + 1) * bins).matmul(cumulate).matmul(select);
            Var own = g.mul(Tensor.of(Shape.of(n, n), eye)).sum(0, true);
            Var term = own.sub(g).div(-sigma).exp().mul(Tensor.of(Shape.of(n, n), pairs)).sum().div(count);
            rank = rank == null ? term : rank.add(term);
        }
        return rank == null ? nll : nll.add(rank.mul(alpha));
    }

    /**
     * Trains by Adam on random mini-batches.
     *
     * @param x covariates
     * @param time observed times
     * @param event 0 if censored, else the cause
     * @param steps Adam steps
     * @param batch rows per step
     * @param lr the learning rate
     * @param alpha the ranking loss weight
     * @param seed the batch seed
     * @return this
     */
    public DeepHit fit(double[][] x, double[] time, int[] event, int steps, int batch, float lr, float alpha,
            long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int d = x[0].length;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * d];
            double[] bt = new double[batch];
            int[] be = new int[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(x.length);
                for (int j = 0; j < d; j++) {
                    bx[i * d + j] = (float) x[k][j];
                }
                bt[i] = time[k];
                be[i] = event[k];
            }
            loss(forward(Var.of(Tensor.of(Shape.of(batch, d), bx)), true), bt, be, alpha, 0.1f).backward();
            opt.step(parameters());
        }
        return this;
    }

    /**
     * Predicted cumulative incidence per cause.
     *
     * @param x covariates
     * @param grid times
     * @return {@code [causes][n][grid]} values of {@code F_c(t | x)}, interpolated inside bins
     */
    public double[][][] incidence(double[][] x, double[] grid) {
        int n = x.length;
        int d = x[0].length;
        int bins = bins();
        int cells = causes * bins + 1;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                f[i * d + j] = (float) x[i][j];
            }
        }
        float[] logits = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(n, d), f)), false)).value().toArray();
        double[][][] out = new double[causes][n][grid.length];
        for (int i = 0; i < n; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < cells; k++) {
                max = Math.max(max, logits[i * cells + k]);
            }
            double z = 0;
            double[] p = new double[cells];
            for (int k = 0; k < cells; k++) {
                p[k] = Math.exp(logits[i * cells + k] - max);
                z += p[k];
            }
            for (int c = 0; c < causes; c++) {
                for (int g = 0; g < grid.length; g++) {
                    int j = bin(grid[g]);
                    double s = 0;
                    for (int k = 0; k < j; k++) {
                        s += p[c * bins + k] / z;
                    }
                    double frac = Math.clamp((grid[g] - cuts[j]) / (cuts[j + 1] - cuts[j]), 0, 1);
                    out[c][i][g] = s + frac * p[c * bins + j] / z;
                }
            }
        }
        return out;
    }

    /**
     * Predicted overall survival {@code 1 - sum_c F_c(t | x)}.
     *
     * @param x covariates
     * @param grid times
     * @return {@code [n][grid]}
     */
    public double[][] survival(double[][] x, double[] grid) {
        double[][][] f = incidence(x, grid);
        double[][] s = new double[x.length][grid.length];
        for (int i = 0; i < x.length; i++) {
            for (int g = 0; g < grid.length; g++) {
                double total = 0;
                for (int c = 0; c < causes; c++) {
                    total += f[c][i][g];
                }
                s[i][g] = 1 - total;
            }
        }
        return s;
    }

    @Override
    public String toString() {
        return "DeepHit(" + causes + " causes, " + bins() + " bins)";
    }
}
