package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Training-data attribution: how much each training example moved the model, for finding
 * mislabelled data and for predicting what removing data would do.
 *
 * <ul>
 *   <li><b>Influence functions</b> ({@link #influence}; Koh and Liang 2017,
 *       <a href="https://arxiv.org/abs/1703.04730">arXiv:1703.04730</a>): removing example
 *       {@code i} from {@code n} moves the optimum by about {@code (1/n) H^-1 g_i}, so a
 *       target loss changes by {@code (1/n) g_t^T H^-1 g_i}; the self-influence
 *       {@code g_i^T H^-1 g_i} is large for examples the model can only fit by memorizing them.
 *       The Hessian {@code H} of the mean training loss (plus damping) is formed exactly
 *       from Hessian-vector products ({@link Var#gradients(Var, List, boolean)} with a
 *       graph), so restrict {@code parameters} to a few thousand, such as the last layer.</li>
 *   <li><b>TracIn</b> ({@link TracIn}; Pruthi et al. 2020,
 *       <a href="https://arxiv.org/abs/2002.08484">arXiv:2002.08484</a>): sums, over
 *       checkpoints of a run, the learning rate times the dot product of the example's
 *       gradient with the target's gradient (or with itself, the self-influence), the
 *       first-order loss change each SGD step on it caused.</li>
 * </ul>
 *
 * <p>Both take a loss as a function of a batch (inputs, targets) to a mean-loss {@link Var}
 * over the current parameters, and form per-example gradients one example at a time.
 */
public final class TrainingDataAttribution {

    private final List<Var> parameters;
    private final BiFunction<Tensor, Tensor, Var> loss;

    /**
     * Creates the attribution.
     *
     * @param parameters the parameters to attribute through
     * @param loss the mean loss of a batch of inputs and targets, at the current parameters
     */
    public TrainingDataAttribution(List<Var> parameters, BiFunction<Tensor, Tensor, Var> loss) {
        this.parameters = List.copyOf(parameters);
        this.loss = loss;
    }

    private int size() {
        int p = 0;
        for (Var v : parameters) {
            p += (int) v.value().shape().size();
        }
        return p;
    }

    private double[] flat(List<Var> grads) {
        double[] out = new double[size()];
        int o = 0;
        for (Var g : grads) {
            double[] v = g.value().toDoubleArray();
            System.arraycopy(v, 0, out, o, v.length);
            o += v.length;
        }
        return out;
    }

    /**
     * The gradient of the mean loss of a batch, flattened over the parameters.
     *
     * @param x inputs
     * @param y targets
     * @return the gradient
     */
    public double[] gradient(Tensor x, Tensor y) {
        return flat(Var.gradients(loss.apply(x, y), parameters, false));
    }

    /**
     * Each example's loss gradient.
     *
     * @param x inputs, examples along axis 0
     * @param y targets, examples along axis 0
     * @return {@code [n][parameters]}
     */
    public double[][] perExampleGradients(Tensor x, Tensor y) {
        int n = x.shape().dim(0);
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = gradient(x.slice(0, i, i + 1), y.slice(0, i, i + 1));
        }
        return out;
    }

    /**
     * The Hessian of the mean loss over a batch, plus {@code damping} on the diagonal, from
     * one Hessian-vector product per parameter.
     *
     * @param x inputs
     * @param y targets
     * @param damping added to the diagonal
     * @return {@code [p, p]}, float64
     */
    public Tensor hessian(Tensor x, Tensor y, double damping) {
        int p = size();
        List<Var> g = Var.gradients(loss.apply(x, y), parameters, true);
        double[] h = new double[p * p];
        int row = 0;
        for (int k = 0; k < parameters.size(); k++) {
            int len = (int) parameters.get(k).value().shape().size();
            for (int e = 0; e < len; e++) {
                // d/dw (g_k[e]) is row `row` of H
                Var ge = g.get(k).reshape(len).slice(0, e, e + 1).sum();
                double[] hv = flat(Var.gradients(ge, parameters, false));
                System.arraycopy(hv, 0, h, row * p, p);
                h[row * p + row] += damping;
                row++;
            }
        }
        // symmetrize away round-off
        for (int i = 0; i < p; i++) {
            for (int j = i + 1; j < p; j++) {
                double m = 0.5 * (h[i * p + j] + h[j * p + i]);
                h[i * p + j] = m;
                h[j * p + i] = m;
            }
        }
        return Tensor.of(Shape.of(p, p), h, DType.FLOAT64, Device.cpu());
    }

    /**
     * Influence scores.
     *
     * @param removal the predicted change of the target loss when each example is removed,
     *        {@code (1/n) g_t^T H^-1 g_i}: positive for examples that help the target
     * @param self each example's self-influence {@code g_i^T H^-1 g_i}
     */
    public record Influence(double[] removal, double[] self) {
    }

    /**
     * Influence-function scores of every training example on a target set.
     *
     * @param trainX training inputs
     * @param trainY training targets
     * @param targetX target inputs (a validation set)
     * @param targetY target targets
     * @param damping the Hessian damping
     * @return the scores
     */
    public Influence influence(Tensor trainX, Tensor trainY, Tensor targetX, Tensor targetY, double damping) {
        int n = trainX.shape().dim(0);
        int p = size();
        Tensor h = hessian(trainX, trainY, damping);
        Tensor l = Linalg.cholesky(h);
        double[] gt = gradient(targetX, targetY);
        Tensor gtCol = Tensor.of(Shape.of(p, 1), gt, DType.FLOAT64, Device.cpu());
        double[] s = Linalg.solveTriangular(l.transpose(), Linalg.solveTriangular(l, gtCol, true), false)
                .toDoubleArray();
        double[][] g = perExampleGradients(trainX, trainY);
        double[] flatG = new double[n * p];
        for (int i = 0; i < n; i++) {
            System.arraycopy(g[i], 0, flatG, i * p, p);
        }
        // L^-1 g_i for all i at once: self-influence is |L^-1 g_i|^2
        Tensor gt64 = Tensor.of(Shape.of(n, p), flatG, DType.FLOAT64, Device.cpu()).transpose();
        double[] z = Linalg.solveTriangular(l, gt64, true).toDoubleArray();
        double[] removal = new double[n];
        double[] self = new double[n];
        for (int i = 0; i < n; i++) {
            double dot = 0;
            for (int j = 0; j < p; j++) {
                dot += s[j] * g[i][j];
            }
            removal[i] = dot / n;
            double sq = 0;
            for (int j = 0; j < p; j++) {
                sq += z[j * n + i] * z[j * n + i];
            }
            self[i] = sq;
        }
        return new Influence(removal, self);
    }

    /**
     * TracIn: per-example gradient dot products summed over checkpoints. Call
     * {@link #checkpoint} at a few points of a training run (each epoch's end, say).
     */
    public static final class TracIn {

        private final TrainingDataAttribution attribution;
        private final Tensor trainX;
        private final Tensor trainY;
        private final Tensor targetX;
        private final Tensor targetY;
        private final double[] removal;
        private final double[] self;
        private final List<Double> rates = new ArrayList<>();

        /**
         * Creates the accumulator.
         *
         * @param attribution the parameters and loss
         * @param trainX training inputs
         * @param trainY training targets
         * @param targetX target inputs, or null for self-influence only
         * @param targetY target targets
         */
        public TracIn(TrainingDataAttribution attribution, Tensor trainX, Tensor trainY, Tensor targetX,
                Tensor targetY) {
            this.attribution = attribution;
            this.trainX = trainX;
            this.trainY = trainY;
            this.targetX = targetX;
            this.targetY = targetY;
            int n = trainX.shape().dim(0);
            this.removal = new double[n];
            this.self = new double[n];
        }

        /**
         * Adds the current parameters as a checkpoint.
         *
         * @param learningRate the learning rate around this checkpoint
         * @param stepsPerExample how many SGD steps each example took part in since the last
         *        checkpoint, divided by the batch size (the weight of one example in those steps)
         */
        public void checkpoint(double learningRate, double stepsPerExample) {
            double[][] g = attribution.perExampleGradients(trainX, trainY);
            double[] gt = targetX == null ? null : attribution.gradient(targetX, targetY);
            double w = learningRate * stepsPerExample;
            for (int i = 0; i < g.length; i++) {
                double sq = 0;
                double dot = 0;
                for (int j = 0; j < g[i].length; j++) {
                    sq += g[i][j] * g[i][j];
                    if (gt != null) {
                        dot += g[i][j] * gt[j];
                    }
                }
                self[i] += w * sq;
                removal[i] += w * dot;
            }
            rates.add(learningRate);
        }

        /** @return how many checkpoints were added */
        public int checkpoints() {
            return rates.size();
        }

        /**
         * @return the scores: {@code removal} is the target-loss increase each example's steps
         *         prevented (positive for helpful examples), {@code self} the summed self-influence
         */
        public Influence scores() {
            return new Influence(removal.clone(), self.clone());
        }
    }
}
