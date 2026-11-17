package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.EvolutionStrategy;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * An optimizer whose update rule is a tiny neural network, meta-trained instead of
 * hand-designed. For every parameter element it reads a few features of the gradient
 * history (the gradient and its momentum, both normalised by a running RMS, and the sign
 * agreement of the two) and a small MLP turns them into the step. The MLP's weights are
 * meta-trained with an {@link EvolutionStrategy} on short training runs of toy problems
 * ({@link Task}), which needs no gradients through the unrolled training (the hard part of
 * learned optimizers) and directly optimises what we care about, the loss after training.
 *
 * <p>It is a plain {@link Optimizer}, so a meta-trained rule drops into a {@code Trainer}
 * like Adam does.
 *
 * <p>Andrychowicz et al. 2016, "Learning to learn by gradient descent by gradient descent",
 * <a href="https://arxiv.org/abs/1606.04474">arXiv:1606.04474</a>; Metz et al. 2022,
 * "VeLO", <a href="https://arxiv.org/abs/2211.09760">arXiv:2211.09760</a> (ES meta-training
 * of per-parameter MLP optimizers).
 */
public final class LearnedOptimizer implements Optimizer {

    /** Features per element fed to the rule. */
    public static final int FEATURES = 3;

    /**
     * A toy training problem for meta-training: a loss with an analytic gradient.
     */
    public interface Task {
        /** @return the starting parameters */
        double[] initial();

        /**
         * Computes the loss and its gradient.
         *
         * @param x the parameters
         * @param grad receives the gradient
         * @return the loss
         */
        double loss(double[] x, double[] grad);
    }

    private final int hidden;
    private final double[] meta;
    private float learningRate;
    private final Map<Var, float[][]> state = new IdentityHashMap<>();

    private LearnedOptimizer(int hidden, double[] meta, float learningRate) {
        this.hidden = hidden;
        this.meta = meta;
        this.learningRate = learningRate;
    }

    /**
     * The number of meta-parameters of a rule with {@code hidden} units.
     *
     * @param hidden the hidden width
     * @return the meta-parameter count
     */
    public static int metaSize(int hidden) {
        return FEATURES * hidden + hidden + hidden + 1;
    }

    /**
     * Wraps meta-parameters as an optimizer.
     *
     * @param hidden the hidden width
     * @param meta the meta-parameters, {@link #metaSize} of them
     * @param learningRate the scale of the rule's output
     * @return the optimizer
     */
    public static LearnedOptimizer of(int hidden, double[] meta, float learningRate) {
        if (meta.length != metaSize(hidden)) {
            throw new IllegalArgumentException("expected " + metaSize(hidden) + " meta-parameters");
        }
        return new LearnedOptimizer(hidden, meta.clone(), learningRate);
    }

    /**
     * Random small meta-parameters, a starting point for meta-training.
     *
     * @param hidden the hidden width
     * @param seed the seed
     * @return the meta-parameters
     */
    public static double[] initialMeta(int hidden, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] m = new double[metaSize(hidden)];
        for (int i = 0; i < m.length; i++) {
            m[i] = r.nextGaussian() * 0.3;
        }
        return m;
    }

    /** @return a copy of the meta-parameters */
    public double[] meta() {
        return meta.clone();
    }

    /**
     * The rule's step for one element (before the minus sign and learning rate).
     *
     * @param g the gradient
     * @param m the gradient's momentum
     * @param v the gradient's running second moment
     * @return the step direction
     */
    double rule(double g, double m, double v) {
        double rms = Math.sqrt(v) + 1e-8;
        double f0 = g / rms;
        double f1 = m / rms;
        double f2 = Math.signum(g) * Math.signum(m);
        int h = hidden;
        double out = meta[FEATURES * h + 2 * h];
        for (int j = 0; j < h; j++) {
            double a = meta[FEATURES * h + j] + meta[j] * f0 + meta[h + j] * f1 + meta[2 * h + j] * f2;
            out += meta[FEATURES * h + h + j] * Math.tanh(a);
        }
        return out;
    }

    /**
     * Applies one step to a flat parameter vector, the form used for meta-training.
     *
     * @param x the parameters, updated in place
     * @param g their gradient
     * @param m the momentum state, updated in place
     * @param v the second-moment state, updated in place
     * @param rate the learning rate
     */
    void apply(double[] x, double[] g, double[] m, double[] v, double rate) {
        for (int i = 0; i < x.length; i++) {
            m[i] = 0.9 * m[i] + 0.1 * g[i];
            v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
            x[i] -= rate * rule(g[i], m[i], Math.max(v[i], 0.001 * g[i] * g[i]));
        }
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            Tensor grad = p.grad();
            if (grad == null) {
                continue;
            }
            float[] g = grad.to(DType.FLOAT32).toArray();
            float[] x = p.value().to(DType.FLOAT32).toArray();
            float[][] s = state.computeIfAbsent(p, k -> new float[][] {new float[g.length], new float[g.length]});
            for (int i = 0; i < g.length; i++) {
                s[0][i] = 0.9f * s[0][i] + 0.1f * g[i];
                s[1][i] = 0.999f * s[1][i] + 0.001f * g[i] * g[i];
                x[i] -= (float) (learningRate * rule(g[i], s[0][i], Math.max(s[1][i], 0.001 * g[i] * g[i])));
            }
            Tensor t = Tensor.of(p.value().shape(), x, p.value().device());
            p.assign(p.value().dtype() == DType.FLOAT32 ? t : t.to(p.value().dtype()));
            p.zeroGrad();
        }
    }

    @Override
    public void forget(Var parameter) {
        state.remove(parameter);
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float rate) {
        this.learningRate = rate;
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (int i = 0; i < parameters.size(); i++) {
            float[][] s = state.get(parameters.get(i));
            if (s != null) {
                out.put(i + ".m", Tensor.of(Shape.of(s[0].length), s[0].clone()));
                out.put(i + ".v", Tensor.of(Shape.of(s[1].length), s[1].clone()));
            }
        }
        return out;
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> saved) {
        state.clear();
        for (int i = 0; i < parameters.size(); i++) {
            Tensor m = saved.get(i + ".m");
            Tensor v = saved.get(i + ".v");
            if (m != null && v != null) {
                state.put(parameters.get(i), new float[][] {m.toArray(), v.toArray()});
            }
        }
    }

    /**
     * Trains a task for some steps with this rule.
     *
     * @param task the task
     * @param steps the number of steps
     * @return the mean of {@code log(loss / initial loss)} over the run: lower is better
     */
    public double unroll(Task task, int steps) {
        double[] x = task.initial();
        double[] g = new double[x.length];
        double[] m = new double[x.length];
        double[] v = new double[x.length];
        double first = Math.max(1e-12, task.loss(x, g));
        double sum = 0;
        for (int t = 0; t < steps; t++) {
            apply(x, g, m, v, learningRate);
            double l = task.loss(x, g);
            if (!Double.isFinite(l) || l > 1e6 * first) {
                return 10.0 * (steps - t) / steps + Math.log(1e6);   // diverged
            }
            sum += Math.log(Math.max(l, 1e-12) / first);
        }
        return sum / steps;
    }

    /**
     * Meta-trains a rule with an evolution strategy on random tasks.
     *
     * @param hidden the hidden width
     * @param tasks makes a random task from a random source
     * @param learningRate the output scale
     * @param steps training steps per unroll
     * @param tasksPerEvaluation tasks averaged per evaluation
     * @param iterations evolution strategy iterations
     * @param seed the seed
     * @return the meta-trained optimizer
     */
    public static LearnedOptimizer metaTrain(int hidden, java.util.function.Function<SplittableRandom, Task> tasks,
            float learningRate, int steps, int tasksPerEvaluation, int iterations, long seed) {
        EvolutionStrategy es = EvolutionStrategy.of(initialMeta(hidden, seed)).sigma(0.05).learningRate(0.03)
                .pairs(12).seed(seed);
        int[] iteration = {0};
        for (int it = 0; it < iterations; it++) {
            iteration[0] = it;
            es.step(meta -> {
                // same tasks for every candidate of an iteration: a fair comparison
                SplittableRandom r = new SplittableRandom(seed * 7919 + iteration[0]);
                LearnedOptimizer opt = of(hidden, meta, learningRate);
                double total = 0;
                for (int k = 0; k < tasksPerEvaluation; k++) {
                    total += opt.unroll(tasks.apply(r), steps);
                }
                return -total / tasksPerEvaluation;
            });
        }
        return of(hidden, es.center(), learningRate);
    }

    /**
     * A random ill-conditioned quadratic {@code sum_i c_i (x_i - t_i)^2 / 2} with curvatures
     * spread over two orders of magnitude.
     *
     * @param dim the dimension
     * @param random the random source
     * @return the task
     */
    public static Task quadratic(int dim, SplittableRandom random) {
        double[] c = new double[dim];
        double[] target = new double[dim];
        double[] start = new double[dim];
        for (int i = 0; i < dim; i++) {
            c[i] = Math.exp(random.nextDouble(Math.log(0.05), Math.log(5)));
            target[i] = random.nextGaussian();
            start[i] = random.nextGaussian();
        }
        return new Task() {
            @Override
            public double[] initial() {
                return start.clone();
            }

            @Override
            public double loss(double[] x, double[] grad) {
                double l = 0;
                for (int i = 0; i < dim; i++) {
                    double d = x[i] - target[i];
                    l += 0.5 * c[i] * d * d;
                    grad[i] = c[i] * d;
                }
                return l;
            }
        };
    }

    /**
     * Random least-squares regression: fit {@code w} to {@code y = X w* + noise}.
     *
     * @param samples the number of rows
     * @param dim the number of features
     * @param random the random source
     * @return the task
     */
    public static Task linearRegression(int samples, int dim, SplittableRandom random) {
        double[][] x = new double[samples][dim];
        double[] y = new double[samples];
        double[] w = new double[dim];
        for (int j = 0; j < dim; j++) {
            w[j] = random.nextGaussian();
        }
        double scale = Math.exp(random.nextDouble(-1, 1));
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < dim; j++) {
                x[i][j] = random.nextGaussian() * scale;
                y[i] += x[i][j] * w[j];
            }
            y[i] += 0.1 * random.nextGaussian();
        }
        return new Task() {
            @Override
            public double[] initial() {
                return new double[dim];
            }

            @Override
            public double loss(double[] p, double[] grad) {
                java.util.Arrays.fill(grad, 0);
                double l = 0;
                for (int i = 0; i < samples; i++) {
                    double e = -y[i];
                    for (int j = 0; j < dim; j++) {
                        e += x[i][j] * p[j];
                    }
                    l += e * e / samples;
                    for (int j = 0; j < dim; j++) {
                        grad[j] += 2 * e * x[i][j] / samples;
                    }
                }
                return l;
            }
        };
    }

    @Override
    public String toString() {
        return "LearnedOptimizer(hidden=" + hidden + ", lr=" + learningRate + ")";
    }
}
