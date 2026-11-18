package io.github.zzuegg.brainiac.experiments.estimation;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DecisionInstances;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.diffopt.ConstrainedSolvers;
import io.github.zzuegg.brainiac.experimental.diffopt.Hypergradients;
import io.github.zzuegg.brainiac.experimental.diffopt.LearnedOptimizers;
import io.github.zzuegg.brainiac.experimental.diffopt.VarianceReduced;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.experimental.train.DAdaptation;
import io.github.zzuegg.brainiac.experimental.train.Lbfgs;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Optimisation methods (12.5.6-12.5.9) against their classical baselines.
 * <ul>
 *   <li>{@link #weightDecay}: per-feature weight decay on a 200-feature regression with 10 useful
 *       features, tuned by IFT hypergradients (Neumann and CG), unrolled and truncated
 *       differentiation, against grid and random search over one global decay;</li>
 *   <li>{@link #dataReweighting}: per-example weights on digits with 30% flipped labels, tuned by
 *       IFT hypergradients on a clean validation set: test accuracy and the AUROC of the weights
 *       for spotting flipped labels, against plain training, the per-example loss and a
 *       clean-data oracle;</li>
 *   <li>{@link #finiteSums}: SVRG, SAGA and SARAH against SGD, Adam and full-batch L-BFGS on
 *       conditioned least squares and logistic regression, and on a small digits MLP;</li>
 *   <li>{@link #constrained}: Frank-Wolfe variants against projected gradient, FISTA and ADMM on the
 *       simplex, the l1 ball and trace-norm matrix completion;</li>
 *   <li>{@link #learnedOptimizers}: an LSTM L2O and Celo-lite meta-trained in seconds and tested on
 *       unseen tasks against tuned and default Adam and D-Adaptation.</li>
 * </ul>
 */
public final class OptimisationMethods {

    private OptimisationMethods() {
    }

    private static Tensor rows(double[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, d), v);
    }

    private static Tensor column(double[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = (float) y[i];
        }
        return Tensor.of(Shape.of(y.length, 1), v);
    }

    // ------------------------------------------------------------------------------ 12.5.6

    /** Ridge with per-feature decay {@code exp(lambda_j)}: {@code (X^T X / n + diag(e^lambda)) w = X^T y / n}. */
    private static double[] ridge(double[][] x, double[] y, double[] logDecay) {
        int p = x[0].length;
        int n = x.length;
        double[][] m = new double[p][p];
        double[] rhs = new double[p];
        for (double[] row : x) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < p; k++) {
                    m[j][k] += row[j] * row[k] / n;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                rhs[j] += x[i][j] * y[i] / n;
            }
        }
        for (int j = 0; j < p; j++) {
            m[j][j] += Math.exp(logDecay[j]);
        }
        return solve(m, rhs);
    }

    /** A dense solve in double precision through core {@code Linalg.solve}. */
    private static double[] solve(double[][] m, double[] b) {
        int n = b.length;
        double[] flat = new double[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, flat, i * n, n);
        }
        Tensor a = Tensor.of(Shape.of(n, n), flat, DType.FLOAT64, Device.cpu());
        Tensor rhs = Tensor.of(Shape.of(n, 1), b.clone(), DType.FLOAT64, Device.cpu());
        return Linalg.solve(a, rhs).toDoubleArray();
    }

    private static double mse(double[][] x, double[] y, double[] w) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double r = -y[i];
            for (int j = 0; j < w.length; j++) {
                r += x[i][j] * w[j];
            }
            s += r * r / x.length;
        }
        return s;
    }

    /**
     * Per-feature weight decay by hypergradients against global-decay search.
     *
     * @param outerSteps hypergradient steps
     * @param seed the seed
     * @return method to test MSE (the learned decays are evaluated by an exact ridge solve)
     */
    public static Map<String, Double> weightDecay(int outerSteps, long seed) {
        int p = 200;
        int useful = 10;
        SplittableRandom r = new SplittableRandom(seed);
        double[] truth = new double[p];
        for (int j = 0; j < useful; j++) {
            truth[j] = r.nextGaussian() * 1.5;
        }
        double[][][] x = new double[3][][];
        double[][] y = new double[3][];
        int[] sizes = {100, 100, 1000};
        for (int s = 0; s < 3; s++) {
            x[s] = new double[sizes[s]][p];
            y[s] = new double[sizes[s]];
            for (int i = 0; i < sizes[s]; i++) {
                double v = 0;
                for (int j = 0; j < p; j++) {
                    x[s][i][j] = r.nextGaussian();
                    v += x[s][i][j] * truth[j];
                }
                y[s][i] = v + r.nextGaussian() * 0.5;
            }
        }
        Map<String, Double> out = new LinkedHashMap<>();
        // grid and random search over one global decay, selected on validation
        double bestVal = Double.POSITIVE_INFINITY;
        double bestTest = 0;
        for (int g = -8; g <= 4; g++) {
            double[] lam = new double[p];
            java.util.Arrays.fill(lam, g * Math.log(10) / 2);
            double[] w = ridge(x[0], y[0], lam);
            double v = mse(x[1], y[1], w);
            if (v < bestVal) {
                bestVal = v;
                bestTest = mse(x[2], y[2], w);
            }
        }
        out.put("grid search (global)", bestTest);
        bestVal = Double.POSITIVE_INFINITY;
        for (int k = 0; k < 13; k++) {
            double[] lam = new double[p];
            java.util.Arrays.fill(lam, r.nextDouble(-9.2, 4.6));
            double[] w = ridge(x[0], y[0], lam);
            double v = mse(x[1], y[1], w);
            if (v < bestVal) {
                bestVal = v;
                bestTest = mse(x[2], y[2], w);
            }
        }
        out.put("random search (global)", bestTest);
        Tensor xt = rows(x[0]);
        Tensor yt = column(y[0]);
        Tensor xv = rows(x[1]);
        Tensor yv = column(y[1]);
        Hypergradients.Problem problem = new Hypergradients.Problem(
                (ws, lam) -> Var.of(xt).matmul(ws.get(0)).sub(yt).square().mean()
                        .add(lam.exp().mul(ws.get(0).square()).sum()),
                ws -> Var.of(xv).matmul(ws.get(0)).sub(yv).square().mean());
        Map<String, Hypergradients> methods = new LinkedHashMap<>();
        methods.put("IFT Neumann (10 terms)", Hypergradients.implicit(Hypergradients.Method.IFT_NEUMANN, 10, 0.05, 40,
                0.05));
        methods.put("IFT CG (10 iterations)", Hypergradients.implicit(Hypergradients.Method.IFT_CG, 10, 0, 40, 0.05));
        methods.put("unrolled (40 steps)", Hypergradients.unrolled(40, 0.05, 40));
        methods.put("truncated (last 5 of 40)", Hypergradients.unrolled(40, 0.05, 5));
        for (Map.Entry<String, Hypergradients> e : methods.entrySet()) {
            Var w = Var.param(Tensor.zeros(Shape.of(p, 1)));
            Var lam = Var.param(Tensor.full(Shape.of(p, 1), (float) Math.log(0.05)));
            Optimizer outer = Optimizer.adam(0.1f);
            for (int s = 0; s < outerSteps; s++) {
                Tensor hg = e.getValue().hypergradient(problem, List.of(w), lam);
                lam.setGrad(hg);
                outer.step(List.of(lam));
                lam.zeroGrad();
            }
            double[] learned = lam.value().toDoubleArray();
            out.put(e.getKey(), mse(x[2], y[2], ridge(x[0], y[0], learned)));
        }
        return out;
    }

    /**
     * Per-example data weights learned by IFT hypergradients on noisy-label digits.
     *
     * @param outerSteps hypergradient steps
     * @param seed the seed
     * @return "test accuracy / ..." and "AUROC / ..." rows
     */
    public static Map<String, Double> dataReweighting(int outerSteps, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 40, 8, 0.03, seed, seed + 1);
        ToyTasks.Data val = ToyTasks.digits(10, 10, 8, 0.03, seed, seed + 2);
        ToyTasks.Data test = ToyTasks.digits(10, 50, 8, 0.03, seed, seed + 3);
        LabelCorruption.Noisy noisy = LabelCorruption.symmetric(train.y(), 10, 0.3, seed);
        boolean[] flipped = noisy.flipped();
        int n = train.size();
        Tensor xt = train.inputs();
        Tensor ids = ids(noisy.noisy());
        Tensor xv = val.inputs();
        Tensor idv = val.ids();
        Map<String, Double> out = new LinkedHashMap<>();
        // plain training and the clean oracle
        Var plainW = trainSoftmax(xt, ids, null, seed);
        out.put("test accuracy / plain (noisy labels)", accuracy(plainW, test));
        double[] perExample = Var.noGrad(() -> Functional.crossEntropyPerRow(Var.of(xt).matmul(plainW), ids)).value()
                .toDoubleArray();
        out.put("AUROC / per-example loss", MlMetrics.auroc(perExample, flipped));
        float[] cleanMask = new float[n];
        for (int i = 0; i < n; i++) {
            cleanMask[i] = flipped[i] ? 0 : 1;
        }
        out.put("test accuracy / clean-data oracle", accuracy(trainSoftmax(xt, ids,
                Tensor.of(Shape.of(n), cleanMask), seed), test));
        Hypergradients.Problem problem = new Hypergradients.Problem(
                (ws, omega) -> Functional.crossEntropyPerRow(Var.of(xt).matmul(ws.get(0)), ids)
                        .mul(omega.reshape(n).sigmoid().mul(2f)).mean().add(ws.get(0).square().sum().mul(1e-3f)),
                ws -> Functional.crossEntropy(Var.of(xv).matmul(ws.get(0)), idv));
        Var w = Var.param(Tensor.zeros(Shape.of(64, 10)));
        Var omega = Var.param(Tensor.zeros(Shape.of(n, 1)));
        Hypergradients hg = Hypergradients.implicit(Hypergradients.Method.IFT_NEUMANN, 10, 0.3, 30, 0.5);
        Optimizer outer = Optimizer.adam(0.3f);
        for (int s = 0; s < outerSteps; s++) {
            Tensor g = hg.hypergradient(problem, List.of(w), omega);
            omega.setGrad(g.mul(n));   // per-example hypergradients are O(1/n)
            outer.step(List.of(omega));
            omega.zeroGrad();
        }
        double[] om = omega.value().toDoubleArray();
        double[] suspicion = new double[n];
        float[] weights = new float[n];
        for (int i = 0; i < n; i++) {
            suspicion[i] = -om[i];
            weights[i] = (float) (2 / (1 + Math.exp(-om[i])));
        }
        out.put("AUROC / IFT data weights", MlMetrics.auroc(suspicion, flipped));
        out.put("test accuracy / IFT reweighted", accuracy(trainSoftmax(xt, ids, Tensor.of(Shape.of(n), weights), seed),
                test));
        return out;
    }

    private static Tensor ids(int[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = y[i];
        }
        return Tensor.of(Shape.of(y.length), v);
    }

    private static Var trainSoftmax(Tensor x, Tensor ids, Tensor weights, long seed) {
        Var w = Var.param(Tensor.zeros(Shape.of(64, 10)));
        Optimizer opt = Optimizer.adam(0.05f);
        for (int s = 0; s < 200; s++) {
            Var per = Functional.crossEntropyPerRow(Var.of(x).matmul(w), ids);
            Var loss = weights == null ? per.mean() : per.mul(weights).sum().div(weights.sum().item());
            loss.add(w.square().sum().mul(1e-3f)).backward();
            opt.step(List.of(w));
        }
        return w;
    }

    private static double accuracy(Var w, ToyTasks.Data test) {
        return ToyTasks.accuracy(Var.noGrad(() -> Var.of(test.inputs()).matmul(w)).value(), test.y());
    }

    // ------------------------------------------------------------------------------ 12.5.7

    /**
     * Variance-reduced solvers on finite sums.
     *
     * @param epochs passes over the data
     * @param seed the seed
     * @return "problem / method" to {final suboptimality f - f*, gradient evaluations / n}; the
     *     MLP rows hold the final training loss instead of a suboptimality
     */
    public static Map<String, double[]> finiteSums(int epochs, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        DecisionInstances.FiniteSum ls = DecisionInstances.leastSquares(1000, 20, 1e2, 0.1, seed);
        DecisionInstances.FiniteSum lg = DecisionInstances.logistic(1000, 20, 1e2, seed + 1);
        double l2 = 1e-3;
        VarianceReduced.FiniteSum[] problems = {VarianceReduced.leastSquares(ls.a(), ls.b(), l2),
            VarianceReduced.logistic(lg.a(), lg.b(), l2)};
        String[] names = {"least squares k=1e2", "logistic k=1e2"};
        for (int pi = 0; pi < 2; pi++) {
            VarianceReduced.FiniteSum f = problems[pi];
            double[][] a = pi == 0 ? ls.a() : lg.a();
            double lmax = 0;
            for (double[] row : a) {
                double s = 0;
                for (double v : row) {
                    s += v * v;
                }
                lmax = Math.max(lmax, s);
            }
            lmax = (pi == 0 ? lmax : lmax / 4) + l2;
            double fStar = newtonOptimum(f, a, pi == 1, l2);
            int n = f.size();
            double step = 1 / (3 * lmax);
            double[] w0 = new double[f.dimension()];
            for (VarianceReduced.Method m : VarianceReduced.Method.values()) {
                double lr = m == VarianceReduced.Method.ADAM ? 0.01 : step;
                VarianceReduced.Trace t = VarianceReduced.run(m, f, w0, lr, epochs, false, seed);
                out.put(names[pi] + " / " + m, new double[] {Math.max(t.losses()[epochs] - fStar, 0),
                    t.gradientEvaluations() / (double) n});
            }
            VarianceReduced.Trace decayed = VarianceReduced.run(VarianceReduced.Method.SGD, f, w0, 4 * step, epochs,
                    true, seed);
            out.put(names[pi] + " / SGD decayed", new double[] {Math.max(decayed.losses()[epochs] - fStar, 0),
                decayed.gradientEvaluations() / (double) n});
            out.put(names[pi] + " / L-BFGS (full batch)", lbfgs(f, epochs, fStar));
        }
        // a small digits MLP: no gain expected from variance reduction
        ToyTasks.Data digits = ToyTasks.digits(10, 20, 8, 0.1, seed);
        int n = digits.size();
        Tensor x = digits.inputs();
        Tensor ids = digits.ids();
        SplittableRandom r = new SplittableRandom(seed);
        Var w1 = Var.param(Tensor.normal(Shape.of(64, 8), 0, 0.2f, r));
        Var w2 = Var.param(Tensor.normal(Shape.of(8, 10), 0, 0.3f, r));
        List<Var> ps = List.of(w1, w2);
        VarianceReduced.FiniteSum mlp = VarianceReduced.ofModel(ps, i -> Functional.crossEntropy(
                Var.of(x.slice(0, i, i + 1)).matmul(w1).tanh().matmul(w2), ids.slice(0, i, i + 1)), n);
        double[] start = io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo.flatten(ps);
        int mlpEpochs = Math.max(2, epochs / 4);
        for (VarianceReduced.Method m : new VarianceReduced.Method[] {VarianceReduced.Method.SGD,
            VarianceReduced.Method.SVRG, VarianceReduced.Method.ADAM}) {
            VarianceReduced.Trace t = VarianceReduced.run(m, mlp, start, m == VarianceReduced.Method.ADAM ? 0.01 : 0.1,
                    mlpEpochs, false, seed);
            out.put("digits MLP / " + m, new double[] {t.losses()[mlpEpochs], t.gradientEvaluations() / (double) n});
        }
        return out;
    }

    /** The optimum by Newton's method (exact for least squares in one step). */
    private static double newtonOptimum(VarianceReduced.FiniteSum f, double[][] a, boolean logistic, double l2) {
        int d = a[0].length;
        int n = a.length;
        double[] w = new double[d];
        double[] g = new double[d];
        for (int it = 0; it < (logistic ? 30 : 2); it++) {
            f.fullGradient(w, g);
            double[][] h = new double[d][d];
            for (double[] row : a) {
                double z = 0;
                for (int j = 0; j < d; j++) {
                    z += row[j] * w[j];
                }
                double c = logistic ? 1 / (1 + Math.exp(-z)) * (1 - 1 / (1 + Math.exp(-z))) : 1;
                for (int j = 0; j < d; j++) {
                    for (int k = 0; k < d; k++) {
                        h[j][k] += c * row[j] * row[k] / n;
                    }
                }
            }
            double[] q = new double[d];
            for (int j = 0; j < d; j++) {
                h[j][j] += l2;
                q[j] = -g[j];
            }
            double[] step = solve(h, q);
            for (int j = 0; j < d; j++) {
                w[j] += step[j];
            }
        }
        return f.loss(w);
    }

    private static double[] lbfgs(VarianceReduced.FiniteSum f, int epochs, double fStar) {
        int d = f.dimension();
        Var w = Var.param(Tensor.zeros(Shape.of(d)));
        Lbfgs opt = Lbfgs.fullBatch(1, 10);
        double[] g = new double[d];
        Optimizer.Reevaluation re = () -> {
            double[] wv = w.value().toDoubleArray();
            f.fullGradient(wv, g);
            float[] gf = new float[d];
            for (int j = 0; j < d; j++) {
                gf[j] = (float) g[j];
            }
            w.setGrad(Tensor.of(Shape.of(d), gf));
            return (float) f.loss(wv);
        };
        for (int it = 0; it < epochs; it++) {
            re.lossAndGradients();
            opt.step(List.of(w), re);
        }
        double loss = f.loss(w.value().toDoubleArray());
        return new double[] {Math.max(loss - fStar, 0), opt.evaluations() + epochs};
    }

    // ------------------------------------------------------------------------------ 12.5.8

    /**
     * Constrained solvers at a fixed iteration budget.
     *
     * @param iterations iterations per solver
     * @param seed the seed
     * @return "problem / method" to {suboptimality, support size or rank, ms per iteration, test
     *     RMSE (completion only, else 0)}
     */
    public static Map<String, double[]> constrained(int iterations, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        int m = 100;
        int d = 40;
        double[][] a = new double[m][d];
        for (double[] row : a) {
            for (int j = 0; j < d; j++) {
                row[j] = r.nextGaussian();
            }
        }
        // simplex: a sparse portfolio-like truth
        double[] xs = new double[d];
        xs[3] = 0.5;
        xs[11] = 0.3;
        xs[27] = 0.2;
        double[] bs = matVec(a, xs);
        for (int i = 0; i < m; i++) {
            bs[i] += 0.3 * r.nextGaussian();
        }
        // l1 ball: a sparse signed truth, radius below its l1 norm so the constraint binds
        double[] xl = new double[d];
        xl[0] = 2;
        xl[5] = -1.5;
        xl[9] = 1;
        double[] bl = matVec(a, xl);
        for (int i = 0; i < m; i++) {
            bl[i] += 0.3 * r.nextGaussian();
        }
        double lip = topEigen(a);
        Object[][] problems = {
            {"simplex LS", ConstrainedSolvers.leastSquares(a, bs), ConstrainedSolvers.Set.simplex(1)},
            {"l1-ball LS", ConstrainedSolvers.leastSquares(a, bl), ConstrainedSolvers.Set.l1Ball(3.5)},
        };
        for (Object[] pr : problems) {
            ConstrainedSolvers.Objective f = (ConstrainedSolvers.Objective) pr[1];
            ConstrainedSolvers.Set set = (ConstrainedSolvers.Set) pr[2];
            double fStar = ConstrainedSolvers.solve(ConstrainedSolvers.Method.FISTA, f, set, lip, 5000, d)
                    .objective()[5000];
            for (ConstrainedSolvers.Method method : ConstrainedSolvers.Method.values()) {
                ConstrainedSolvers.Result res = ConstrainedSolvers.solve(method, f, set,
                        method == ConstrainedSolvers.Method.ADMM ? lip / 10 : lip, iterations, d);
                out.put(pr[0] + " / " + method, new double[] {Math.max(0, res.objective()[iterations] - fStar),
                    ConstrainedSolvers.support(res.x()), res.millis(), 0});
            }
        }
        // trace-norm matrix completion
        int rows = 40;
        int cols = 40;
        double[][] u = new double[rows][3];
        double[][] v = new double[cols][3];
        for (double[] row : u) {
            for (int k = 0; k < 3; k++) {
                row[k] = r.nextGaussian();
            }
        }
        for (double[] row : v) {
            for (int k = 0; k < 3; k++) {
                row[k] = r.nextGaussian();
            }
        }
        double[] truth = new double[rows * cols];
        boolean[] observed = new boolean[rows * cols];
        double[][] tm = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double s = 0;
                for (int k = 0; k < 3; k++) {
                    s += u[i][k] * v[j][k];
                }
                truth[i * cols + j] = s;
                tm[i][j] = s;
                observed[i * cols + j] = r.nextDouble() < 0.4;
            }
        }
        double nuclear = 0;
        for (double s : ThinSvd.of(tm).s()) {
            nuclear += s;
        }
        double[] noisy = truth.clone();
        for (int i = 0; i < noisy.length; i++) {
            noisy[i] += 0.1 * r.nextGaussian();
        }
        ConstrainedSolvers.Objective f = ConstrainedSolvers.maskedSquares(observed, noisy);
        ConstrainedSolvers.Set set = ConstrainedSolvers.Set.traceNormBall(nuclear, rows, cols);
        double fStar = ConstrainedSolvers.solve(ConstrainedSolvers.Method.FISTA, f, set, 1, 300, rows * cols)
                .objective()[300];
        for (ConstrainedSolvers.Method method : new ConstrainedSolvers.Method[] {ConstrainedSolvers.Method.FRANK_WOLFE,
            ConstrainedSolvers.Method.PROJECTED_GRADIENT, ConstrainedSolvers.Method.FISTA,
            ConstrainedSolvers.Method.ADMM}) {
            ConstrainedSolvers.Result res = ConstrainedSolvers.solve(method, f, set, 1, iterations, rows * cols);
            double[][] xm = new double[rows][cols];
            double se = 0;
            int unseen = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    xm[i][j] = res.x()[i * cols + j];
                    if (!observed[i * cols + j]) {
                        se += Math.pow(xm[i][j] - truth[i * cols + j], 2);
                        unseen++;
                    }
                }
            }
            double[] sv = ThinSvd.of(xm).s();
            int rank = 0;
            for (double s : sv) {
                rank += s > 1e-3 * sv[0] ? 1 : 0;
            }
            out.put("trace-norm completion / " + method, new double[] {Math.max(0, res.objective()[iterations] - fStar),
                rank, res.millis(), Math.sqrt(se / unseen)});
        }
        return out;
    }

    private static double[] matVec(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += a[i][j] * x[j];
            }
        }
        return y;
    }

    private static double topEigen(double[][] a) {
        int d = a[0].length;
        double[] v = new double[d];
        java.util.Arrays.fill(v, 1);
        double lambda = 0;
        for (int it = 0; it < 100; it++) {
            double[] av = matVec(a, v);
            double[] w = new double[d];
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < d; j++) {
                    w[j] += a[i][j] * av[i];
                }
            }
            double norm = 0;
            for (double x : w) {
                norm += x * x;
            }
            norm = Math.sqrt(norm);
            lambda = norm;
            for (int j = 0; j < d; j++) {
                v[j] = w[j] / norm;
            }
        }
        return lambda;
    }

    // ------------------------------------------------------------------------------ 12.5.9

    /** A meta-test task: a model, its parameters and its loss. */
    private record VarTask(List<Var> parameters, Supplier<Var> loss) {
    }

    private static VarTask spiralTask(int hidden, long seed) {
        ToyTasks.Data data = ToyTasks.spiral(40, 2, 0.15, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Var w1 = Var.param(Tensor.normal(Shape.of(2, hidden), 0, (float) (1 / Math.sqrt(2)), r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        Var w2 = Var.param(Tensor.normal(Shape.of(hidden, 2), 0, (float) (1 / Math.sqrt(hidden)), r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1, 2)));
        Tensor x = data.inputs();
        Tensor ids = data.ids();
        return new VarTask(List.of(w1, b1, w2, b2),
                () -> Functional.crossEntropy(Var.of(x).matmul(w1).add(b1).tanh().matmul(w2).add(b2), ids));
    }

    private static VarTask digitsTask(long seed) {
        ToyTasks.Data data = ToyTasks.digits(10, 20, 8, 0.1, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Var w1 = Var.param(Tensor.normal(Shape.of(64, 16), 0, 0.125f, r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1, 16)));
        Var w2 = Var.param(Tensor.normal(Shape.of(16, 10), 0, 0.25f, r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1, 10)));
        Tensor x = data.inputs();
        Tensor ids = data.ids();
        return new VarTask(List.of(w1, b1, w2, b2),
                () -> Functional.crossEntropy(Var.of(x).matmul(w1).add(b1).tanh().matmul(w2).add(b2), ids));
    }

    /** Trains a fresh copy of a task; returns the final loss (NaN-safe: 1e9 on divergence). */
    private static double train(Supplier<VarTask> make, Supplier<Optimizer> optimizer, int steps) {
        VarTask task = make.get();
        Optimizer opt = optimizer.get();
        if (opt instanceof LearnedOptimizers.CeloLite c) {
            c.horizon(steps);
        }
        double last = 0;
        for (int s = 0; s < steps; s++) {
            Var loss = task.loss().get();
            last = loss.value().toDoubleArray()[0];
            if (!Double.isFinite(last)) {
                return 1e9;
            }
            if (opt instanceof LearnedOptimizers.CeloLite c) {
                c.observeLoss(last);
            }
            loss.backward();
            opt.step(task.parameters());
            for (Var p : task.parameters()) {
                p.zeroGrad();
            }
        }
        double fin = task.loss().get().value().toDoubleArray()[0];
        return Double.isFinite(fin) ? fin : 1e9;
    }

    /**
     * Learned optimisers meta-trained briefly, then tested on unseen tasks.
     *
     * @param metaIterations meta-iterations (per Celo stage; LSTM uses the same count)
     * @param steps meta-test steps (the long-horizon row uses 5x)
     * @param seed the seed
     * @return "task / optimiser" to final loss divided by tuned Adam's final loss
     */
    public static Map<String, Double> learnedOptimizers(int metaIterations, int steps, long seed) {
        LearnedOptimizers.LstmL2O lstm = LearnedOptimizers.LstmL2O.create(seed);
        lstm.metaTrain(10, metaIterations, 40, 20, 0.01f, seed);
        ToyTasks.Data spiral = ToyTasks.spiral(30, 2, 0.15, seed + 100);
        double[][] sx = new double[spiral.size()][2];
        for (int i = 0; i < sx.length; i++) {
            sx[i][0] = spiral.x()[i][0];
            sx[i][1] = spiral.x()[i][1];
        }
        List<Function<SplittableRandom, LearnedOptimizers.Task>> tasks = List.of(
            rr -> LearnedOptimizers.quadratic(10, rr),
            rr -> LearnedOptimizers.mlp(sx, spiral.y(), 2, 8));
        LearnedOptimizers.CeloLite celo = LearnedOptimizers.metaTrainCelo(tasks, true, metaIterations, 8, 60, 20, 0.05,
                0.02, seed);
        LearnedOptimizers.CeloLite celoNoAug = LearnedOptimizers.metaTrainCelo(tasks, false, metaIterations, 8, 60, 20,
                0.05, 0.02, seed);
        Map<String, Supplier<VarTask>> testTasks = new LinkedHashMap<>();
        testTasks.put("spiral MLP h8 (in-distribution)", () -> spiralTask(8, seed + 7));
        testTasks.put("spiral MLP h32 (wider)", () -> spiralTask(32, seed + 8));
        testTasks.put("digits MLP 64-16-10", () -> digitsTask(seed + 9));
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, Supplier<VarTask>> e : testTasks.entrySet()) {
            evaluate(e.getKey(), e.getValue(), steps, lstm, celo, celoNoAug, out);
        }
        evaluate("spiral MLP h8 x5 horizon", testTasks.get("spiral MLP h8 (in-distribution)"), 5 * steps, lstm, celo,
                celoNoAug, out);
        return out;
    }

    private static void evaluate(String name, Supplier<VarTask> task, int steps, LearnedOptimizers.LstmL2O lstm,
            LearnedOptimizers.CeloLite celo, LearnedOptimizers.CeloLite celoNoAug, Map<String, Double> out) {
        double best = Double.POSITIVE_INFINITY;
        for (float lr : new float[] {3e-3f, 1e-2f, 3e-2f, 1e-1f}) {
            best = Math.min(best, train(task, () -> Optimizer.adam(lr), steps));
        }
        List<String> names = new ArrayList<>();
        List<Supplier<Optimizer>> opts = new ArrayList<>();
        names.add("Adam lr=1e-3 (untuned)");
        opts.add(() -> Optimizer.adam(1e-3f));
        names.add("D-Adaptation SGD");
        opts.add(DAdaptation::sgd);
        names.add("LSTM L2O");
        opts.add(() -> lstm);
        names.add("Celo-lite");
        opts.add(() -> LearnedOptimizers.CeloLite.of(celo.rule(), celo.scheduler()));
        names.add("Celo-lite without augmentation");
        opts.add(() -> LearnedOptimizers.CeloLite.of(celoNoAug.rule(), celoNoAug.scheduler()));
        out.put(name + " / Adam tuned (loss)", best);
        for (int i = 0; i < names.size(); i++) {
            double fin = train(task, opts.get(i), steps);
            out.put(name + " / " + names.get(i), fin / best);
        }
    }
}
