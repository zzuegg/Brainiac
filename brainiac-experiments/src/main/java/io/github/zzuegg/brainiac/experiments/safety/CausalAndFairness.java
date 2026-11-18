package io.github.zzuegg.brainiac.experiments.safety;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.ml.CausalEstimators;
import io.github.zzuegg.brainiac.experimental.ml.Dagma;
import io.github.zzuegg.brainiac.experimental.ml.GreedyEquivalenceSearch;
import io.github.zzuegg.brainiac.experimental.ml.Notears;
import io.github.zzuegg.brainiac.experimental.nn.TreatmentEffectNet;
import io.github.zzuegg.brainiac.experimental.train.FairnessInterventions;
import io.github.zzuegg.brainiac.experimental.train.GroupRobustTraining;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Causal inference and fairness on synthetic data with a known ground truth.
 *
 * <p><b>Treatment effects</b> ({@link #effects}): five covariates, of which {@code x0}
 * raises both the chance of treatment and the outcome (confounding); the outcome is
 * {@code 1.5 x0 + x1^2 + 0.5 x3 + t (1 + x0) + noise}, so the true ATE is 1 and the effect
 * varies with {@code x0}. Scores: the ATE error and, for estimators of the conditional
 * effect, PEHE {@code sqrt(mean (tau_hat - tau)^2)}. The naive difference in means and an
 * S-learner MLP are the baselines for IPW, AIPW ({@link CausalEstimators}), TARNet and
 * DragonNet ({@link TreatmentEffectNet}).
 *
 * <p><b>Structure</b> ({@link #structure}): a random linear-Gaussian DAG on eight nodes;
 * NOTEARS ({@link Notears}) against the skeleton phase of the PC algorithm (partial
 * correlation tests with conditioning sets of up to two neighbours). Scores: the
 * structural Hamming distance and the skeleton distance. The follow-up
 * ({@link #structureShootout}) adds DAGMA ({@link Dagma}) and greedy equivalence search
 * ({@link GreedyEquivalenceSearch}) on Erdos-Renyi DAGs of 10-20 nodes with linear or
 * non-linear ({@code tanh}) mechanisms, scoring SHD, true and false positive rates and
 * wall time.
 *
 * <p><b>Group fairness</b> ({@link #fairness}): a protected attribute {@code a} shifts
 * the historical labels and leaks into the features through proxies. ERM against
 * reweighing, adversarial debiasing and equalized-odds thresholds
 * ({@link FairnessInterventions}); scores: accuracy, demographic-parity and equalized-odds
 * gaps.
 *
 * <p><b>Spurious correlation</b> ({@link #spurious}): two-channel "coloured" shapes
 * (horizontal against vertical bars from {@link ShapeImages}), drawn in the channel that
 * matches the class 90% (environment 1) or 80% (environment 2) of the time in training and
 * independently at test. ERM against IRM, GroupDRO, V-REx and Just Train Twice
 * ({@link GroupRobustTraining}; JTT is the only one that needs neither group nor environment labels);
 * score: worst-group test accuracy over the four (class, colour) groups.
 */
public final class CausalAndFairness {

    private CausalAndFairness() {
    }

    // ------------------------------------------------------------------ treatment effects

    /** The treatment-effect estimators. */
    public enum EffectMethod {
        /** Difference in means. */
        NAIVE,
        /** One MLP on covariates and treatment. */
        S_LEARNER,
        /** Inverse propensity weighting with a logistic propensity. */
        IPW,
        /** Doubly robust, logistic propensity and linear outcome models. */
        AIPW,
        /** TARNet. */
        TARNET,
        /** DragonNet with targeted regularization. */
        DRAGONNET
    }

    /**
     * Treatment-effect scores.
     *
     * @param ate the estimated ATE (truth 1)
     * @param ateError {@code |ate - 1|}
     * @param pehe the CATE error, NaN for estimators of the average only
     */
    public record EffectScores(double ate, double ateError, double pehe) {
    }

    private record Observational(double[][] x, int[] t, double[] y, double[] tau) {
    }

    private static Observational scm(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][5];
        int[] t = new int[n];
        double[] y = new double[n];
        double[] tau = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 5; j++) {
                x[i][j] = r.nextGaussian();
            }
            double e = 1 / (1 + Math.exp(-(1.2 * x[i][0] - 0.6 * x[i][1] + 0.4 * x[i][2])));
            t[i] = r.nextDouble() < e ? 1 : 0;
            tau[i] = 1 + x[i][0];
            y[i] = 1.5 * x[i][0] + x[i][1] * x[i][1] + 0.5 * x[i][3] + t[i] * tau[i] + 0.5 * r.nextGaussian();
        }
        return new Observational(x, t, y, tau);
    }

    private static Tensor matrix(double[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, d), v);
    }

    private static Tensor vector(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    private static Tensor vector(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    private static double pehe(double[] est, double[] tau) {
        double s = 0;
        for (int i = 0; i < tau.length; i++) {
            s += (est[i] - tau[i]) * (est[i] - tau[i]) / tau.length;
        }
        return Math.sqrt(s);
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double d : v) {
            s += d / v.length;
        }
        return s;
    }

    /**
     * Estimates the treatment effect with one method.
     *
     * @param method the estimator
     * @param seed the seed
     * @return the scores
     */
    public static EffectScores effects(EffectMethod method, long seed) {
        Observational d = scm(2000, seed);
        int n = d.t().length;
        double ate;
        double pehe = Double.NaN;
        switch (method) {
            case NAIVE -> ate = CausalEstimators.naive(d.t(), d.y());
            case IPW -> ate = CausalEstimators.ipw(d.t(), d.y(), CausalEstimators.propensity(d.x(), d.t(), 1e-3), 0.01);
            case AIPW -> {
                double[][] mu = CausalEstimators.outcomes(d.x(), d.t(), d.y(), 1e-3);
                ate = CausalEstimators.aipw(d.t(), d.y(), CausalEstimators.propensity(d.x(), d.t(), 1e-3), mu[0], mu[1],
                        0.01);
                double[] cate = new double[n];
                for (int i = 0; i < n; i++) {
                    cate[i] = mu[1][i] - mu[0][i];
                }
                pehe = pehe(cate, d.tau());
            }
            case S_LEARNER -> {
                double[][] xt = new double[n][6];
                for (int i = 0; i < n; i++) {
                    System.arraycopy(d.x()[i], 0, xt[i], 0, 5);
                    xt[i][5] = d.t()[i];
                }
                Model m = Model.sequential().input(6).dense(64, Activation.RELU).dense(64, Activation.RELU)
                        .dense(1, Activation.IDENTITY).seed(seed).build();
                Tensor tx = matrix(xt);
                Tensor ty = vector(d.y()).reshape(n, 1);
                Optimizer opt = Optimizer.adam(3e-3f);
                SplittableRandom r = new SplittableRandom(seed);
                for (int s = 0; s < 2000; s++) {
                    int from = r.nextInt(n - 128);
                    m.forward(Var.of(tx.slice(0, from, from + 128)), true).sub(ty.slice(0, from, from + 128)).square()
                            .mean().backward();
                    opt.step(m.parameters());
                }
                double[][] x1 = new double[n][];
                double[][] x0 = new double[n][];
                for (int i = 0; i < n; i++) {
                    x1[i] = xt[i].clone();
                    x1[i][5] = 1;
                    x0[i] = xt[i].clone();
                    x0[i][5] = 0;
                }
                double[] cate = m.forward(Var.of(matrix(x1))).value().sub(m.forward(Var.of(matrix(x0))).value())
                        .toDoubleArray();
                ate = mean(cate);
                pehe = pehe(cate, d.tau());
            }
            default -> {
                TreatmentEffectNet net = new TreatmentEffectNet(method == EffectMethod.TARNET
                        ? TreatmentEffectNet.Kind.TARNET : TreatmentEffectNet.Kind.DRAGONNET, 5, 64,
                        new SplittableRandom(seed));
                Tensor tx = matrix(d.x());
                Tensor tt = vector(d.t());
                Tensor ty = vector(d.y());
                Optimizer opt = Optimizer.adam(3e-3f);
                SplittableRandom r = new SplittableRandom(seed);
                for (int s = 0; s < 2000; s++) {
                    int from = r.nextInt(n - 128);
                    Tensor bx = tx.slice(0, from, from + 128);
                    net.loss(net.forward(Var.of(bx)), tt.slice(0, from, from + 128), ty.slice(0, from, from + 128), 1f,
                            1f).backward();
                    opt.step(net.parameters());
                }
                double[] cate = net.cate(tx).toDoubleArray();
                ate = net.ate(tx);
                pehe = pehe(cate, d.tau());
            }
        }
        return new EffectScores(ate, Math.abs(ate - 1), pehe);
    }

    // ------------------------------------------------------------------ structure

    /**
     * Structure-learning scores.
     *
     * @param notearsShd NOTEARS's structural Hamming distance
     * @param notearsSkeleton NOTEARS's skeleton distance
     * @param pcSkeleton the PC skeleton's distance
     * @param edges the true number of edges (the SHD of the empty graph)
     */
    public record StructureScores(int notearsShd, int notearsSkeleton, int pcSkeleton, int edges) {
    }

    /**
     * Recovers a random DAG from samples.
     *
     * @param seed the seed
     * @return the scores
     */
    public static StructureScores structure(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = 8;
        int n = 1000;
        int[] order = r.ints(0, d).distinct().limit(d).toArray();
        double[][] w = new double[d][d];
        int edges = 0;
        for (int a = 0; a < d; a++) {
            for (int b = a + 1; b < d; b++) {
                if (r.nextDouble() < 0.3) {
                    w[order[a]][order[b]] = (0.5 + 1.5 * r.nextDouble()) * (r.nextBoolean() ? 1 : -1);
                    edges++;
                }
            }
        }
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < d; a++) {
                int j = order[a];
                double v = r.nextGaussian();
                for (int k = 0; k < d; k++) {
                    v += x[i][k] * w[k][j];
                }
                x[i][j] = v;
            }
        }
        double[][] est = new Notears(0.05, 0.3, 400, 12).fit(x);
        return new StructureScores(Notears.shd(w, est), Notears.skeletonDistance(w, est),
                Notears.skeletonDistance(w, pcSkeleton(x, 0.01)), edges);
    }

    /** Structure learners compared by {@link #structureShootout}. */
    public enum StructureMethod {
        /** NOTEARS, the trace-of-matrix-power acyclicity with an augmented Lagrangian. */
        NOTEARS,
        /** DAGMA, the log-determinant acyclicity on a central path. */
        DAGMA,
        /** Greedy forward-backward search with the BIC score. */
        GES,
        /** The PC algorithm's skeleton (undirected, so SHD counts every edge it finds once). */
        PC_SKELETON
    }

    /**
     * One structure learner's scores on one graph.
     *
     * @param method the learner
     * @param shd the structural Hamming distance (a reversed edge counts once)
     * @param skeleton the skeleton distance (directions ignored)
     * @param tpr the share of true edges found, direction ignored
     * @param fpr the share of absent pairs wrongly connected
     * @param millis wall time
     * @param edges the true number of edges
     */
    public record StructureRow(StructureMethod method, int shd, int skeleton, double tpr, double fpr, long millis,
            int edges) {
    }

    /**
     * Compares the structure learners on an Erdos-Renyi DAG with two expected parents per node.
     *
     * @param d the number of nodes
     * @param nonlinear use {@code x_j = sum w tanh(2 x_k) + noise} instead of a linear SEM
     * @param seed the seed
     * @return one row per {@link StructureMethod}
     */
    public static List<StructureRow> structureShootout(int d, boolean nonlinear, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 1000;
        int[] order = r.ints(0, d).distinct().limit(d).toArray();
        double[][] w = new double[d][d];
        int edges = 0;
        double p = 2.0 / (d - 1);
        for (int a = 0; a < d; a++) {
            for (int b = a + 1; b < d; b++) {
                if (r.nextDouble() < p) {
                    w[order[a]][order[b]] = (0.5 + 1.5 * r.nextDouble()) * (r.nextBoolean() ? 1 : -1);
                    edges++;
                }
            }
        }
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < d; a++) {
                int j = order[a];
                double v = r.nextGaussian();
                for (int k = 0; k < d; k++) {
                    v += (nonlinear ? Math.tanh(2 * x[i][k]) : x[i][k]) * w[k][j];
                }
                x[i][j] = v;
            }
        }
        List<StructureRow> rows = new ArrayList<>();
        for (StructureMethod m : StructureMethod.values()) {
            long t0 = System.nanoTime();
            double[][] est = switch (m) {
                case NOTEARS -> new Notears(0.05, 0.3, 400, 12).fit(x);
                case DAGMA -> new Dagma(0.05, 0.3, 1500, 4).fit(x);
                case GES -> new GreedyEquivalenceSearch(1, 4).fit(x);
                case PC_SKELETON -> pcSkeleton(x, 0.01);
            };
            long ms = (System.nanoTime() - t0) / 1_000_000;
            int tp = 0;
            int fp = 0;
            for (int i = 0; i < d; i++) {
                for (int j = i + 1; j < d; j++) {
                    boolean t = w[i][j] != 0 || w[j][i] != 0;
                    boolean e = est[i][j] != 0 || est[j][i] != 0;
                    tp += t && e ? 1 : 0;
                    fp += !t && e ? 1 : 0;
                }
            }
            int pairs = d * (d - 1) / 2;
            rows.add(new StructureRow(m, Notears.shd(w, est), Notears.skeletonDistance(w, est),
                    edges == 0 ? 1 : tp / (double) edges, fp / (double) Math.max(1, pairs - edges), ms, edges));
        }
        return rows;
    }

    /** The PC algorithm's skeleton: drop i-j when a set of up to two neighbours makes them independent. */
    private static double[][] pcSkeleton(double[][] x, double alpha) {
        int n = x.length;
        int d = x[0].length;
        double[][] corr = correlation(x);
        boolean[][] adj = new boolean[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                adj[i][j] = i != j;
            }
        }
        double critical = 2.576;   // two-sided z at alpha = 0.01
        if (alpha != 0.01) {
            throw new IllegalArgumentException("only alpha = 0.01 is tabulated");
        }
        for (int level = 0; level <= 2; level++) {
            for (int i = 0; i < d; i++) {
                for (int j = i + 1; j < d; j++) {
                    if (!adj[i][j]) {
                        continue;
                    }
                    List<Integer> nb = new ArrayList<>();
                    for (int k = 0; k < d; k++) {
                        if (k != i && k != j && (adj[i][k] || adj[j][k])) {
                            nb.add(k);
                        }
                    }
                    for (int[] s : subsets(nb, level)) {
                        double pc = partialCorrelation(corr, i, j, s);
                        double z = 0.5 * Math.log((1 + pc) / (1 - pc)) * Math.sqrt(n - s.length - 3);
                        if (Math.abs(z) < critical) {
                            adj[i][j] = false;
                            adj[j][i] = false;
                            break;
                        }
                    }
                }
            }
        }
        double[][] out = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = adj[i][j] ? 1 : 0;
            }
        }
        return out;
    }

    private static List<int[]> subsets(List<Integer> items, int size) {
        List<int[]> out = new ArrayList<>();
        if (size == 0) {
            out.add(new int[0]);
        } else if (size == 1) {
            for (int a : items) {
                out.add(new int[] {a});
            }
        } else {
            for (int a = 0; a < items.size(); a++) {
                for (int b = a + 1; b < items.size(); b++) {
                    out.add(new int[] {items.get(a), items.get(b)});
                }
            }
        }
        return out;
    }

    private static double[][] correlation(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] m = new double[d];
        double[] sd = new double[d];
        for (double[] r : x) {
            for (int j = 0; j < d; j++) {
                m[j] += r[j] / n;
            }
        }
        double[][] c = new double[d][d];
        for (double[] r : x) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    c[a][b] += (r[a] - m[a]) * (r[b] - m[b]) / n;
                }
            }
        }
        for (int j = 0; j < d; j++) {
            sd[j] = Math.sqrt(c[j][j]);
        }
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                c[a][b] /= sd[a] * sd[b];
            }
        }
        return c;
    }

    /** The partial correlation of i and j given s, from the inverse of their correlation submatrix. */
    private static double partialCorrelation(double[][] corr, int i, int j, int[] s) {
        int k = s.length + 2;
        int[] idx = new int[k];
        idx[0] = i;
        idx[1] = j;
        System.arraycopy(s, 0, idx, 2, s.length);
        double[] sub = new double[k * k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                sub[a * k + b] = corr[idx[a]][idx[b]];
            }
        }
        double[] p = Linalg.inverse(Tensor.of(Shape.of(k, k), sub, DType.FLOAT64, Device.cpu())).toDoubleArray();
        return Math.clamp(-p[1] / Math.sqrt(p[0] * p[k + 1]), -0.999999, 0.999999);
    }

    // ------------------------------------------------------------------ group fairness

    /** The fairness interventions. */
    public enum FairnessMethod {
        /** Plain empirical risk minimization. */
        ERM,
        /** Reweighing to make label and attribute independent. */
        REWEIGHING,
        /** Adversarial debiasing, equalized-odds adversary. */
        ADVERSARIAL,
        /** ERM plus per-group thresholds for equalized odds. */
        EQUALIZED_ODDS_POST
    }

    /**
     * Fairness scores on the test set.
     *
     * @param accuracy accuracy against the (biased) labels
     * @param dpGap the demographic-parity gap
     * @param eoGap the equalized-odds gap
     */
    public record FairnessScores(double accuracy, double dpGap, double eoGap) {
    }

    private record Applicants(double[][] x, int[] y, int[] a) {
    }

    private static Applicants applicants(int n, SplittableRandom r) {
        double[][] x = new double[n][4];
        int[] y = new int[n];
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextInt(2);
            double skill = r.nextGaussian();
            x[i][0] = skill + 0.3 * r.nextGaussian();
            x[i][1] = skill + 0.8 * a[i] + 0.5 * r.nextGaussian();   // a proxy of the attribute
            x[i][2] = a[i] + 0.7 * r.nextGaussian();
            x[i][3] = r.nextGaussian();
            y[i] = skill + 0.8 * a[i] + 0.5 * r.nextGaussian() > 0.4 ? 1 : 0;   // historically biased labels
        }
        return new Applicants(x, y, a);
    }

    private static Model classifier(int inputs, long seed) {
        return Model.sequential().input(inputs).dense(16, Activation.RELU).dense(1, Activation.IDENTITY).seed(seed)
                .build();
    }

    /**
     * Trains and scores one fairness intervention.
     *
     * @param method the intervention
     * @param seed the seed
     * @return the test scores
     */
    public static FairnessScores fairness(FairnessMethod method, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Applicants train = applicants(4000, r);
        Applicants held = applicants(2000, r);
        Applicants test = applicants(4000, r);
        int n = train.y().length;
        Tensor x = matrix(train.x());
        Tensor y = vector(train.y()).reshape(n, 1);
        Tensor a = vector(train.a()).reshape(n, 1);
        double[] weights = method == FairnessMethod.REWEIGHING ? FairnessInterventions.reweighing(train.y(), train.a())
                : null;
        Tensor w = weights == null ? null : vector(weights).reshape(n, 1);
        Model clf = classifier(4, seed);
        Model adv = Model.sequential().input(3).dense(8, Activation.RELU).dense(1, Activation.IDENTITY).seed(seed + 1)
                .build();
        Optimizer opt = Optimizer.adam(3e-3f);
        Optimizer advOpt = Optimizer.adam(3e-3f);
        int batch = 200;
        for (int s = 0; s < 1500; s++) {
            int from = r.nextInt(n - batch);
            Tensor bx = x.slice(0, from, from + batch);
            Tensor by = y.slice(0, from, from + batch);
            Var logit = clf.forward(Var.of(bx), true);
            Var task = w == null ? FairnessInterventions.binaryCrossEntropy(logit, by)
                    : logit.softplus().sub(logit.mul(by)).mul(w.slice(0, from, from + batch)).mean();
            if (method == FairnessMethod.ADVERSARIAL) {
                Tensor ba = a.slice(0, from, from + batch);
                Var advLoss = FairnessInterventions.binaryCrossEntropy(
                        adv.forward(FairnessInterventions.adversaryInput(logit, by, true)), ba);
                float alpha = Math.min(1f, s / 300f);
                FairnessInterventions.adversarialLoss(task, advLoss, alpha).backward();
                adv.parameters().forEach(Var::zeroGrad);
                opt.step(clf.parameters());
                FairnessInterventions.binaryCrossEntropy(
                        adv.forward(FairnessInterventions.adversaryInput(Var.of(logit.value()), by, true)), ba).backward();
                advOpt.step(adv.parameters());
            } else {
                task.backward();
                opt.step(clf.parameters());
            }
        }
        double[] thresholds = {0, 0};
        if (method == FairnessMethod.EQUALIZED_ODDS_POST) {
            double[] heldScores = clf.forward(Var.of(matrix(held.x()))).value().toDoubleArray();
            thresholds = FairnessInterventions.equalizedOddsThresholds(heldScores, held.y(), held.a(), 0.02);
        }
        double[] scores = clf.forward(Var.of(matrix(test.x()))).value().toDoubleArray();
        int[] pred = new int[scores.length];
        int correct = 0;
        for (int i = 0; i < scores.length; i++) {
            pred[i] = scores[i] >= thresholds[test.a()[i]] ? 1 : 0;
            correct += pred[i] == test.y()[i] ? 1 : 0;
        }
        return new FairnessScores(correct / (double) scores.length,
                FairnessInterventions.demographicParityGap(pred, test.a()),
                FairnessInterventions.equalizedOddsGap(pred, test.y(), test.a()));
    }

    // ------------------------------------------------------------------ spurious correlation

    /** The group-robust methods. */
    public enum RobustMethod {
        /** Pooled ERM. */
        ERM,
        /** IRMv1 over the two environments. */
        IRM,
        /** GroupDRO over the four (class, colour) groups. */
        GROUP_DRO,
        /** V-REx: risk plus the variance of the two environments' risks (environment labels only). */
        VREX,
        /** Just Train Twice: upsample a short ERM run's training errors (no group or environment labels). */
        JTT
    }

    /**
     * Spurious-correlation scores.
     *
     * @param average mean test accuracy
     * @param worstGroup the lowest accuracy among the (class, colour) groups
     */
    public record RobustScores(double average, double worstGroup) {
    }

    private record Coloured(Tensor x, Tensor y, int[] group) {
    }

    private static Coloured coloured(int n, double agreement, long seed) {
        Tensor[] g = new ShapeImages(8, 0.25f).generate(4 * n, seed);
        float[] img = g[0].toArray();
        float[] ids = g[1].toArray();
        SplittableRandom r = new SplittableRandom(seed + 1);
        float[] out = new float[n * 128];
        float[] y = new float[n];
        int[] group = new int[n];
        int k = 0;
        for (int i = 0; i < ids.length && k < n; i++) {
            if (ids[i] > 1) {
                continue;   // keep the horizontal (0) and vertical (1) bars
            }
            int cls = (int) ids[i];
            int colour = r.nextDouble() < agreement ? cls : 1 - cls;
            System.arraycopy(img, i * 64, out, k * 128 + colour * 64, 64);
            for (int p = 0; p < 64; p++) {   // the other channel gets noise only
                out[k * 128 + (1 - colour) * 64 + p] = (float) (0.25 * r.nextGaussian());
            }
            y[k] = cls;
            group[k] = 2 * cls + colour;
            k++;
        }
        return new Coloured(Tensor.of(Shape.of(n, 128), out), Tensor.of(Shape.of(n), y), group);
    }

    /**
     * Trains and scores one method on the coloured shapes.
     *
     * @param method the method
     * @param seed the seed
     * @return the test scores
     */
    public static RobustScores spurious(RobustMethod method, long seed) {
        Coloured env1 = coloured(1000, 0.9, seed);
        Coloured env2 = coloured(1000, 0.8, seed + 10);
        Coloured test = coloured(2000, 0.5, seed + 20);
        Model m = Model.sequential().input(128).dense(64, Activation.RELU).dense(2, Activation.IDENTITY).seed(seed)
                .build();
        Optimizer opt = Optimizer.adam(1e-3f);
        GroupRobustTraining.GroupDro dro = new GroupRobustTraining.GroupDro(4, 0.01);
        SplittableRandom r = new SplittableRandom(seed);
        int half = 50;
        GroupRobustTraining.JustTrainTwice jtt = method == RobustMethod.JTT ? identifyErrors(env1, env2, seed) : null;
        Tensor pooledX = Tensor.concat(0, env1.x(), env2.x());
        Tensor pooledY = Tensor.concat(0, env1.y(), env2.y());
        for (int s = 0; s < 1500; s++) {
            int f1 = r.nextInt(1000 - half);
            int f2 = r.nextInt(1000 - half);
            Tensor x1 = env1.x().slice(0, f1, f1 + half);
            Tensor y1 = env1.y().slice(0, f1, f1 + half);
            Tensor x2 = env2.x().slice(0, f2, f2 + half);
            Tensor y2 = env2.y().slice(0, f2, f2 + half);
            Var loss;
            switch (method) {
                case ERM -> loss = Functional.crossEntropy(m.forward(Var.of(Tensor.concat(0, x1, x2)), true),
                        Tensor.concat(0, y1, y2));
                case IRM -> {
                    Var l1 = m.forward(Var.of(x1), true);
                    Var l2 = m.forward(Var.of(x2), true);
                    Var risk = Functional.crossEntropy(l1, y1).add(Functional.crossEntropy(l2, y2)).mul(0.5f);
                    Var penalty = GroupRobustTraining.irmPenalty(l1, z -> Functional.crossEntropy(z, y1))
                            .add(GroupRobustTraining.irmPenalty(l2, z -> Functional.crossEntropy(z, y2)));
                    float weight = s < 500 ? 1f : 10f;
                    loss = risk.add(penalty.mul(weight));
                    if (weight > 1) {
                        loss = loss.mul(1f / weight);   // keep the step size comparable
                    }
                }
                case VREX -> {
                    Var r1 = Functional.crossEntropy(m.forward(Var.of(x1), true), y1);
                    Var r2 = Functional.crossEntropy(m.forward(Var.of(x2), true), y2);
                    float weight = s < 500 ? 1f : 100f;   // the paper's penalty warm-up
                    loss = r1.add(r2).mul(0.5f).add(GroupRobustTraining.vrexPenalty(List.of(r1, r2)).mul(weight))
                            .mul(1f / weight);
                }
                case JTT -> {
                    Tensor idx = rowsTensor(jtt.sample(2 * half, r));
                    loss = Functional.crossEntropy(m.forward(Var.of(pooledX.gather(0, idx)), true),
                            pooledY.gather(0, idx));
                }
                default -> {
                    Tensor bx = Tensor.concat(0, x1, x2);
                    Tensor by = Tensor.concat(0, y1, y2);
                    int[] groups = new int[2 * half];
                    for (int i = 0; i < half; i++) {
                        groups[i] = env1.group()[f1 + i];
                        groups[half + i] = env2.group()[f2 + i];
                    }
                    Var logits = m.forward(Var.of(bx), true);
                    List<Var> losses = new ArrayList<>();
                    for (int g = 0; g < 4; g++) {
                        int gg = g;
                        int[] rows = java.util.stream.IntStream.range(0, groups.length).filter(i -> groups[i] == gg)
                                .toArray();
                        losses.add(rows.length == 0 ? null
                                : Functional.crossEntropy(logits.gather(0, rowsTensor(rows)), by.gather(0, rowsTensor(rows))));
                    }
                    loss = dro.loss(losses);
                }
            }
            loss.backward();
            opt.step(m.parameters());
        }
        float[] pred = m.forward(Var.of(test.x())).value().argmax(1).toArray();
        float[] y = test.y().toArray();
        double[] hit = new double[4];
        double[] cnt = new double[4];
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            boolean ok = pred[i] == y[i];
            correct += ok ? 1 : 0;
            hit[test.group()[i]] += ok ? 1 : 0;
            cnt[test.group()[i]]++;
        }
        double worst = 1;
        for (int g = 0; g < 4; g++) {
            worst = Math.min(worst, hit[g] / cnt[g]);
        }
        return new RobustScores(correct / (double) y.length, worst);
    }

    /** JTT's first run: 300 steps of pooled ERM, then its training mistakes, upsampled 20 times. */
    private static GroupRobustTraining.JustTrainTwice identifyErrors(Coloured env1, Coloured env2, long seed) {
        Tensor x = Tensor.concat(0, env1.x(), env2.x());
        Tensor y = Tensor.concat(0, env1.y(), env2.y());
        Model id = Model.sequential().input(128).dense(64, Activation.RELU).dense(2, Activation.IDENTITY)
                .seed(seed + 7).build();
        Optimizer opt = Optimizer.adam(1e-3f);
        SplittableRandom r = new SplittableRandom(seed + 7);
        int n = x.shape().dim(0);
        for (int s = 0; s < 300; s++) {
            int[] rows = new int[100];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = r.nextInt(n);
            }
            Tensor idx = rowsTensor(rows);
            Functional.crossEntropy(id.forward(Var.of(x.gather(0, idx)), true), y.gather(0, idx)).backward();
            opt.step(id.parameters());
        }
        float[] pred = id.forward(Var.of(x)).value().argmax(1).toArray();
        float[] truth = y.toArray();
        int[] p = new int[n];
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = (int) pred[i];
            t[i] = (int) truth[i];
        }
        return new GroupRobustTraining.JustTrainTwice(GroupRobustTraining.JustTrainTwice.errorSet(p, t), 20);
    }

    private static Tensor rowsTensor(int[] rows) {
        float[] v = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            v[i] = rows[i];
        }
        return Tensor.of(Shape.of(rows.length), v);
    }
}
