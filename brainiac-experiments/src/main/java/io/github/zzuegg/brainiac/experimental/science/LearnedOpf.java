package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Learning-to-optimise for DC optimal power flow on a {@link DcPowerFlow}.
 *
 * <ul>
 *   <li>{@link Method#MLP_DIRECT}: a network maps loads to every generator's output (a
 *       sigmoid times its capacity), trained on LP solutions; power balance and line limits
 *       are not enforced.</li>
 *   <li>{@link Method#DEEP_OPF} (Pan et al., IEEE TPS 2021): the network predicts the
 *       non-slack generators only and the slack generator takes the power balance, so the
 *       equality holds by construction.</li>
 *   <li>{@link Method#DC3} (Donti, Rolnick and Kolter, ICLR 2021): the same equality
 *       completion, trained on the DC3 loss (cost plus a squared penalty on the inequality
 *       violations of the completed dispatch, plus a supervised term towards the LP
 *       solution), and at inference followed by gradient steps on the inequality violation
 *       through the completion.</li>
 * </ul>
 * {@link FeasibilityClassifier} predicts from the loads whether any dispatch is feasible and
 * repairs an infeasible profile with a counterfactual load change (Mohammadian, Van Boven and
 * Baker, 2025), with {@link ModelAgnosticExplainer#counterfactual} as the search (WP1's recourse
 * block was not available in this worktree).
 */
public final class LearnedOpf {

    /** The learned dispatch. */
    public enum Method {
        /** Direct prediction of all outputs. */
        MLP_DIRECT,
        /** Non-slack prediction with balance completion. */
        DEEP_OPF,
        /** Completion, DC3 loss and inequality correction. */
        DC3
    }

    private final Method method;
    private final DcPowerFlow pf;
    private final SmallNet net;
    private int correctionSteps = 1000;

    /**
     * An untrained model.
     *
     * @param method the method
     * @param pf the network's power flow
     * @param hidden the hidden width (two layers)
     * @param seed the seed
     */
    public LearnedOpf(Method method, DcPowerFlow pf, int hidden, long seed) {
        this.method = method;
        this.pf = pf;
        int gens = pf.grid().generators();
        int outputs = method == Method.MLP_DIRECT ? gens : gens - 1;
        this.net = new SmallNet(new int[] {pf.grid().buses(), hidden, hidden, outputs}, seed);
    }

    static double[] normalise(ScienceSims.PowerGrid grid, double[] load) {
        double[] x = new double[load.length];
        for (int b = 0; b < load.length; b++) {
            double base = grid.baseLoad()[b];
            x[b] = base > 0 ? (load[b] / base - 1) / 0.3 : 0;
        }
        return x;
    }

    private Tensor row(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return Tensor.of(Shape.of(1, v.length), f);
    }

    /**
     * Trains full-batch with Adam.
     *
     * @param loads training load profiles (feasible ones)
     * @param dispatch their LP dispatches
     * @param steps Adam steps
     * @param learningRate the step
     * @return this model
     */
    public LearnedOpf train(double[][] loads, double[][] dispatch, int steps, double learningRate) {
        ScienceSims.PowerGrid grid = pf.grid();
        int n = loads.length;
        int gens = grid.generators();
        int buses = grid.buses();
        int m = grid.lines();
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            x[i] = normalise(grid, loads[i]);
        }
        Tensor xt = SmallNet.tensor(x);
        double[] gmax = grid.genMax();
        int outputs = method == Method.MLP_DIRECT ? gens : gens - 1;
        double[] capOut = new double[outputs];
        for (int j = 0; j < outputs; j++) {
            capOut[j] = gmax[method == Method.MLP_DIRECT ? j : j + 1];
        }
        Tensor cap = row(capOut);
        // targets scaled by capacity
        double[][] target = new double[n][outputs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < outputs; j++) {
                int g = method == Method.MLP_DIRECT ? j : j + 1;
                target[i][j] = dispatch[i][g] / gmax[g];
            }
        }
        Tensor yt = SmallNet.tensor(target);
        // completion g = z E + D e0 (E: [gens-1][gens], slack column -1)
        float[] e = new float[(gens - 1) * gens];
        for (int j = 0; j < gens - 1; j++) {
            e[j * gens] = -1;
            e[j * gens + j + 1] = 1;
        }
        Tensor eT = Tensor.of(Shape.of(gens - 1, gens), e);
        float[] dCol = new float[n * gens];
        double[][] loadFlowsArr = new double[n][m];
        double total = 0;
        for (int i = 0; i < n; i++) {
            dCol[i * gens] = (float) DcPowerFlow.sum(loads[i]);
            total += DcPowerFlow.sum(loads[i]) / n;
            loadFlowsArr[i] = pf.flows(new double[gens], loads[i]);
        }
        Tensor dT = Tensor.of(Shape.of(n, gens), dCol);
        Tensor loadFlows = SmallNet.tensor(loadFlowsArr);
        float[] gp = new float[gens * m];
        for (int l = 0; l < m; l++) {
            for (int g = 0; g < gens; g++) {
                gp[g * m + l] = (float) pf.generatorPtdf()[l][g];
            }
        }
        Tensor gpT = Tensor.of(Shape.of(gens, m), gp);
        float[] lim = new float[m];
        for (int l = 0; l < m; l++) {
            lim[l] = (float) (grid.limit()[l] / total);
        }
        Tensor limT = Tensor.of(Shape.of(1, m), lim);
        float[] costs = new float[gens];
        float[] gm = new float[gens];
        double maxCost = Arrays.stream(grid.genCost()).max().orElse(1);
        for (int g = 0; g < gens; g++) {
            costs[g] = (float) (grid.genCost()[g] / maxCost);
            gm[g] = (float) (gmax[g] / total);
        }
        Tensor costT = Tensor.of(Shape.of(gens, 1), costs);
        Tensor gmT = Tensor.of(Shape.of(1, gens), gm);
        Optimizer opt = Optimizer.adam((float) learningRate);
        float invTotal = (float) (1 / total);
        for (int s = 0; s < steps; s++) {
            opt.minimize(net.parameters(), () -> {
                Var u = net.forward(Var.of(xt)).sigmoid();
                Var sup = u.sub(Var.of(yt)).square().mean();
                if (method != Method.DC3) {
                    return sup;
                }
                // dispatch in units of the mean total load
                Var g = u.mul(cap).mul(invTotal).matmul(eT).add(Var.of(dT).mul(invTotal));
                Var f = g.matmul(gpT).add(Var.of(loadFlows).mul(invTotal));
                Var lineV = f.sub(limT).relu().square().add(f.neg().sub(limT).relu().square()).sum(1, true);
                Var boundV = g.sub(gmT).relu().square().add(g.neg().relu().square()).sum(1, true);
                Var cost = g.matmul(costT);
                return sup.add(cost.mean().mul(0.01f)).add(lineV.add(boundV).mean().mul(100f));
            });
        }
        return this;
    }

    /**
     * @param steps inequality-correction steps for {@link Method#DC3} (default 1000)
     * @return this model
     */
    public LearnedOpf correctionSteps(int steps) {
        this.correctionSteps = steps;
        return this;
    }

    /**
     * Predicts dispatches.
     *
     * @param loads load profiles
     * @return {@code [scenario][generator]} outputs (MW)
     */
    public double[][] predict(double[][] loads) {
        ScienceSims.PowerGrid grid = pf.grid();
        int gens = grid.generators();
        double[][] x = new double[loads.length][];
        for (int i = 0; i < loads.length; i++) {
            x[i] = normalise(grid, loads[i]);
        }
        double[][] raw = net.predict(x);
        double[][] out = new double[loads.length][gens];
        for (int i = 0; i < loads.length; i++) {
            if (method == Method.MLP_DIRECT) {
                for (int g = 0; g < gens; g++) {
                    out[i][g] = sigmoid(raw[i][g]) * grid.genMax()[g];
                }
                continue;
            }
            double[] z = new double[gens - 1];
            for (int j = 0; j < gens - 1; j++) {
                z[j] = sigmoid(raw[i][j]) * grid.genMax()[j + 1];
            }
            if (method == Method.DC3) {
                correct(z, loads[i]);
            }
            out[i] = complete(z, loads[i]);
        }
        return out;
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    private double[] complete(double[] z, double[] load) {
        double[] g = new double[z.length + 1];
        double s = 0;
        for (int j = 0; j < z.length; j++) {
            g[j + 1] = z[j];
            s += z[j];
        }
        g[0] = DcPowerFlow.sum(load) - s;
        return g;
    }

    private void correct(double[] z, double[] load) {
        ScienceSims.PowerGrid grid = pf.grid();
        double[][] gp = pf.generatorPtdf();
        int gens = grid.generators();
        for (int s = 0; s < correctionSteps; s++) {
            double[] g = complete(z, load);
            double[] f = pf.flows(g, load);
            double[] dg = new double[gens];
            double v = 0;
            for (int l = 0; l < f.length; l++) {
                double exc = Math.abs(f[l]) - grid.limit()[l];
                if (exc > 0) {
                    v += 0.5 * exc * exc;
                    for (int k = 0; k < gens; k++) {
                        dg[k] += Math.signum(f[l]) * exc * gp[l][k];
                    }
                }
            }
            for (int k = 0; k < gens; k++) {
                double hi = g[k] - grid.genMax()[k];
                if (hi > 0) {
                    v += 0.5 * hi * hi;
                    dg[k] += hi;
                }
                if (g[k] < 0) {
                    v += 0.5 * g[k] * g[k];
                    dg[k] += g[k];
                }
            }
            if (v < 1e-10) {
                return;
            }
            double[] grad = new double[z.length];
            double norm = 0;
            for (int j = 0; j < z.length; j++) {
                grad[j] = dg[j + 1] - dg[0];
                norm += grad[j] * grad[j];
            }
            if (norm < 1e-18) {
                return;
            }
            // Polyak step towards zero violation
            double step = v / norm;
            for (int j = 0; j < z.length; j++) {
                z[j] = Math.min(grid.genMax()[j + 1], Math.max(0, z[j] - step * grad[j]));
            }
        }
    }

    /**
     * A classifier of load profiles into feasible and infeasible, with counterfactual repair.
     */
    public static final class FeasibilityClassifier {

        private final DcPowerFlow pf;
        private final SmallNet net;

        /**
         * An untrained classifier.
         *
         * @param pf the power flow
         * @param hidden the hidden width
         * @param seed the seed
         */
        public FeasibilityClassifier(DcPowerFlow pf, int hidden, long seed) {
            this.pf = pf;
            this.net = new SmallNet(new int[] {pf.grid().buses(), hidden, hidden, 1}, seed);
        }

        /**
         * Trains on labelled profiles by logistic loss.
         *
         * @param loads profiles
         * @param feasible labels from the LP
         * @param steps Adam steps
         * @param learningRate the step
         * @return this classifier
         */
        public FeasibilityClassifier train(double[][] loads, boolean[] feasible, int steps, double learningRate) {
            double[][] x = new double[loads.length][];
            float[] y = new float[loads.length];
            for (int i = 0; i < loads.length; i++) {
                x[i] = normalise(pf.grid(), loads[i]);
                y[i] = feasible[i] ? 1 : 0;
            }
            Tensor xt = SmallNet.tensor(x);
            Tensor yt = Tensor.of(Shape.of(loads.length, 1), y);
            Optimizer opt = Optimizer.adam((float) learningRate);
            for (int s = 0; s < steps; s++) {
                opt.minimize(net.parameters(), () -> {
                    Var z = net.forward(Var.of(xt));
                    return z.softplus().sub(z.mul(Var.of(yt))).mean();
                });
            }
            return this;
        }

        /**
         * @param loads profiles
         * @return the log-odds of feasibility per profile
         */
        public double[] logit(double[][] loads) {
            double[][] x = new double[loads.length][];
            for (int i = 0; i < loads.length; i++) {
                x[i] = normalise(pf.grid(), loads[i]);
            }
            double[][] o = net.predict(x);
            double[] out = new double[loads.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = o[i][0];
            }
            return out;
        }

        /**
         * Proposes a sparse load change that the classifier deems feasible.
         *
         * @param load an infeasible profile
         * @param margin how far past the decision boundary to aim, in log-odds
         * @return the repaired profile, or null when the search did not flip the classifier
         */
        public double[] repair(double[] load, double margin) {
            Function<double[][], double[]> model = this::logit;
            double[] scale = new double[load.length];
            for (int b = 0; b < load.length; b++) {
                scale[b] = Math.max(1, 0.3 * pf.grid().baseLoad()[b]);
            }
            ModelAgnosticExplainer.Counterfactual cf = new ModelAgnosticExplainer(model)
                    .counterfactual(load, scale, 0, margin, 25, 0.05);
            if (!cf.valid()) {
                return null;
            }
            double[] out = cf.changed();
            for (int b = 0; b < out.length; b++) {
                out[b] = Math.max(0, out[b]);
            }
            return out;
        }
    }
}
