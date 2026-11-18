package io.github.zzuegg.brainiac.experiments.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierFeatures;
import io.github.zzuegg.brainiac.experimental.sciml.CollocationSampler;
import io.github.zzuegg.brainiac.experimental.sciml.ConstrainedFields;
import io.github.zzuegg.brainiac.experimental.sciml.NystromNewtonCg;
import io.github.zzuegg.brainiac.experimental.sciml.PinnNaturalGradient;
import io.github.zzuegg.brainiac.experimental.sciml.PdeProblems;
import io.github.zzuegg.brainiac.experimental.sciml.PhysicsInformedNetwork;
import io.github.zzuegg.brainiac.experimental.sciml.PhysicsInformedKan;
import io.github.zzuegg.brainiac.experimental.sciml.PinnTraining;
import io.github.zzuegg.brainiac.experimental.sciml.PirateNet;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.DoubleBinaryOperator;
import java.util.function.Supplier;

/**
 * The physics-informed shootout (wave 8, items 8.3.1-8.3.3 and 8.3.17): PINNs on the classic
 * benchmark PDEs, scored by relative L2 error against {@link PdeProblems} reference solutions.
 *
 * <ul>
 *   <li>1-D heat {@code u_t = 0.1 u_xx} on {@code [0,1]^2} with {@code u0 = sin(pi x)} (exact);
 *   <li>1-D Poisson {@code -u'' = f} with {@code u = sin(pi x) + 0.5 sin(3 pi x)} (finite
 *       differences);
 *   <li>viscous Burgers {@code u_t + u u_x = (0.01/pi) u_xx}, {@code u0 = -sin(pi x)} on
 *       {@code [-1,1] x [0,1]} (pseudo-spectral, 1024 modes);
 *   <li>Allen-Cahn {@code u_t = 1e-4 u_xx + 5u - 5u^3}, {@code u0 = x^2 cos(pi x)}, periodic
 *       (pseudo-spectral);
 *   <li>advection {@code u_t + beta u_x = 0}, {@code u0 = sin x} on {@code [0, 2pi]}, periodic,
 *       a known failure case at large {@code beta}.
 * </ul>
 *
 * <p>Baselines: the same network fit to 100 noisy solution samples (no physics), and a
 * coarse run of the reference solver (the error floor of classical numerics at a similar cost).
 */
public final class PdeShootout {

    private PdeShootout() {
    }

    /**
     * A benchmark PDE with its reference solution.
     *
     * @param name the name
     * @param problem the PINN problem
     * @param reference the reference solution {@code u(x, t)} ({@code t} ignored if steady)
     * @param evalPoints the points the error is measured at
     * @param equation the periodic equation for the coarse-solver baseline, or {@code null}
     * @param coefficient its coefficient
     */
    public record Pde(String name, PinnTraining.Problem problem, DoubleBinaryOperator reference,
            double[][] evalPoints, PdeProblems.Equation equation, double coefficient) {

        /** @return the input coordinates of the field */
        public int inputs() {
            return problem.domain().dimension();
        }

        /** @return the reference at the evaluation points */
        public double[] referenceValues() {
            double[] v = new double[evalPoints.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = reference.applyAsDouble(evalPoints[i][0], evalPoints[i].length > 1 ? evalPoints[i][1] : 0);
            }
            return v;
        }
    }

    /**
     * A shootout row.
     *
     * @param method the method
     * @param error relative L2 error against the reference
     * @param millis wall time
     */
    public record Row(String method, double error, double millis) {
    }

    private static final SplittableRandom CONDITION_RANDOM = new SplittableRandom(1234);

    private static double[][] evalGrid(double x0, double x1, double t0, double t1, int nx, int nt) {
        return CollocationSampler.box(new double[] {x0, t0}, new double[] {x1, t1}).grid(nx, nt);
    }

    private static double[] apply(double[][] pts, DoubleBinaryOperator f) {
        double[] v = new double[pts.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = f.applyAsDouble(pts[i][0], pts[i][1]);
        }
        return v;
    }

    private static List<PinnTraining.Term> initialAndDirichlet(CollocationSampler box, DoubleBinaryOperator u0,
            int initial, int boundary) {
        SplittableRandom r = new SplittableRandom(CONDITION_RANDOM.split().nextLong());
        double[][] ic = box.face(1, false, initial, r);
        double[][] left = box.face(0, false, boundary, r);
        double[][] right = box.face(0, true, boundary, r);
        List<PinnTraining.Term> terms = new ArrayList<>();
        terms.add(PinnTraining.dirichlet("initial", ic, apply(ic, u0)));
        terms.add(PinnTraining.dirichlet("left", left, new double[boundary]));
        terms.add(PinnTraining.dirichlet("right", right, new double[boundary]));
        return terms;
    }

    private static List<PinnTraining.Term> initialAndPeriodic(CollocationSampler box, DoubleBinaryOperator u0,
            int initial, int boundary) {
        SplittableRandom r = new SplittableRandom(CONDITION_RANDOM.split().nextLong());
        double[][] ic = box.face(1, false, initial, r);
        double[][] left = box.face(0, false, boundary, r);
        double[][] right = new double[boundary][];
        for (int i = 0; i < boundary; i++) {
            right[i] = new double[] {box.high(0), left[i][1]};
        }
        List<PinnTraining.Term> terms = new ArrayList<>();
        terms.add(PinnTraining.dirichlet("initial", ic, apply(ic, u0)));
        terms.add(PinnTraining.periodic("periodic", left, right, 0));
        return terms;
    }

    /**
     * The heat equation {@code u_t = 0.1 u_xx}, exact solution {@code e^(-0.1 pi^2 t) sin(pi x)}.
     *
     * @return the PDE
     */
    public static Pde heat() {
        double kappa = 0.1;
        CollocationSampler box = CollocationSampler.box(new double[] {0, 0}, new double[] {1, 1});
        DoubleBinaryOperator exact = (x, t) -> Math.exp(-kappa * Math.PI * Math.PI * t) * Math.sin(Math.PI * x);
        PinnTraining.Problem p = new PinnTraining.Problem(d -> d.first(1).sub(d.second(0).mul((float) kappa)), box, 1,
                initialAndDirichlet(box, (x, t) -> Math.sin(Math.PI * x), 64, 32));
        return new Pde("heat", p, exact, evalGrid(0, 1, 0, 1, 33, 11), PdeProblems.Equation.HEAT, kappa);
    }

    /**
     * The 1-D Poisson problem {@code -u'' = f} on {@code [0, 1]} with zero boundary values;
     * the reference is the finite-difference solution on 1023 points.
     *
     * @return the PDE
     */
    public static Pde poisson() {
        CollocationSampler box = CollocationSampler.box(new double[] {0}, new double[] {1});
        double pi2 = Math.PI * Math.PI;
        int n = 1023;
        double[] f = new double[n];
        for (int i = 0; i < n; i++) {
            double x = (i + 1.0) / (n + 1);
            f[i] = pi2 * Math.sin(Math.PI * x) + 4.5 * pi2 * Math.sin(3 * Math.PI * x);
        }
        double[] u = PdeProblems.poisson1d(f);
        DoubleBinaryOperator ref = (x, t) -> {
            double s = x * (n + 1) - 1;
            int i = (int) Math.floor(s);
            double a = i < 0 ? 0 : i >= n ? 0 : u[i];
            double b = i + 1 >= n ? 0 : i + 1 < 0 ? 0 : u[i + 1];
            double w = s - i;
            return a * (1 - w) + b * w;
        };
        PhysicsInformedNetwork.Residual r = d -> {
            Var x = d.coordinate(0);
            Var src = x.mul((float) Math.PI).sin().mul((float) pi2)
                    .add(x.mul((float) (3 * Math.PI)).sin().mul((float) (4.5 * pi2)));
            // divided by pi^2 so the residual and boundary terms start at similar scales
            return d.second(0).add(src).mul((float) (1 / pi2));
        };
        List<PinnTraining.Term> bc = List.of(
                PinnTraining.dirichlet("ends", new double[][] {{0}, {1}}, new double[] {0, 0}));
        double[][] eval = new double[101][];
        for (int i = 0; i <= 100; i++) {
            eval[i] = new double[] {i / 100.0};
        }
        return new Pde("poisson", new PinnTraining.Problem(r, box, -1, bc), ref, eval, null, 0);
    }

    /**
     * Viscous Burgers' equation with {@code nu = 0.01/pi}, the shock-forming PINN benchmark.
     *
     * @return the PDE
     */
    public static Pde burgers() {
        double nu = 0.01 / Math.PI;
        int n = 1024;
        double[] u0 = new double[n];
        for (int j = 0; j < n; j++) {
            u0[j] = -Math.sin(Math.PI * (-1 + 2.0 * j / n));
        }
        PdeProblems.Solution sol = PdeProblems.solve(PdeProblems.Equation.BURGERS, u0, -1, 2, nu, 1, 101, 1e-3);
        CollocationSampler box = CollocationSampler.box(new double[] {-1, 0}, new double[] {1, 1});
        PinnTraining.Problem p = new PinnTraining.Problem(
                d -> d.first(1).add(d.u().mul(d.first(0))).sub(d.second(0).mul((float) nu)), box, 1,
                initialAndDirichlet(box, (x, t) -> -Math.sin(Math.PI * x), 100, 50));
        return new Pde("burgers", p, sol::at, evalGrid(-1, 1, 0, 1, 65, 11), PdeProblems.Equation.BURGERS, nu);
    }

    /**
     * Allen-Cahn {@code u_t - 1e-4 u_xx + 5u^3 - 5u = 0} with {@code u0 = x^2 cos(pi x)} on
     * {@code [-1, 1]}, periodic.
     *
     * @return the PDE
     */
    public static Pde allenCahn() {
        int n = 512;
        double[] u0 = new double[n];
        for (int j = 0; j < n; j++) {
            double x = -1 + 2.0 * j / n;
            u0[j] = x * x * Math.cos(Math.PI * x);
        }
        PdeProblems.Solution sol = PdeProblems.solve(PdeProblems.Equation.ALLEN_CAHN, u0, -1, 2, 1e-4, 1, 101, 1e-3);
        CollocationSampler box = CollocationSampler.box(new double[] {-1, 0}, new double[] {1, 1});
        PinnTraining.Problem p = new PinnTraining.Problem(d -> {
            Var u = d.u();
            return d.first(1).sub(d.second(0).mul(1e-4f)).add(u.mul(u).mul(u).mul(5f)).sub(u.mul(5f));
        }, box, 1, initialAndPeriodic(box, (x, t) -> x * x * Math.cos(Math.PI * x), 100, 50));
        return new Pde("allen-cahn", p, sol::at, evalGrid(-1, 1, 0, 1, 65, 11), PdeProblems.Equation.ALLEN_CAHN,
                1e-4);
    }

    /**
     * Advection {@code u_t + beta u_x = 0} with {@code u0 = sin x} on {@code [0, 2 pi]},
     * periodic; exact solution {@code sin(x - beta t)}.
     *
     * @param beta the speed; large values make the plain PINN fail
     * @return the PDE
     */
    public static Pde advection(double beta) {
        double l = 2 * Math.PI;
        CollocationSampler box = CollocationSampler.box(new double[] {0, 0}, new double[] {l, 1});
        PinnTraining.Problem p = new PinnTraining.Problem(d -> d.first(1).add(d.first(0).mul((float) beta)), box, 1,
                initialAndPeriodic(box, (x, t) -> Math.sin(x), 100, 50));
        return new Pde("advection", p, (x, t) -> Math.sin(x - beta * t), evalGrid(0, l, 0, 1, 65, 11),
                PdeProblems.Equation.ADVECTION, beta);
    }

    /**
     * Relative L2 error of a trained field against the reference.
     *
     * @param pde the problem
     * @param field the field network
     * @return the error
     */
    public static double error(Pde pde, Layer field) {
        double[] pred = PhysicsInformedNetwork.of(field).predict(pde.evalPoints());
        return PdeProblems.relativeL2(pred, pde.referenceValues());
    }

    /**
     * Trains a PINN and scores it.
     *
     * @param name the row name
     * @param pde the problem
     * @param field the field network
     * @param options the recipe
     * @param seed the seed
     * @return the row
     */
    public static Row pinn(String name, Pde pde, Layer field, PinnTraining.Options options, long seed) {
        PhysicsInformedNetwork net = PhysicsInformedNetwork.of(field);
        PinnTraining.Result r = PinnTraining.train(net, pde.problem(), options, seed);
        return new Row(name, error(pde, field), r.millis());
    }

    /**
     * The no-physics baseline: the same network fit by Adam to noisy solution samples.
     *
     * @param pde the problem
     * @param field the field network
     * @param samples the samples, for example 100
     * @param noise the noise standard deviation relative to the solution's RMS
     * @param steps Adam steps
     * @param seed the seed
     * @return the row
     */
    public static Row dataFit(Pde pde, Layer field, int samples, double noise, int steps, long seed) {
        long start = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        double[][] pts = pde.problem().domain().uniform(samples, r);
        double[] y = new double[samples];
        double rms = 0;
        for (int i = 0; i < samples; i++) {
            y[i] = pde.reference().applyAsDouble(pts[i][0], pts[i].length > 1 ? pts[i][1] : 0);
            rms += y[i] * y[i];
        }
        rms = Math.sqrt(rms / samples);
        for (int i = 0; i < samples; i++) {
            y[i] += noise * rms * r.nextGaussian();
        }
        Tensor x = CollocationSampler.tensor(pts);
        Tensor t = CollocationSampler.column(y);
        Optimizer adam = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            Var loss = field.forward(Var.of(x)).sub(t).square().mean();
            loss.backward();
            adam.step(field.parameters());
        }
        return new Row("data-fit MLP", error(pde, field), (System.nanoTime() - start) / 1e6);
    }

    /**
     * The classical baseline at low resolution: the reference solver on a coarse grid
     * (finite differences on 15 points for Poisson; 32 spectral modes otherwise), scored
     * against the fine reference.
     *
     * @param pde the problem
     * @return the row
     */
    public static Row coarseSolver(Pde pde) {
        long start = System.nanoTime();
        double[] pred;
        switch (pde.name()) {
            case "poisson" -> {
                int n = 15;
                double[] f = new double[n];
                double pi2 = Math.PI * Math.PI;
                for (int i = 0; i < n; i++) {
                    double x = (i + 1.0) / (n + 1);
                    f[i] = pi2 * Math.sin(Math.PI * x) + 4.5 * pi2 * Math.sin(3 * Math.PI * x);
                }
                double[] u = PdeProblems.poisson1d(f);
                pred = new double[pde.evalPoints().length];
                for (int i = 0; i < pred.length; i++) {
                    double s = pde.evalPoints()[i][0] * (n + 1) - 1;
                    int k = (int) Math.floor(s);
                    double a = k < 0 || k >= n ? 0 : u[k];
                    double b = k + 1 >= n || k + 1 < 0 ? 0 : u[k + 1];
                    pred[i] = a * (1 - (s - k)) + b * (s - k);
                }
            }
            default -> {
                int n = 32;
                double x0 = pde.problem().domain().low(0);
                double len = pde.problem().domain().high(0) - x0;
                double[] u0 = new double[n];
                for (int j = 0; j < n; j++) {
                    u0[j] = pde.reference().applyAsDouble(x0 + len * j / n, 0);
                }
                PdeProblems.Equation eq = pde.equation();
                double coef = pde.coefficient();
                if (eq == PdeProblems.Equation.HEAT) {
                    // the heat problem is Dirichlet on [0, 1]: extend it oddly to a period of 2
                    double[] ext = new double[2 * n];
                    for (int j = 0; j < 2 * n; j++) {
                        ext[j] = Math.sin(Math.PI * (2.0 * j / (2 * n)));
                    }
                    PdeProblems.Solution s = PdeProblems.solve(eq, ext, 0, 2, coef, 1, 11, 1e-2);
                    pred = apply(pde.evalPoints(), s::at);
                } else {
                    PdeProblems.Solution s = PdeProblems.solve(eq, u0, x0, len, coef, 1, 11, 1e-2);
                    pred = apply(pde.evalPoints(), s::at);
                }
            }
        }
        return new Row("coarse solver", PdeProblems.relativeL2(pred, pde.referenceValues()),
                (System.nanoTime() - start) / 1e6);
    }

    /**
     * The 8.3.2 fixes on one PDE: plain Adam + L-BFGS, then each fix alone and all together.
     *
     * @param pde the problem
     * @param width the MLP width
     * @param depth the MLP hidden layers
     * @param base the shared recipe (steps, collocation)
     * @param causalEpsilon the causal strength
     * @param seed the seed
     * @return one row per recipe
     */
    public static List<Row> fixes(Pde pde, int width, int depth, Supplier<PinnTraining.Options> base,
            double causalEpsilon, long seed) {
        List<Row> rows = new ArrayList<>();
        int every = 50;
        rows.add(pinn("plain", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed), base.get(),
                seed));
        rows.add(pinn("self-adaptive", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed),
                base.get().selfAdaptive(0.02f), seed));
        rows.add(pinn("NTK balancing", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed),
                base.get().ntkBalancing(every, 8), seed));
        rows.add(pinn("causal", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed),
                base.get().causalWeights(causalEpsilon, 16), seed));
        rows.add(pinn("RAR-D", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed),
                base.get().residualResampling(every, 1, 1), seed));
        rows.add(pinn("all four", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed),
                base.get().selfAdaptive(0.02f).ntkBalancing(every, 8).causalWeights(causalEpsilon, 16)
                        .residualResampling(every, 1, 1), seed));
        return rows;
    }

    /**
     * The 8.3.3 depth sweep: a plain tanh MLP with three layers per block and a PirateNet, or
     * a deep physics-informed KAN with and without residual gating, at each depth (in blocks).
     *
     * @param pde the problem
     * @param width the width
     * @param depths the block counts
     * @param options the recipe, created fresh per run
     * @param kans {@code false} for the MLP and PirateNet, {@code true} for the two PIKANs
     * @param seed the seed
     * @return rows named {@code "<model> x<depth>"}
     */
    public static List<Row> depthSweep(Pde pde, int width, int[] depths,
            Supplier<PinnTraining.Options> options, boolean kans, long seed) {
        List<Row> rows = new ArrayList<>();
        for (int d : depths) {
            if (!kans) {
                rows.add(pinn("MLP x" + d, pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, 3 * d, 1, seed),
                        options.get(), seed));
                PirateNet pirate = PirateNet.of(pde.inputs(), 1, width, d, 1.0, seed);
                if (pde.problem().timeAxis() >= 0) {
                    fitInitial(pde, pirate);   // the paper's initial-condition least squares
                }
                rows.add(pinn("PirateNet x" + d, pde, pirate, options.get(), seed));
            } else {
                rows.add(pinn("PIKAN x" + d, pde, PhysicsInformedKan.of(
                        pde.inputs(), 1, width / 2, d, 5, false, seed), options.get(), seed));
                rows.add(pinn("gated PIKAN x" + d, pde, PhysicsInformedKan.of(
                        pde.inputs(), 1, width / 2, d, 5, true, seed), options.get(), seed));
            }
        }
        return rows;
    }

    private static void fitInitial(Pde pde, PirateNet net) {
        double[][] pts = pde.problem().domain().latinHypercube(128, new SplittableRandom(5));
        if (pde.inputs() > 1) {
            for (double[] p : pts) {
                p[1] = pde.problem().domain().low(1);
            }
        }
        double[] v = new double[pts.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = pde.reference().applyAsDouble(pts[i][0], pts[i].length > 1 ? pts[i][1] : 0);
        }
        net.fitOutput(pts, v);
    }

    /**
     * A constraint row: field error and the largest constraint violation.
     *
     * @param method the method
     * @param error relative L2 error of the vector field
     * @param maxViolation the largest {@code |divergence|} (or continuity residual) on the grid
     */
    public record ConstraintRow(String method, double error, double maxViolation) {
    }

    private static Layer chain(Layer first, Layer second) {
        return new Layer() {
            @Override
            public Var forward(Var input) {
                return second.forward(first.forward(input));
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(first.parameters());
                p.addAll(second.parameters());
                return p;
            }
        };
    }

    private static ConstraintRow fitField(String name, Layer field, double[][] samples, double[][] targets,
            double penalty, boolean continuity, double[][] grid, double[][] gridTargets, CollocationSampler box,
            int steps, long seed) {
        Tensor x = CollocationSampler.tensor(samples);
        Tensor y = CollocationSampler.tensor(targets);
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer adam = Optimizer.adam(5e-3f);
        for (int s = 0; s < steps; s++) {
            Var loss = field.forward(Var.param(x)).sub(y).square().mean();
            if (penalty > 0) {
                Var pts = Var.param(CollocationSampler.tensor(box.uniform(128, r)));
                Var v = continuity ? ConstrainedFields.continuityResidual(field, pts)
                        : ConstrainedFields.divergence(field, pts);
                loss = loss.add(v.square().mean().mul((float) penalty));
            }
            loss.backward();
            adam.step(field.parameters());
        }
        Var g = Var.param(CollocationSampler.tensor(grid));
        float[] pred = field.forward(g).value().toArray();
        Var viol = continuity ? ConstrainedFields.continuityResidual(field, g) : ConstrainedFields.divergence(field, g);
        double max = 0;
        for (float v : viol.value().toArray()) {
            max = Math.max(max, Math.abs(v));
        }
        double num = 0;
        double den = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int c = 0; c < 2; c++) {
                double d = pred[2 * i + c] - gridTargets[i][c];
                num += d * d;
                den += gridTargets[i][c] * gridTargets[i][c];
            }
        }
        return new ConstraintRow(name, Math.sqrt(num / den), max);
    }

    /**
     * 8.3.17, incompressible flow: reconstruct the Taylor-Green vortex
     * {@code (sin x cos y, -cos x sin y)} on {@code [0, 2 pi]^2} from sparse noisy velocity
     * samples, with a hard stream-function constraint, soft divergence penalties, an
     * unconstrained MLP and a Fourier-feature MLP.
     *
     * @param samples the velocity samples
     * @param noise the noise standard deviation
     * @param penalties the soft-penalty weights to try
     * @param steps Adam steps
     * @param seed the seed
     * @return one row per method
     */
    public static List<ConstraintRow> taylorGreen(int samples, double noise, double[] penalties, int steps,
            long seed) {
        double l = 2 * Math.PI;
        CollocationSampler box = CollocationSampler.box(new double[] {0, 0}, new double[] {l, l});
        SplittableRandom r = new SplittableRandom(seed);
        double[][] pts = box.uniform(samples, r);
        double[][] tg = new double[samples][];
        for (int i = 0; i < samples; i++) {
            double x = pts[i][0];
            double y = pts[i][1];
            tg[i] = new double[] {Math.sin(x) * Math.cos(y) + noise * r.nextGaussian(),
                -Math.cos(x) * Math.sin(y) + noise * r.nextGaussian()};
        }
        double[][] grid = box.grid(21, 21);
        double[][] gt = new double[grid.length][];
        for (int i = 0; i < grid.length; i++) {
            double x = grid[i][0];
            double y = grid[i][1];
            gt[i] = new double[] {Math.sin(x) * Math.cos(y), -Math.cos(x) * Math.sin(y)};
        }
        List<ConstraintRow> rows = new ArrayList<>();
        rows.add(fitField("stream function (hard)", ConstrainedFields.streamFunction(
                PhysicsInformedNetwork.tanhMlp(2, 32, 2, 1, seed)), pts, tg, 0, false, grid, gt, box, steps, seed));
        rows.add(fitField("unconstrained MLP", PhysicsInformedNetwork.tanhMlp(2, 32, 2, 2, seed), pts, tg, 0, false,
                grid, gt, box, steps, seed));
        for (double p : penalties) {
            rows.add(fitField("soft penalty " + p, PhysicsInformedNetwork.tanhMlp(2, 32, 2, 2, seed), pts, tg, p, false,
                    grid, gt, box, steps, seed));
        }
        rows.add(fitField("Fourier-feature MLP", chain(FourierFeatures.gaussian(2, 16, 0.1f, seed, true),
                PhysicsInformedNetwork.tanhMlp(34, 32, 2, 2, seed)), pts, tg, 0, false, grid, gt, box, steps, seed));
        return rows;
    }

    /**
     * 8.3.17, a 1-D advected density: recover {@code (rho, J)} of a Gaussian bump moving at speed
     * 0.4 on {@code [0, 1] x [0, 1]} from sparse noisy samples, with the exact conservation law,
     * an unconstrained MLP and a soft continuity penalty.
     *
     * @param samples the samples
     * @param noise the noise standard deviation
     * @param steps Adam steps
     * @param seed the seed
     * @return one row per method
     */
    public static List<ConstraintRow> advectedDensity(int samples, double noise, int steps, long seed) {
        double c = 0.4;
        CollocationSampler box = CollocationSampler.box(new double[] {0, 0}, new double[] {1, 1});
        java.util.function.DoubleBinaryOperator rho = (x, t) -> Math.exp(-Math.pow((x - 0.3 - c * t) / 0.15, 2));
        SplittableRandom r = new SplittableRandom(seed);
        double[][] pts = box.uniform(samples, r);
        double[][] tg = new double[samples][];
        for (int i = 0; i < samples; i++) {
            double v = rho.applyAsDouble(pts[i][0], pts[i][1]);
            tg[i] = new double[] {v + noise * r.nextGaussian(), c * v + noise * r.nextGaussian()};
        }
        double[][] grid = box.grid(21, 21);
        double[][] gt = new double[grid.length][];
        for (int i = 0; i < grid.length; i++) {
            double v = rho.applyAsDouble(grid[i][0], grid[i][1]);
            gt[i] = new double[] {v, c * v};
        }
        List<ConstraintRow> rows = new ArrayList<>();
        rows.add(fitField("conservation law (hard)", ConstrainedFields.conservationLaw(
                PhysicsInformedNetwork.tanhMlp(2, 32, 2, 1, seed)), pts, tg, 0, true, grid, gt, box, steps, seed));
        rows.add(fitField("unconstrained MLP", PhysicsInformedNetwork.tanhMlp(2, 32, 2, 2, seed), pts, tg, 0, true,
                grid, gt, box, steps, seed));
        rows.add(fitField("soft penalty 1.0", PhysicsInformedNetwork.tanhMlp(2, 32, 2, 2, seed), pts, tg, 1, true,
                grid, gt, box, steps, seed));
        return rows;
    }

    // ---------------------------------------------------------------- 9.5.14 curriculum, marching, periodic

    private static boolean isPeriodic(Pde pde) {
        return pde.problem().conditions().stream().anyMatch(t -> t.name().equals("periodic"));
    }

    /** The boundary terms of one time window: periodic pairs or zero Dirichlet ends on the space axis. */
    private static List<PinnTraining.Term> windowBoundaries(CollocationSampler box, boolean periodic, int n,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] left = box.face(0, false, n, r);
        List<PinnTraining.Term> terms = new ArrayList<>();
        if (periodic) {
            double[][] right = new double[n][];
            for (int i = 0; i < n; i++) {
                right[i] = new double[] {box.high(0), left[i][1]};
            }
            terms.add(PinnTraining.periodic("periodic", left, right, 0));
        } else {
            terms.add(PinnTraining.dirichlet("left", left, new double[n]));
            terms.add(PinnTraining.dirichlet("right", box.face(0, true, n, r), new double[n]));
        }
        return terms;
    }

    private static Layer exactlyPeriodic(Pde pde, int width, int depth, long seed) {
        CollocationSampler box = pde.problem().domain();
        return PinnTraining.periodicField(PhysicsInformedNetwork.tanhMlp(pde.inputs() + 1, width, depth, 1, seed), 0,
                box.low(0), box.high(0) - box.low(0));
    }

    /**
     * 9.5.14: the problem-side fixes for hard PDEs against the plain PINN and the coarse solver:
     * a coefficient curriculum (when {@code family} is given, for example advection speeds
     * {@code 1 -> beta}), sequence-to-sequence time marching, the exact periodic embedding
     * (periodic problems only, with the periodic loss term dropped), marching plus the
     * embedding, and marching plus the embedding plus causal weights.
     *
     * @param pde the problem
     * @param family the same PDE at another coefficient, or {@code null} to skip the curriculum
     * @param stages the curriculum coefficients, ending at the target
     * @param width the MLP width
     * @param depth the MLP hidden layers
     * @param options the recipe (whole runs and each curriculum stage or time window)
     * @param windows time windows for marching
     * @param causalEpsilon the causal strength for the last row, 0 to skip it
     * @param seed the seed
     * @return one row per method, the coarse solver last
     */
    public static List<Row> hardFixes(Pde pde, java.util.function.DoubleFunction<Pde> family, double[] stages,
            int width, int depth, Supplier<PinnTraining.Options> options, int windows, double causalEpsilon,
            long seed) {
        List<Row> rows = new ArrayList<>();
        boolean periodic = isPeriodic(pde);
        rows.add(pinn("plain", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed), options.get(),
                seed));
        if (family != null) {
            List<PinnTraining.Problem> problems = new ArrayList<>();
            for (double c : stages) {
                problems.add(family.apply(c).problem());
            }
            Layer field = PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed);
            PinnTraining.Result r = PinnTraining.curriculum(PhysicsInformedNetwork.of(field), problems, options.get(),
                    seed);
            rows.add(new Row("curriculum", error(pde, field), r.millis()));
        }
        java.util.function.ToDoubleFunction<double[]> u0 = x -> pde.reference().applyAsDouble(x[0], 0);
        int bn = 50;
        PinnTraining.Marched m = PinnTraining.timeMarching(
                () -> PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed), pde.problem(), u0,
                box -> windowBoundaries(box, periodic, bn, seed), windows, 100, options.get(), seed);
        rows.add(new Row("time marching", error(pde, m.field()), m.millis()));
        if (periodic) {
            Layer field = exactlyPeriodic(pde, width, depth, seed);
            PinnTraining.Result r = PinnTraining.train(PhysicsInformedNetwork.of(field),
                    PinnTraining.withoutTerm(pde.problem(), "periodic"), options.get(), seed);
            rows.add(new Row("periodic embedding", error(pde, field), r.millis()));
            PinnTraining.Marched mp = PinnTraining.timeMarching(() -> exactlyPeriodic(pde, width, depth, seed),
                    pde.problem(), u0, box -> List.of(), windows, 100, options.get(), seed);
            rows.add(new Row("marching + periodic", error(pde, mp.field()), mp.millis()));
            if (causalEpsilon > 0) {
                PinnTraining.Marched mc = PinnTraining.timeMarching(() -> exactlyPeriodic(pde, width, depth, seed),
                        pde.problem(), u0, box -> List.of(), windows, 100,
                        options.get().causalWeights(causalEpsilon, 8), seed);
                rows.add(new Row("marching + periodic + causal", error(pde, mc.field()), mc.millis()));
            }
        }
        rows.add(coarseSolver(pde));
        return rows;
    }

    // ---------------------------------------------------------------- 9.5.15 second-order optimisers

    /**
     * 9.5.15: second-order optimisers on one PDE, all from the same initial network: Adam then
     * L-BFGS (the {@link PinnTraining} default), the same followed by Nyström Newton-CG (the
     * Rathore et al. recipe), energy natural gradient from scratch, and Levenberg-Marquardt from
     * scratch on the same residuals.
     *
     * @param pde the problem
     * @param width the MLP width (keep the network under about 2,000 parameters)
     * @param depth the MLP hidden layers
     * @param collocation collocation points
     * @param adamSteps Adam steps of the first-order recipe
     * @param lbfgs L-BFGS iterations of the first-order recipe
     * @param nncg Newton-CG iterations after it
     * @param engd ENGD and Levenberg-Marquardt iterations
     * @param seed the seed
     * @return one row per optimiser
     */
    public static List<Row> optimisers(Pde pde, int width, int depth, int collocation, int adamSteps, int lbfgs,
            int nncg, int engd, long seed) {
        List<Row> rows = new ArrayList<>();
        PinnTraining.Options first = new PinnTraining.Options().adam(adamSteps, 5e-3f).lbfgs(lbfgs)
                .collocation(collocation);
        rows.add(pinn("Adam + L-BFGS", pde, PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed), first,
                seed));
        Layer f = PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed);
        PhysicsInformedNetwork net = PhysicsInformedNetwork.of(f);
        PinnTraining.Result warm = PinnTraining.train(net, pde.problem(), first, seed);
        PinnNaturalGradient.Result newton = NystromNewtonCg.of(20, 1e-6, 20, 5).train(net, pde.problem(), collocation,
                nncg, null, seed);
        rows.add(new Row("Adam + L-BFGS + NNCG", error(pde, f), warm.millis() + newton.millis()));
        Layer g = PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed);
        PinnNaturalGradient.Result e = PinnNaturalGradient.engd(1e-8).train(PhysicsInformedNetwork.of(g),
                pde.problem(), collocation, engd, null, seed);
        rows.add(new Row("ENGD", error(pde, g), e.millis()));
        Layer h = PhysicsInformedNetwork.tanhMlp(pde.inputs(), width, depth, 1, seed);
        PinnNaturalGradient.Result lm = PinnNaturalGradient.levenbergMarquardt(1e-2).train(
                PhysicsInformedNetwork.of(h), pde.problem(), collocation, engd, null, seed);
        rows.add(new Row("Levenberg-Marquardt", error(pde, h), lm.millis()));
        return rows;
    }
}
