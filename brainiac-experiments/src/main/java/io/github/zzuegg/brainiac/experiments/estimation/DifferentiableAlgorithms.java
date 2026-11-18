package io.github.zzuegg.brainiac.experiments.estimation;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DecisionInstances;
import io.github.zzuegg.brainiac.experimental.diffopt.DecisionFocused;
import io.github.zzuegg.brainiac.experimental.diffopt.PerturbedOptimizers;
import io.github.zzuegg.brainiac.experimental.diffopt.QpLayer;
import io.github.zzuegg.brainiac.experimental.diffopt.SoftSorting;
import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;
import io.github.zzuegg.brainiac.experimental.nn.SatNetLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Differentiable algorithms (12.5.2-12.5.5): learning through sorting, combinatorial solvers,
 * decision problems and quadratic programs.
 * <ul>
 *   <li>{@link #sorting}: sets of two-digit number images ranked by a small MLP scorer supervised
 *       only by the true ordering, with every {@link SoftSorting} relaxation, against pairwise
 *       (RankNet) and listwise (ListNet) ranking losses and a regression oracle that sees the
 *       numbers;</li>
 *   <li>{@link #warcraftPaths}: Warcraft-lite shortest paths: a per-cell network predicts terrain
 *       costs and is trained from the optimal path alone by the perturbed Fenchel-Young loss or
 *       blackbox differentiation, against a network predicting the path cells directly and a
 *       cost-regression oracle;</li>
 *   <li>{@link #decisions}: predict-then-optimise on grid shortest paths with polynomial costs of
 *       increasing misspecification, plus knapsack and portfolio rows, by normalised regret;</li>
 *   <li>{@link #polytopeProjection} and {@link #sudoku}: OptNet layers learning an unknown polytope
 *       and the rules of 4x4 Sudoku, against an MLP and SATNet.</li>
 * </ul>
 */
public final class DifferentiableAlgorithms {

    private DifferentiableAlgorithms() {
    }

    /**
     * One sorting row.
     *
     * @param exact the share of test sets ranked exactly right
     * @param elementwise the share of positions ranked right
     * @param msPerStep training milliseconds per step
     */
    public record SortRow(double exact, double elementwise, double msPerStep) {
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

    /**
     * Ranking supervision through relaxed sorting, with a 50-32-1 MLP scorer.
     *
     * @param n items per set, for example 5 or 9
     * @param steps Adam steps
     * @param seed the seed
     * @return method to its row
     */
    public static Map<String, SortRow> sorting(int n, int steps, long seed) {
        DecisionInstances.SortingSets train = DecisionInstances.sortingSets(400, n, 0.2, seed);
        DecisionInstances.SortingSets test = DecisionInstances.sortingSets(200, n, 0.2, seed + 1);
        List<String> methods = new ArrayList<>();
        for (SoftSorting.Method m : SoftSorting.Method.values()) {
            methods.add(m.name());
        }
        methods.addAll(List.of("FAST_SOFT_RANK", "RANKNET_PAIRWISE", "LISTNET", "REGRESSION_ORACLE"));
        Map<String, SortRow> out = new LinkedHashMap<>();
        int batch = 16;
        for (String method : methods) {
            SplittableRandom r = new SplittableRandom(seed);
            Var w0 = Var.param(Tensor.normal(Shape.of(50, 32), 0, 0.2f, r));
            Var b0 = Var.param(Tensor.zeros(Shape.of(1, 32)));
            Var w = Var.param(Tensor.normal(Shape.of(32, 1), 0, 0.2f, r));
            Var bias = Var.param(Tensor.zeros(Shape.of(1, 1)));
            List<Var> ps = List.of(w0, b0, w, bias);
            Optimizer opt = Optimizer.adam(0.01f);
            long t0 = System.nanoTime();
            for (int s = 0; s < steps; s++) {
                double[][] x = new double[batch * n][];
                float[] perm = new float[batch * n * n];
                float[] ranks = new float[batch * n];
                float[] values = new float[batch * n];
                float[] sign = new float[batch * n * n];
                for (int b = 0; b < batch; b++) {
                    int set = r.nextInt(train.images().length);
                    double[] v = new double[n];
                    for (int i = 0; i < n; i++) {
                        x[b * n + i] = train.images()[set][i];
                        v[i] = train.values()[set][i];
                        values[b * n + i] = (float) (v[i] / 10);
                    }
                    int[] order = SoftSorting.argsortDescending(v);
                    for (int i = 0; i < n; i++) {
                        perm[b * n * n + i * n + order[i]] = 1;
                        ranks[b * n + order[i]] = n - i;
                        for (int j = 0; j < n; j++) {
                            sign[b * n * n + i * n + j] = (float) Math.signum(v[i] - v[j]);
                        }
                    }
                }
                Var scores = Var.of(rows(x)).matmul(w0).add(b0).relu().matmul(w).add(bias).reshape(batch, n);
                Var loss = switch (method) {
                    case "FAST_SOFT_RANK" -> SoftSorting.fastSoftRank(scores, 0.3)
                            .sub(Var.of(Tensor.of(Shape.of(batch, n), ranks))).square().mean();
                    case "RANKNET_PAIRWISE" -> {
                        Var diff = scores.reshape(batch, n, 1).sub(scores.reshape(batch, 1, n));
                        yield diff.mul(Tensor.of(Shape.of(batch, n, n), sign)).neg().softplus()
                                .mul(Tensor.of(Shape.of(batch, n, n), sign).abs()).mean();
                    }
                    case "LISTNET" -> {
                        Var target = Var.of(Tensor.of(Shape.of(batch, n), ranks)).softmax(1);
                        yield target.mul(scores.logSoftmax(1)).sum().neg().div(batch);
                    }
                    case "REGRESSION_ORACLE" -> scores.sub(Var.of(Tensor.of(Shape.of(batch, n), values))).square()
                            .mean();
                    default -> {
                        Var p = SoftSorting.permutation(scores, SoftSorting.Method.valueOf(method), 1.0);
                        yield p.add(1e-9f).log().mul(Tensor.of(Shape.of(batch, n, n), perm)).sum().neg()
                                .div(batch * n);
                    }
                };
                loss.backward();
                opt.step(ps);
            }
            double ms = (System.nanoTime() - t0) / 1e6 / steps;
            int exact = 0;
            int right = 0;
            for (int set = 0; set < test.images().length; set++) {
                double[] v = new double[n];
                for (int i = 0; i < n; i++) {
                    v[i] = test.values()[set][i];
                }
                Tensor xs = rows(test.images()[set]);
                double[] sc = Var.noGrad(() -> Var.of(xs).matmul(w0).add(b0).relu().matmul(w).add(bias)).value()
                        .toDoubleArray();
                int[] po = SoftSorting.argsortDescending(sc);
                int[] to = SoftSorting.argsortDescending(v);
                int ok = 0;
                for (int i = 0; i < n; i++) {
                    ok += po[i] == to[i] ? 1 : 0;
                }
                right += ok;
                exact += ok == n ? 1 : 0;
            }
            out.put(method, new SortRow(exact / (double) test.images().length,
                    right / (double) (test.images().length * n), ms));
        }
        return out;
    }

    /**
     * One shortest-path row.
     *
     * @param optimalRate the share of test maps whose predicted path is optimal
     * @param regret the mean normalised regret {@code (cost(path) - cost(best)) / cost(best)}
     */
    public record PathRow(double optimalRate, double regret) {
    }

    /**
     * Warcraft-lite shortest paths learned from the optimal path.
     *
     * @param side cells per side, for example 8
     * @param steps Adam steps
     * @param seed the seed
     * @return method to its row
     */
    public static Map<String, PathRow> warcraftPaths(int side, int steps, long seed) {
        DecisionInstances.TerrainMaps train = DecisionInstances.terrainMaps(120, side, 0.3, seed);
        DecisionInstances.TerrainMaps test = DecisionInstances.terrainMaps(100, side, 0.3, seed + 1);
        int cells = side * side;
        double[][] trainPaths = new double[train.costs().length][];
        for (int m = 0; m < trainPaths.length; m++) {
            trainPaths[m] = PerturbedOptimizers.shortestPathCells(train.costs()[m], side, true);
        }
        PerturbedOptimizers.Solver solver = PerturbedOptimizers.gridShortestPath(side, true, 1e-3);
        Map<String, PathRow> out = new LinkedHashMap<>();
        int batch = 8;
        for (String method : List.of("PERTURBED_FY", "BLACKBOX", "DIRECT_PATH", "COST_REGRESSION_ORACLE")) {
            SplittableRandom r = new SplittableRandom(seed);
            boolean direct = method.equals("DIRECT_PATH");
            int hidden = 16;
            Var w1 = Var.param(Tensor.normal(Shape.of(9, hidden), 0, 0.3f, r));
            Var b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
            Var w2 = Var.param(Tensor.normal(Shape.of(hidden, direct ? 2 : 1), 0, 0.3f, r));
            Var b2 = Var.param(Tensor.zeros(Shape.of(1, direct ? 2 : 1)));
            List<Var> ps = new ArrayList<>(List.of(w1, b1, w2, b2));
            Var w3 = null;
            Var b3 = null;
            if (direct) {
                w3 = Var.param(Tensor.normal(Shape.of(2 * cells, cells), 0, 0.05f, r));
                b3 = Var.param(Tensor.zeros(Shape.of(1, cells)));
                ps.add(w3);
                ps.add(b3);
            }
            Optimizer opt = Optimizer.adam(0.01f);
            for (int s = 0; s < steps; s++) {
                int[] pick = new int[batch];
                double[][] x = new double[batch * cells][];
                float[] path = new float[batch * cells];
                float[] cost = new float[batch * cells];
                for (int b = 0; b < batch; b++) {
                    pick[b] = r.nextInt(train.costs().length);
                    for (int c = 0; c < cells; c++) {
                        x[b * cells + c] = train.patches()[pick[b]][c];
                        path[b * cells + c] = (float) trainPaths[pick[b]][c];
                        cost[b * cells + c] = (float) train.costs()[pick[b]][c];
                    }
                }
                Var hcell = Var.of(rows(x)).matmul(w1).add(b1).tanh().matmul(w2).add(b2);
                Tensor pathT = Tensor.of(Shape.of(batch, cells), path);
                Var loss;
                if (direct) {
                    Var logits = hcell.reshape(batch, 2 * cells).matmul(w3).add(b3);
                    Var p = logits.sigmoid().clamp(1e-5f, 1 - 1e-5f);
                    loss = p.log().mul(pathT).add(p.neg().add(1f).log().mul(pathT.neg().add(1f))).mean().neg();
                } else {
                    Var c = hcell.reshape(batch, cells).softplus().add(0.05f);
                    loss = switch (method) {
                        case "PERTURBED_FY" -> PerturbedOptimizers.fenchelYoungLoss(c.neg(), pathT, solver, 1.0, 8,
                                PerturbedOptimizers.Noise.GAUSSIAN, r);
                        case "BLACKBOX" -> {
                            Var y = PerturbedOptimizers.blackbox(c.neg(), solver, 10);
                            // Hamming distance to the true path
                            yield y.mul(pathT.neg().add(1f)).add(y.neg().add(1f).mul(pathT)).sum().div(batch);
                        }
                        default -> c.sub(Var.of(Tensor.of(Shape.of(batch, cells), cost))).square().mean();
                    };
                }
                loss.backward();
                opt.step(ps);
            }
            int optimal = 0;
            double regret = 0;
            for (int m = 0; m < test.costs().length; m++) {
                Tensor xs = rows(test.patches()[m]);
                Var hcell = Var.noGrad(() -> Var.of(xs).matmul(w1).add(b1).tanh().matmul(w2).add(b2));
                double[] pred;
                if (direct) {
                    Var w3f = w3;
                    Var b3f = b3;
                    double[] p = Var.noGrad(() -> hcell.reshape(1, 2 * cells).matmul(w3f).add(b3f).sigmoid())
                            .value().toDoubleArray();
                    pred = new double[cells];
                    for (int c = 0; c < cells; c++) {
                        pred[c] = 1 - p[c] + 1e-3;
                    }
                } else {
                    double[] hv = hcell.value().toDoubleArray();
                    pred = new double[cells];
                    for (int c = 0; c < cells; c++) {
                        pred[c] = Math.log1p(Math.exp(hv[c])) + 0.05;
                    }
                }
                double[] y = PerturbedOptimizers.shortestPathCells(pred, side, true);
                double[] best = PerturbedOptimizers.shortestPathCells(test.costs()[m], side, true);
                double cy = 0;
                double cb = 0;
                for (int c = 0; c < cells; c++) {
                    cy += y[c] * test.costs()[m][c];
                    cb += best[c] * test.costs()[m][c];
                }
                optimal += cy - cb < 1e-9 ? 1 : 0;
                regret += (cy - cb) / cb / test.costs().length;
            }
            out.put(method, new PathRow(optimal / (double) test.costs().length, regret));
        }
        return out;
    }

    /**
     * Predict-then-optimise with a linear cost model: normalised test regret per loss.
     *
     * @param degrees the polynomial degrees of the true cost (misspecification)
     * @param trainSize training instances, for example 200
     * @param epochs full-batch Adam steps
     * @param seed the seed
     * @return "problem d=k / loss" to normalised regret
     */
    public static Map<String, Double> decisions(int[] degrees, int trainSize, int epochs, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        int grid = 5;
        DecisionFocused.Problem paths = DecisionFocused.gridPaths(grid);
        for (int deg : degrees) {
            DecisionInstances.PathInstances tr = DecisionInstances.shortestPaths(trainSize, grid, 5, deg, 0.5, seed,
                    seed + 10);
            DecisionInstances.PathInstances te = DecisionInstances.shortestPaths(400, grid, 5, deg, 0.5, seed,
                    seed + 11);
            for (DecisionFocused.Loss loss : DecisionFocused.Loss.values()) {
                double[][] pred = fitLinear(tr.features(), tr.costs(), loss, paths, 0.25, epochs, seed, te.features());
                out.put("path d=" + deg + " / " + loss, DecisionFocused.normalisedRegret(paths, pred, te.costs()));
            }
            out.put("path d=" + deg + " / TWO_STAGE_FOREST", DecisionFocused.normalisedRegret(paths,
                    forest(tr.features(), tr.costs(), te.features(), seed), te.costs()));
        }
        // knapsack and portfolio at degree 4 (costs are negated values / returns)
        int deg = 4;
        DecisionInstances.Knapsacks kt = DecisionInstances.knapsacks(Math.min(trainSize, 120), 10, 2, 5, deg, seed,
                seed + 20);
        DecisionInstances.Knapsacks ke = DecisionInstances.knapsacks(100, 10, 2, 5, deg, seed, seed + 21);
        DecisionFocused.Problem knap = DecisionFocused.knapsack(kt.weights(), kt.capacity());
        double[][] kxTr = itemFeatures(kt.features());
        double[][] kxTe = itemFeatures(ke.features());
        double[][] kcTr = negate(kt.values());
        double[][] kcTe = negate(ke.values());
        for (DecisionFocused.Loss loss : new DecisionFocused.Loss[] {DecisionFocused.Loss.TWO_STAGE_MSE,
            DecisionFocused.Loss.SPO_PLUS, DecisionFocused.Loss.PG_BACKWARD}) {
            double[][] pred = fitLinear(kxTr, kcTr, loss, knap, 1.0, epochs / 2, seed, kxTe);
            out.put("knapsack d=4 / " + loss, DecisionFocused.normalisedRegret(knap, pred, kcTe));
        }
        DecisionInstances.Portfolios pt = DecisionInstances.portfolios(trainSize, 12, 5, deg, seed, seed + 30);
        DecisionInstances.Portfolios pe = DecisionInstances.portfolios(200, 12, 5, deg, seed, seed + 31);
        DecisionFocused.Problem port = DecisionFocused.portfolio(12, pt.cap());
        for (DecisionFocused.Loss loss : new DecisionFocused.Loss[] {DecisionFocused.Loss.TWO_STAGE_MSE,
            DecisionFocused.Loss.SPO_PLUS, DecisionFocused.Loss.PG_BACKWARD}) {
            double[][] pred = fitLinear(pt.features(), negate(pt.returns()), loss, port, 0.05, epochs / 2, seed,
                    pe.features());
            out.put("portfolio d=4 / " + loss, DecisionFocused.normalisedRegret(port, pred, negate(pe.returns())));
        }
        return out;
    }

    /** Flattens per-item features into one row per instance. */
    private static double[][] itemFeatures(double[][][] f) {
        int items = f[0].length;
        int p = f[0][0].length;
        double[][] x = new double[f.length][items * p];
        for (int m = 0; m < f.length; m++) {
            for (int i = 0; i < items; i++) {
                System.arraycopy(f[m][i], 0, x[m], i * p, p);
            }
        }
        return x;
    }

    private static double[][] negate(double[][] v) {
        double[][] out = new double[v.length][v[0].length];
        for (int i = 0; i < v.length; i++) {
            for (int j = 0; j < v[0].length; j++) {
                out[i][j] = -v[i][j];
            }
        }
        return out;
    }

    /** A linear cost model {@code c^ = W x + b} trained full-batch with one decision loss. */
    private static double[][] fitLinear(double[][] x, double[][] c, DecisionFocused.Loss loss,
            DecisionFocused.Problem problem, double h, int epochs, long seed, double[][] testX) {
        int p = x[0].length;
        int d = c[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        Var w = Var.param(Tensor.zeros(Shape.of(p, d)));
        double mean = 0;
        for (double[] row : c) {
            for (double v : row) {
                mean += v / (c.length * d);
            }
        }
        Var b = Var.param(Tensor.full(Shape.of(1, d), (float) mean));
        List<Var> ps = List.of(w, b);
        Optimizer opt = Optimizer.adam(0.05f);
        Tensor xs = rows(x);
        for (int e = 0; e < epochs; e++) {
            Var pred = Var.of(xs).matmul(w).add(b);
            DecisionFocused.loss(loss, pred, c, problem, h, r).backward();
            opt.step(ps);
        }
        double[] pv = Var.noGrad(() -> Var.of(rows(testX)).matmul(w).add(b)).value().toDoubleArray();
        double[][] out = new double[testX.length][d];
        for (int i = 0; i < testX.length; i++) {
            System.arraycopy(pv, i * d, out[i], 0, d);
        }
        return out;
    }

    /** Two-stage random forest: one small forest per cost coordinate. */
    private static double[][] forest(double[][] x, double[][] c, double[][] testX, long seed) {
        int d = c[0].length;
        double[][] out = new double[testX.length][d];
        RegressionForest.Settings s = RegressionForest.Settings.defaults().withTrees(10).withMinLeaf(5).withSeed(seed);
        for (int e = 0; e < d; e++) {
            double[] y = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                y[i] = c[i][e];
            }
            double[] pred = RegressionForest.fit(x, y, null, s).predict(testX);
            for (int i = 0; i < testX.length; i++) {
                out[i][e] = pred[i];
            }
        }
        return out;
    }

    /**
     * Learning an unknown polytope: the targets are exact Euclidean projections of Gaussian points
     * onto {@code {x : G* x <= 1}} with six random half-spaces in four dimensions.
     *
     * @param steps Adam steps
     * @param seed the seed
     * @return "OptNet" and "MLP" to {test MSE, share of outputs outside the true polytope},
     *     "exact projection" to {0, 0}
     */
    public static Map<String, double[]> polytopeProjection(int steps, long seed) {
        int n = 4;
        int m = 6;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] gTrue = new double[m][n];
        double[] hTrue = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                gTrue[i][j] = r.nextGaussian();
            }
            hTrue[i] = 1;
        }
        double[][] eye = new double[n][n];
        for (int i = 0; i < n; i++) {
            eye[i][i] = 1;
        }
        int trainN = 256;
        int testN = 200;
        double[][] x = new double[trainN + testN][n];
        double[][] y = new double[trainN + testN][];
        for (int i = 0; i < x.length; i++) {
            double[] q = new double[n];
            for (int j = 0; j < n; j++) {
                x[i][j] = r.nextGaussian() * 1.5;
                q[j] = -x[i][j];
            }
            y[i] = QpLayer.interiorPoint(eye, q, gTrue, hTrue, new double[0][n], new double[0], 40)[0];
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        QpLayer qp = QpLayer.of(n, m, 0, false, 1.0, false, r);
        int hidden = 6;
        Var w1 = Var.param(Tensor.normal(Shape.of(n, hidden), 0, 0.5f, r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        Var w2 = Var.param(Tensor.normal(Shape.of(hidden, n), 0, 0.5f, r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1, n)));
        for (String model : List.of("OptNet", "MLP")) {
            List<Var> ps = model.equals("OptNet") ? qp.parameters() : List.of(w1, b1, w2, b2);
            Optimizer opt = Optimizer.adam(0.02f);
            int batch = 32;
            for (int s = 0; s < steps; s++) {
                double[][] xb = new double[batch][];
                double[][] yb = new double[batch][];
                for (int b = 0; b < batch; b++) {
                    int i = r.nextInt(trainN);
                    xb[b] = x[i];
                    yb[b] = y[i];
                }
                Var pred = model.equals("OptNet") ? qp.forward(Var.of(rows(xb)).neg())
                        : Var.of(rows(xb)).matmul(w1).add(b1).tanh().matmul(w2).add(b2).add(Var.of(rows(xb)));
                pred.sub(Var.of(rows(yb))).square().mean().backward();
                opt.step(ps);
            }
            double[][] xt = java.util.Arrays.copyOfRange(x, trainN, x.length);
            double[][] yt = java.util.Arrays.copyOfRange(y, trainN, y.length);
            Var pred = Var.noGrad(() -> model.equals("OptNet") ? qp.forward(Var.of(rows(xt)).neg())
                    : Var.of(rows(xt)).matmul(w1).add(b1).tanh().matmul(w2).add(b2).add(Var.of(rows(xt))));
            double[] pv = pred.value().toDoubleArray();
            double mse = 0;
            int outside = 0;
            for (int i = 0; i < testN; i++) {
                boolean viol = false;
                for (int j = 0; j < n; j++) {
                    mse += Math.pow(pv[i * n + j] - yt[i][j], 2) / (testN * n);
                }
                for (int c = 0; c < m; c++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) {
                        s += gTrue[c][j] * pv[i * n + j];
                    }
                    viol |= s > hTrue[c] + 0.05;
                }
                outside += viol ? 1 : 0;
            }
            out.put(model, new double[] {mse, outside / (double) testN});
        }
        out.put("exact projection", new double[] {0, 0});
        return out;
    }

    /**
     * 4x4 Sudoku: OptNet ({@code min 0.05 |z|^2 - p . z} with {@code z >= 0} and learned
     * {@code A z = b}) against SATNet and an MLP of similar size, all trained with binary
     * cross-entropy on the 64 one-hot cell bits.
     *
     * @param trainSize training puzzles
     * @param steps Adam steps
     * @param blanks blank cells per puzzle
     * @param seed the seed
     * @return model to {blank-cell accuracy, solved-puzzle rate, training seconds}
     */
    public static Map<String, double[]> sudoku(int trainSize, int steps, int blanks, long seed) {
        DecisionInstances.Sudokus train = DecisionInstances.sudokus(trainSize, blanks, seed);
        DecisionInstances.Sudokus test = DecisionInstances.sudokus(100, blanks, seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        QpLayer qp = QpLayer.of(64, 0, 40, false, 0.1, true, r);
        SatNetLayer sat = new SatNetLayer(64, 96, 16, new SplittableRandom(seed)).sweeps(10);
        Var w1 = Var.param(Tensor.normal(Shape.of(64, 48), 0, 0.15f, r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1, 48)));
        Var w2 = Var.param(Tensor.normal(Shape.of(48, 64), 0, 0.15f, r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1, 64)));
        for (String model : List.of("OptNet", "SATNet", "MLP")) {
            List<Var> ps = switch (model) {
                case "OptNet" -> qp.parameters();
                case "SATNet" -> sat.parameters();
                default -> List.of(w1, b1, w2, b2);
            };
            java.util.function.Function<int[][], Var> forward = puzzles -> {
                int b = puzzles.length;
                float[] onehot = new float[b * 64];
                float[] given = new float[b * 64];
                for (int i = 0; i < b; i++) {
                    for (int c = 0; c < 16; c++) {
                        if (puzzles[i][c] > 0) {
                            onehot[i * 64 + c * 4 + puzzles[i][c] - 1] = 1;
                            for (int v = 0; v < 4; v++) {
                                given[i * 64 + c * 4 + v] = 1;
                            }
                        }
                    }
                }
                Tensor p = Tensor.of(Shape.of(b, 64), onehot);
                return switch (model) {
                    case "OptNet" -> qp.forward(Var.of(p).neg()).clamp(1e-4f, 1 - 1e-4f);
                    case "SATNet" -> sat.forward(Var.of(Tensor.of(Shape.of(b, 128), concat(onehot, given, b))))
                            .clamp(1e-4f, 1 - 1e-4f);
                    default -> Var.of(p).matmul(w1).add(b1).relu().matmul(w2).add(b2).sigmoid().clamp(1e-4f, 1 - 1e-4f);
                };
            };
            Optimizer opt = Optimizer.adam(model.equals("SATNet") ? 0.05f : 0.02f);
            long t0 = System.nanoTime();
            int batch = 16;
            for (int s = 0; s < steps; s++) {
                int[][] pz = new int[batch][];
                float[] y = new float[batch * 64];
                for (int b = 0; b < batch; b++) {
                    int i = r.nextInt(trainSize);
                    pz[b] = train.puzzles()[i];
                    for (int c = 0; c < 16; c++) {
                        y[b * 64 + c * 4 + train.solutions()[i][c] - 1] = 1;
                    }
                }
                Var p = forward.apply(pz);
                Tensor yt = Tensor.of(Shape.of(batch, 64), y);
                p.log().mul(yt).add(p.neg().add(1f).log().mul(yt.neg().add(1f))).mean().neg().backward();
                opt.step(ps);
            }
            double secs = (System.nanoTime() - t0) / 1e9;
            double[] pv = Var.noGrad(() -> forward.apply(test.puzzles())).value().toDoubleArray();
            int right = 0;
            int total = 0;
            int solved = 0;
            for (int i = 0; i < test.puzzles().length; i++) {
                boolean all = true;
                for (int c = 0; c < 16; c++) {
                    if (test.puzzles()[i][c] > 0) {
                        continue;
                    }
                    int best = 0;
                    for (int v = 1; v < 4; v++) {
                        if (pv[i * 64 + c * 4 + v] > pv[i * 64 + c * 4 + best]) {
                            best = v;
                        }
                    }
                    boolean ok = best + 1 == test.solutions()[i][c];
                    right += ok ? 1 : 0;
                    total++;
                    all &= ok;
                }
                solved += all ? 1 : 0;
            }
            out.put(model, new double[] {right / (double) total, solved / (double) test.puzzles().length, secs});
        }
        return out;
    }

    private static float[] concat(float[] a, float[] b, int rows) {
        int da = a.length / rows;
        int db = b.length / rows;
        float[] out = new float[a.length + b.length];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(a, i * da, out, i * (da + db), da);
            System.arraycopy(b, i * db, out, i * (da + db) + da, db);
        }
        return out;
    }
}
