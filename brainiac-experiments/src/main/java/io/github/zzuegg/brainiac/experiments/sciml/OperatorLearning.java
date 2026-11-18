package io.github.zzuegg.brainiac.experiments.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierNeuralOperator1d;
import io.github.zzuegg.brainiac.experimental.nn.LinearAttention;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.sciml.DeepOnet;
import io.github.zzuegg.brainiac.experimental.sciml.MessagePassingPdeSolver;
import io.github.zzuegg.brainiac.experimental.sciml.PdeProblems;
import io.github.zzuegg.brainiac.experimental.sciml.PdeRefiner;
import io.github.zzuegg.brainiac.experimental.sciml.PhysicsAttention;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Operator learning (wave 8, items 8.3.4-8.3.7): neural operators against the
 * {@link FourierNeuralOperator1d} (used unchanged) on 1-D problems from {@link PdeProblems}.
 *
 * <ul>
 *   <li>{@link #deepOnet} (8.3.4): DeepONet, FNO and a grid-to-grid MLP on the antiderivative
 *       operator, Burgers (initial condition to the solution at {@code t = 0.5}) and 1-D Darcy
 *       (coefficient to solution), scored on the training grid and at points between grid
 *       points (the grid models interpolate their output there).
 *   <li>{@link #physicsAttention} (8.3.5): Transolver and LANO physics-attention, a plain
 *       linear-attention transformer, FNO and DeepONet on Burgers, on the regular grid and on
 *       fixed irregular sample points, plus forward time per sample against {@code N}.
 *   <li>{@link #rollouts} (8.3.6): message-passing PDE solvers with and without the pushforward
 *       trick and temporal bundling, an autoregressive FNO and the reference solver at half
 *       resolution on Burgers rollouts with a varying viscosity.
 *   <li>{@link #refiner} (8.3.7): PDE-Refiner against the same network as a one-step model,
 *       an autoregressive FNO and the message-passing solver on Kuramoto-Sivashinsky rollouts.
 * </ul>
 */
public final class OperatorLearning {

    private OperatorLearning() {
    }

    /**
     * Function pairs sampled on a fine periodic grid of {@code [0, 1)}.
     *
     * @param inputs input functions {@code [N][fine]}
     * @param outputs output functions {@code [N][fine]}
     */
    public record Data(double[][] inputs, double[][] outputs) {

        /** @return the pairs */
        public int size() {
            return inputs.length;
        }

        Data range(int from, int to) {
            return new Data(java.util.Arrays.copyOfRange(inputs, from, to),
                    java.util.Arrays.copyOfRange(outputs, from, to));
        }
    }

    /**
     * An operator-learning row.
     *
     * @param model the model
     * @param onGrid relative L2 test error on the training grid
     * @param offGrid relative L2 test error between grid points (or on irregular points)
     * @param millis training time
     */
    public record Row(String model, double onGrid, double offGrid, double millis) {
    }

    /**
     * The antiderivative operator {@code u -> int_0^x u}.
     *
     * @param count pairs
     * @param fine fine-grid points
     * @param seed the seed
     * @return the data
     */
    public static Data antiderivative(int count, int fine, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] in = new double[count][];
        double[][] out = new double[count][];
        for (int i = 0; i < count; i++) {
            in[i] = PdeProblems.randomField(fine, 6, 1.5, r);
            out[i] = PdeProblems.antiderivative(in[i]);
        }
        return new Data(in, out);
    }

    /**
     * Burgers' equation {@code u_t + u u_x = 0.02 u_xx} on {@code [0, 1)}: initial condition to
     * the solution at {@code t = 0.5}.
     *
     * @param count pairs
     * @param fine fine-grid points
     * @param seed the seed
     * @return the data
     */
    public static Data burgers(int count, int fine, long seed) {
        double[][][] p = PdeProblems.pairs(PdeProblems.Equation.BURGERS, count, fine, 1, 0.02, 0.5, 4, seed);
        double[][] in = new double[count][];
        double[][] out = new double[count][];
        for (int i = 0; i < count; i++) {
            in[i] = p[i][0];
            out[i] = p[i][1];
        }
        return new Data(in, out);
    }

    /**
     * 1-D Darcy flow {@code -(a u')' = 1} on {@code [0, 1]} with zero ends: the log-normal
     * coefficient {@code a = exp(field)} to the solution.
     *
     * @param count pairs
     * @param fine fine-grid points
     * @param seed the seed
     * @return the data
     */
    public static Data darcy(int count, int fine, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] in = new double[count][];
        double[][] out = new double[count][];
        for (int i = 0; i < count; i++) {
            double[] field = PdeProblems.randomField(2 * fine, 5, 1.5, r);
            double[] a = new double[fine];
            for (int j = 0; j < fine; j++) {
                a[j] = Math.exp(field[2 * j]);
            }
            // interior solve with faces at the fine points' half steps
            double[] faces = new double[fine];
            for (int j = 0; j < fine; j++) {
                faces[j] = Math.exp(field[(2 * j + 1) % (2 * fine)]);
            }
            double[] f = new double[fine - 1];
            java.util.Arrays.fill(f, 1);
            double[] u = PdeProblems.darcy1d(faces, f);
            double[] full = new double[fine];
            for (int j = 1; j < fine; j++) {
                full[j] = u[j - 1] * 10;
            }
            in[i] = a;
            out[i] = full;
        }
        return new Data(in, out);
    }

    static Tensor grid(double[][] rows, int stride, int offset) {
        int n = rows.length;
        int m = rows[0].length / stride;
        float[] f = new float[n * m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                f[i * m + j] = (float) rows[i][j * stride + offset];
            }
        }
        return Tensor.of(Shape.of(n, m), f);
    }

    static double[] interpolatePeriodic(float[] values, int offset, int m, double position) {
        double s = position * m;
        int j = (int) Math.floor(s);
        double w = s - j;
        return new double[] {values[offset + Math.floorMod(j, m)] * (1 - w)
            + values[offset + Math.floorMod(j + 1, m)] * w};
    }

    static double relative(float[] pred, double[][] truth, int stride, int offset) {
        double num = 0;
        double den = 0;
        int n = truth.length;
        int m = truth[0].length / stride;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double t = truth[i][j * stride + offset];
                double d = pred[i * m + j] - t;
                num += d * d;
                den += t * t;
            }
        }
        return Math.sqrt(num / den);
    }

    /** A grid model: coarse input grid {@code [b, m]} to output on the same grid. */
    private interface GridModel {
        Var apply(Var u);

        List<Var> parameters();
    }

    private static double train(GridModel model, Tensor x, Tensor y, int epochs, int batch, float lr, long seed) {
        long start = System.nanoTime();
        Optimizer opt = Optimizer.adam(lr);
        int n = x.shape().dim(0);
        SplittableRandom r = new SplittableRandom(seed);
        for (int e = 0; e < epochs; e++) {
            for (int b = 0; b < n; b += batch) {
                int s = r.nextInt(Math.max(1, n - batch + 1));
                Tensor xb = x.slice(0, s, Math.min(n, s + batch));
                Tensor yb = y.slice(0, s, Math.min(n, s + batch));
                Var loss = model.apply(Var.of(xb)).sub(yb).square().mean();
                loss.backward();
                opt.step(model.parameters());
            }
        }
        return (System.nanoTime() - start) / 1e6;
    }

    private static GridModel fno(int m, long seed) {
        FourierNeuralOperator1d f = new FourierNeuralOperator1d(1, 1, 12, 8, 2, new SplittableRandom(seed));
        return new GridModel() {
            @Override
            public Var apply(Var u) {
                int b = u.shape().dim(0);
                return f.forward(u.reshape(b, m, 1)).reshape(b, m);
            }

            @Override
            public List<Var> parameters() {
                return f.parameters();
            }
        };
    }

    private static GridModel mlp(int m, long seed) {
        SimpleMlp net = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), m, 64, 64, m);
        return new GridModel() {
            @Override
            public Var apply(Var u) {
                return net.forward(u);
            }

            @Override
            public List<Var> parameters() {
                return net.parameters();
            }
        };
    }

    private static Tensor coordinates(int m, int stride, int offset, int fine) {
        float[] y = new float[m];
        for (int j = 0; j < m; j++) {
            y[j] = (float) ((j * stride + offset) / (double) fine);
        }
        return Tensor.of(Shape.of(m, 1), y);
    }

    /**
     * 8.3.4: DeepONet against FNO and a grid MLP on one dataset.
     *
     * @param data the pairs on a fine grid; the first {@code train} are for training
     * @param train training pairs
     * @param epochs training epochs
     * @param seed the seed
     * @return rows for DeepONet, FNO and MLP
     */
    public static List<Row> deepOnet(Data data, int train, int epochs, long seed) {
        int fine = data.inputs()[0].length;
        int m = fine / 2;
        Data tr = data.range(0, train);
        Data te = data.range(train, data.size());
        Tensor xTrain = grid(tr.inputs(), 2, 0);
        Tensor yTrain = grid(tr.outputs(), 2, 0);
        Tensor xTest = grid(te.inputs(), 2, 0);
        List<Row> rows = new ArrayList<>();
        // DeepONet: queried at the coarse grid in training, anywhere at test time
        DeepOnet onet = DeepOnet.of(m, 1, 64, 2, 32, seed);
        Tensor yOn = coordinates(m, 2, 0, fine);
        Tensor yOff = coordinates(m, 2, 1, fine);
        double ms = train(new GridModel() {
            @Override
            public Var apply(Var u) {
                return onet.forward(u, Var.of(yOn));
            }

            @Override
            public List<Var> parameters() {
                return onet.parameters();
            }
        }, xTrain, yTrain, 10 * epochs, 32, 3e-3f, seed);
        float[] on = Var.noGrad(() -> onet.forward(Var.of(xTest), Var.of(yOn))).value().toArray();
        float[] off = Var.noGrad(() -> onet.forward(Var.of(xTest), Var.of(yOff))).value().toArray();
        rows.add(new Row("DeepONet", relative(on, te.outputs(), 2, 0), relative(off, te.outputs(), 2, 1), ms));
        rows.add(gridRow("FNO", fno(m, seed), xTrain, yTrain, xTest, te, epochs, seed));
        rows.add(gridRow("MLP grid-to-grid", mlp(m, seed), xTrain, yTrain, xTest, te, epochs, seed));
        return rows;
    }

    private static Row gridRow(String name, GridModel model, Tensor xTrain, Tensor yTrain, Tensor xTest, Data te,
            int epochs, long seed) {
        double ms = train(model, xTrain, yTrain, epochs, 32, 2e-3f, seed);
        float[] on = Var.noGrad(() -> model.apply(Var.of(xTest))).value().toArray();
        int m = xTest.shape().dim(1);
        int n = te.size();
        float[] off = new float[n * m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                // midpoint between coarse points j and j + 1 (periodic)
                off[i * m + j] = (float) interpolatePeriodic(on, i * m, m, (j + 0.5) / m)[0];
            }
        }
        return new Row(name, relative(on, te.outputs(), 2, 0), relative(off, te.outputs(), 2, 1), ms);
    }

    // ---------------------------------------------------------------- 8.3.5 physics-attention

    /** A transformer over point features {@code (x, u(x))} with a pluggable attention block. */
    private static final class PointTransformer implements GridModel {
        private final Var lift;
        private final List<Layer> blocks;
        private final Var project;
        private final Var coords;

        PointTransformer(Function<Integer, Layer> block, int channels, int depth, double[] positions, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.lift = Var.param(Tensor.normal(Shape.of(3, channels), 0, 1f, r));
            List<Layer> b = new ArrayList<>();
            for (int i = 0; i < depth; i++) {
                b.add(block.apply(i));
            }
            this.blocks = b;
            this.project = Var.param(Tensor.normal(Shape.of(channels, 1), 0, (float) (1 / Math.sqrt(channels)), r));
            float[] c = new float[positions.length * 2];
            for (int j = 0; j < positions.length; j++) {
                c[2 * j] = (float) Math.sin(2 * Math.PI * positions[j]);
                c[2 * j + 1] = (float) Math.cos(2 * Math.PI * positions[j]);
            }
            this.coords = Var.of(Tensor.of(Shape.of(1, positions.length, 2), c));
        }

        @Override
        public Var apply(Var u) {
            int b = u.shape().dim(0);
            int n = u.shape().dim(1);
            Var x = Var.concat(2, coords.expand(Shape.of(b, n, 2)), u.reshape(b, n, 1)).matmul(lift);
            for (Layer l : blocks) {
                x = l.forward(x);
            }
            return x.matmul(project).reshape(b, n);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(lift, project));
            blocks.forEach(l -> p.addAll(l.parameters()));
            return p;
        }
    }

    private static Var norm(Var x) {
        Var c = x.sub(x.mean(2, true));
        return c.div(c.square().mean(2, true).add(1e-5f).sqrt());
    }

    /** The same pre-norm residual block as {@link PhysicsAttention}, with linear attention over all points. */
    private static Layer linearAttentionBlock(int channels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        LinearAttention att = new LinearAttention(channels, 2, r);
        Var ff1 = Var.param(Tensor.normal(Shape.of(channels, 2 * channels), 0, (float) Math.sqrt(1.0 / channels), r));
        Var ff1b = Var.param(Tensor.zeros(Shape.of(2 * channels)));
        Var ff2 = Var.param(Tensor.normal(Shape.of(2 * channels, channels), 0,
                (float) Math.sqrt(0.5 / channels), r));
        return new Layer() {
            @Override
            public Var forward(Var x) {
                Var h = x.add(att.forward(norm(x)));
                return h.add(io.github.zzuegg.brainiac.nn.Activation.GELU.forward(norm(h).matmul(ff1).add(ff1b))
                        .matmul(ff2));
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(att.parameters());
                p.addAll(List.of(ff1, ff1b, ff2));
                return p;
            }
        };
    }

    private static double[] positions(int m, boolean irregular, long seed) {
        double[] p = new double[m];
        SplittableRandom r = new SplittableRandom(seed);
        for (int j = 0; j < m; j++) {
            p[j] = irregular ? r.nextDouble() : j / (double) m;
        }
        if (irregular) {
            java.util.Arrays.sort(p);
        }
        return p;
    }

    private static Tensor sampleAt(double[][] rows, double[] positions) {
        int n = rows.length;
        int fine = rows[0].length;
        float[] f = new float[n * positions.length];
        float[] all = new float[n * fine];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < fine; j++) {
                all[i * fine + j] = (float) rows[i][j];
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < positions.length; j++) {
                f[i * positions.length + j] = (float) interpolatePeriodic(all, i * fine, fine, positions[j])[0];
            }
        }
        return Tensor.of(Shape.of(n, positions.length), f);
    }

    private static double relative(float[] pred, Tensor truth) {
        float[] t = truth.toArray();
        double num = 0;
        double den = 0;
        for (int i = 0; i < t.length; i++) {
            num += (pred[i] - t[i]) * (pred[i] - t[i]);
            den += t[i] * t[i];
        }
        return Math.sqrt(num / den);
    }

    /** Wraps a grid model so it reads irregular points by interpolating them to its grid first. */
    private static GridModel onGridViaInterpolation(GridModel inner, double[] positions, int m) {
        float[][] toGrid = new float[m][positions.length];
        float[][] fromGrid = new float[positions.length][m];
        for (int j = 0; j < m; j++) {
            double x = j / (double) m;
            int k = 0;
            while (k < positions.length && positions[k] < x) {
                k++;
            }
            int hi = k % positions.length;
            int lo = Math.floorMod(k - 1, positions.length);
            double xl = positions[lo] - (k == 0 ? 1 : 0);
            double xh = positions[hi] + (k == positions.length ? 1 : 0);
            double w = (x - xl) / (xh - xl);
            toGrid[j][lo] += (float) (1 - w);
            toGrid[j][hi] += (float) w;
        }
        for (int i = 0; i < positions.length; i++) {
            double s = positions[i] * m;
            int j = (int) Math.floor(s);
            double w = s - j;
            fromGrid[i][Math.floorMod(j, m)] += (float) (1 - w);
            fromGrid[i][Math.floorMod(j + 1, m)] += (float) w;
        }
        Tensor a = Tensor.of(toGrid).transpose();
        Tensor bm = Tensor.of(fromGrid).transpose();
        return new GridModel() {
            @Override
            public Var apply(Var u) {
                return inner.apply(u.matmul(Var.of(a))).matmul(Var.of(bm));
            }

            @Override
            public List<Var> parameters() {
                return inner.parameters();
            }
        };
    }

    /**
     * 8.3.5: physics-attention against FNO, DeepONet and linear attention on Burgers pairs, on
     * the regular grid ({@code onGrid}) and on fixed irregular points ({@code offGrid}).
     *
     * @param data Burgers pairs
     * @param train training pairs
     * @param points sample points per function
     * @param epochs training epochs
     * @param seed the seed
     * @return one row per model
     */
    public static List<Row> physicsAttention(Data data, int train, int points, int epochs, long seed) {
        Data tr = data.range(0, train);
        Data te = data.range(train, data.size());
        Map<String, double[]> results = new LinkedHashMap<>();
        Map<String, Double> times = new LinkedHashMap<>();
        for (int irregular = 0; irregular < 2; irregular++) {
            double[] pos = positions(points, irregular == 1, seed + 99);
            Tensor xTr = sampleAt(tr.inputs(), pos);
            Tensor yTr = sampleAt(tr.outputs(), pos);
            Tensor xTe = sampleAt(te.inputs(), pos);
            Tensor yTe = sampleAt(te.outputs(), pos);
            Map<String, GridModel> models = new LinkedHashMap<>();
            int c = 16;
            models.put("Transolver", new PointTransformer(i -> PhysicsAttention.of(PhysicsAttention.Mode.TRANSOLVER,
                    c, 8, seed + i), c, 1, pos, seed));
            models.put("LANO", new PointTransformer(i -> PhysicsAttention.of(PhysicsAttention.Mode.LANO, c, 8,
                    seed + i), c, 1, pos, seed));
            models.put("linear attention", new PointTransformer(i -> linearAttentionBlock(c, seed + i), c, 1, pos,
                    seed));
            GridModel f = fno(points, seed);
            models.put("FNO", irregular == 1 ? onGridViaInterpolation(f, pos, points) : f);
            DeepOnet onet = DeepOnet.of(points, 1, 64, 2, 32, seed);
            float[] q = new float[points];
            for (int j = 0; j < points; j++) {
                q[j] = (float) pos[j];
            }
            Tensor yq = Tensor.of(Shape.of(points, 1), q);
            models.put("DeepONet", new GridModel() {
                // trained ten times longer below: it is cheap
                @Override
                public Var apply(Var u) {
                    return onet.forward(u, Var.of(yq));
                }

                @Override
                public List<Var> parameters() {
                    return onet.parameters();
                }
            });
            for (Map.Entry<String, GridModel> e : models.entrySet()) {
                int ep = e.getKey().equals("DeepONet") ? 10 * epochs : epochs;
                double ms = train(e.getValue(), xTr, yTr, ep, 20, 2e-3f, seed);
                float[] pred = Var.noGrad(() -> e.getValue().apply(Var.of(xTe))).value().toArray();
                results.computeIfAbsent(e.getKey(), k -> new double[2])[irregular] = relative(pred, yTe);
                times.merge(e.getKey(), ms, Double::sum);
            }
        }
        List<Row> rows = new ArrayList<>();
        results.forEach((k, v) -> rows.add(new Row(k, v[0], v[1], times.get(k))));
        return rows;
    }

    /**
     * Forward time per sample of a physics-attention block against full softmax attention
     * cost, at several point counts (untrained weights; only cost matters).
     *
     * @param mode the attention variant
     * @param counts point counts
     * @return microseconds per sample for each count
     */
    public static double[] attentionTime(PhysicsAttention.Mode mode, int[] counts) {
        PhysicsAttention block = PhysicsAttention.of(mode, 32, 8, 1);
        double[] out = new double[counts.length];
        for (int i = 0; i < counts.length; i++) {
            Tensor x = Tensor.normal(Shape.of(4, counts[i], 32), 0, 1, new SplittableRandom(1));
            Var.noGrad(() -> block.forward(Var.of(x)));
            long start = System.nanoTime();
            for (int rep = 0; rep < 3; rep++) {
                Var.noGrad(() -> block.forward(Var.of(x)));
            }
            out[i] = (System.nanoTime() - start) / 1e3 / 12;
        }
        return out;
    }

    // ---------------------------------------------------------------- 8.3.6 rollouts

    /**
     * Burgers trajectories {@code u_t + u u_x = nu u_xx} on {@code [0, 2 pi)} with a random
     * viscosity per trajectory.
     *
     * @param count trajectories
     * @param points grid points
     * @param steps snapshots after the initial one
     * @param dt the snapshot interval
     * @param seed the seed
     * @return {@code {trajectories [count][steps + 1][points], viscosities [count][1]}}
     */
    public static double[][][][] burgersTrajectories(int count, int points, int steps, double dt, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] traj = new double[count][][];
        double[][][] nu = new double[count][1][1];
        for (int i = 0; i < count; i++) {
            double v = 0.01 + 0.03 * r.nextDouble();
            nu[i][0][0] = v;
            double[] u0 = PdeProblems.randomField(points, 3, 1, r);
            traj[i] = PdeProblems.solve(PdeProblems.Equation.BURGERS, u0, 0, 2 * Math.PI, v, steps * dt, steps + 1,
                    0.02).u();
        }
        return new double[][][][] {traj, nu};
    }

    /**
     * A rollout row.
     *
     * @param model the model
     * @param mse mean squared error over the rollout
     * @param survival mean steps until the correlation with the truth drops below 0.8
     * @param millis training time
     */
    public record RolloutRow(String model, double mse, double survival, double millis) {
    }

    static double correlation(double[] a, double[] b) {
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += a[i] / a.length;
            mb += b[i] / b.length;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return sab / Math.sqrt(saa * sbb + 1e-30);
    }

    private static RolloutRow score(String name, double[][][] predicted, double[][][] truth, int start, double ms) {
        double mse = 0;
        int count = 0;
        double survival = 0;
        for (int i = 0; i < predicted.length; i++) {
            int alive = predicted[i].length;
            for (int t = 0; t < predicted[i].length; t++) {
                double[] tr = truth[i][start + t];
                for (int j = 0; j < tr.length; j++) {
                    mse += (predicted[i][t][j] - tr[j]) * (predicted[i][t][j] - tr[j]);
                    count++;
                }
                if (alive == predicted[i].length && correlation(predicted[i][t], tr) < 0.8) {
                    alive = t;
                }
            }
            survival += alive;
        }
        return new RolloutRow(name, mse / count, survival / predicted.length, ms);
    }

    private static Tensor window(double[][][] traj, double[][][] nu, int[] ids, int[] starts, int length,
            boolean coef) {
        int b = ids.length;
        int n = traj[0][0].length;
        int c = length + (coef ? 1 : 0);
        float[] f = new float[b * n * c];
        for (int k = 0; k < b; k++) {
            for (int j = 0; j < n; j++) {
                for (int t = 0; t < length; t++) {
                    f[(k * n + j) * c + t] = (float) traj[ids[k]][starts[k] + t][j];
                }
                if (coef) {
                    f[(k * n + j) * c + length] = (float) (nu[ids[k]][0][0] * 10);
                }
            }
        }
        return Tensor.of(Shape.of(b, n, c), f);
    }

    /**
     * 8.3.6: message-passing PDE solvers on Burgers rollouts.
     *
     * @param trainTrajectories training trajectories
     * @param testTrajectories test trajectories
     * @param steps snapshots per trajectory
     * @param iterations training iterations per model
     * @param seed the seed
     * @return rows for MP-PDE (pushforward + bundling), without pushforward, without bundling,
     *     autoregressive FNO and the half-resolution solver
     */
    public static List<RolloutRow> rollouts(int trainTrajectories, int testTrajectories, int steps, int iterations,
            long seed) {
        int n = 32;
        double dt = 0.1;
        double[][][][] all = burgersTrajectories(trainTrajectories + testTrajectories, n, steps, dt, seed);
        double[][][] traj = all[0];
        double[][][] nu = all[1];
        int history = 2;
        List<RolloutRow> rows = new ArrayList<>();
        rows.add(mpRow("MP-PDE (pushforward, bundle 2)", traj, nu, trainTrajectories, history, 2, true, iterations,
                steps, seed));
        rows.add(mpRow("MP-PDE (no pushforward, bundle 2)", traj, nu, trainTrajectories, history, 2, false,
                iterations, steps, seed));
        rows.add(mpRow("MP-PDE (pushforward, no bundling)", traj, nu, trainTrajectories, history, 1, true, iterations,
                steps, seed));
        rows.add(fnoRollout(traj, nu, trainTrajectories, history, iterations, steps, seed));
        // the reference solver at half resolution
        long start = System.nanoTime();
        int test = traj.length - trainTrajectories;
        double[][][] pred = new double[test][][];
        for (int i = 0; i < test; i++) {
            int id = trainTrajectories + i;
            double[] coarse = new double[n / 2];
            for (int j = 0; j < n / 2; j++) {
                coarse[j] = traj[id][history - 1][2 * j];
            }
            double[][] u = PdeProblems.solve(PdeProblems.Equation.BURGERS, coarse, 0, 2 * Math.PI, nu[id][0][0],
                    (steps - history + 1) * dt, steps - history + 2, 0.02).u();
            pred[i] = new double[u.length - 1][n];
            for (int t = 1; t < u.length; t++) {
                for (int j = 0; j < n; j++) {
                    pred[i][t - 1][j] = j % 2 == 0 ? u[t][j / 2] : 0.5 * (u[t][j / 2] + u[t][(j / 2 + 1) % (n / 2)]);
                }
            }
        }
        rows.add(score("solver at half resolution", pred, testSlice(traj, trainTrajectories), history,
                (System.nanoTime() - start) / 1e6));
        return rows;
    }

    private static double[][][] testSlice(double[][][] traj, int from) {
        return java.util.Arrays.copyOfRange(traj, from, traj.length);
    }

    private static RolloutRow mpRow(String name, double[][][] traj, double[][][] nu, int train, int history,
            int bundle, boolean pushforward, int iterations, int steps, long seed) {
        MessagePassingPdeSolver solver = MessagePassingPdeSolver.of(history, bundle, 1, 16, 2, 2, seed);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        long start = System.nanoTime();
        int batch = 8;
        int length = history + (pushforward ? 2 : 1) * bundle;
        for (int it = 0; it < iterations; it++) {
            int[] ids = new int[batch];
            int[] starts = new int[batch];
            for (int k = 0; k < batch; k++) {
                ids[k] = r.nextInt(train);
                starts[k] = r.nextInt(steps + 1 - length + 1);
            }
            Tensor w = window(traj, nu, ids, starts, length, false);
            Tensor c = coefficients(nu, ids, traj[0][0].length);
            boolean pf = pushforward && r.nextDouble() < 0.5;
            Tensor win = pf ? w : w.slice(2, w.shape().dim(2) - history - bundle, w.shape().dim(2));
            Var loss = solver.loss(win, c, pf);
            loss.backward();
            opt.step(solver.parameters());
        }
        double ms = (System.nanoTime() - start) / 1e6;
        int test = traj.length - train;
        int horizon = steps + 1 - history;
        int[] ids = new int[test];
        int[] starts = new int[test];
        for (int i = 0; i < test; i++) {
            ids[i] = train + i;
        }
        Tensor init = window(traj, nu, ids, starts, history, false);
        int bundles = (horizon + bundle - 1) / bundle;
        float[] out = solver.rollout(init, coefficients(nu, ids, traj[0][0].length), bundles).toArray();
        int n = traj[0][0].length;
        int total = bundles * bundle;
        double[][][] pred = new double[test][horizon][n];
        for (int i = 0; i < test; i++) {
            for (int t = 0; t < horizon; t++) {
                for (int j = 0; j < n; j++) {
                    pred[i][t][j] = out[(i * n + j) * total + t];
                }
            }
        }
        return score(name, pred, testSlice(traj, train), history, ms);
    }

    private static Tensor coefficients(double[][][] nu, int[] ids, int n) {
        float[] f = new float[ids.length * n];
        for (int k = 0; k < ids.length; k++) {
            java.util.Arrays.fill(f, k * n, (k + 1) * n, (float) (nu[ids[k]][0][0] * 10));
        }
        return Tensor.of(Shape.of(ids.length, n, 1), f);
    }

    private static RolloutRow fnoRollout(double[][][] traj, double[][][] nu, int train, int history, int iterations,
            int steps, long seed) {
        int n = traj[0][0].length;
        FourierNeuralOperator1d f = new FourierNeuralOperator1d(history + 1, 1, 12, 8, 2, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        long start = System.nanoTime();
        int batch = 8;
        for (int it = 0; it < iterations; it++) {
            int[] ids = new int[batch];
            int[] starts = new int[batch];
            for (int k = 0; k < batch; k++) {
                ids[k] = r.nextInt(train);
                starts[k] = r.nextInt(steps + 1 - history);
            }
            Tensor w = window(traj, nu, ids, starts, history + 1, true);
            Tensor in = Tensor.concat(2, w.slice(2, 0, history), w.slice(2, history + 1, history + 2));
            Tensor target = w.slice(2, history, history + 1);
            Tensor last = w.slice(2, history - 1, history);
            Var loss = f.forward(Var.of(in)).add(last).sub(target).square().mean();
            loss.backward();
            opt.step(f.parameters());
        }
        double ms = (System.nanoTime() - start) / 1e6;
        int test = traj.length - train;
        int horizon = steps + 1 - history;
        double[][][] pred = new double[test][horizon][n];
        int[] ids = new int[test];
        for (int i = 0; i < test; i++) {
            ids[i] = train + i;
        }
        Tensor state = window(traj, nu, ids, new int[test], history, false);
        Tensor c = coefficients(nu, ids, n);
        for (int t = 0; t < horizon; t++) {
            Tensor cur = state;
            Tensor next = Var.noGrad(() -> f.forward(Var.of(Tensor.concat(2, cur, c)))).value()
                    .add(cur.slice(2, history - 1, history));
            float[] v = next.toArray();
            for (int i = 0; i < test; i++) {
                for (int j = 0; j < n; j++) {
                    pred[i][t][j] = v[i * n + j];
                }
            }
            state = Tensor.concat(2, cur.slice(2, 1, history), next);
        }
        return score("FNO autoregressive", pred, testSlice(traj, train), history, ms);
    }

    // ---------------------------------------------------------------- 8.3.7 PDE-Refiner

    /**
     * A refiner row.
     *
     * @param model the model
     * @param correlationTime mean steps until the correlation with the truth drops below 0.8
     * @param lowBandError relative spectral energy error of the lowest third of modes
     * @param highBandError the same for the upper half of the resolved modes
     * @param inferenceMillis rollout time
     */
    public record RefinerRow(String model, double correlationTime, double lowBandError, double highBandError,
            double inferenceMillis) {
    }

    /**
     * Kuramoto-Sivashinsky trajectories on {@code [0, 22)} after a burn-in.
     *
     * @param count trajectories
     * @param points grid points
     * @param steps snapshots after the initial one
     * @param dt the snapshot interval
     * @param seed the seed
     * @return {@code [count][steps + 1][points]}
     */
    public static double[][][] kuramotoSivashinsky(int count, int points, int steps, double dt, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] out = new double[count][][];
        for (int i = 0; i < count; i++) {
            double[] u0 = PdeProblems.randomField(points, 3, 1, r);
            double[] warm = PdeProblems.solve(PdeProblems.Equation.KURAMOTO_SIVASHINSKY, u0, 0, 22, 0, 40, 2, 0.05)
                    .last();
            out[i] = PdeProblems.solve(PdeProblems.Equation.KURAMOTO_SIVASHINSKY, warm, 0, 22, 0, steps * dt,
                    steps + 1, 0.05).u();
        }
        return out;
    }

    private static double[] bandEnergy(double[][] states) {
        int n = states[0].length;
        double low = 0;
        double high = 0;
        for (double[] s : states) {
            double[] spec = PdeProblems.spectrum(s);
            for (int k = 1; k <= n / 2; k++) {
                double e = spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1];
                if (k <= n / 6) {
                    low += e;
                } else if (k > n / 4) {
                    high += e;
                }
            }
        }
        return new double[] {low, high};
    }

    private static RefinerRow refinerScore(String name, double[][][] pred, double[][][] truth, double ms) {
        double alive = 0;
        double[] ep = new double[2];
        double[] et = new double[2];
        for (int i = 0; i < pred.length; i++) {
            int a = pred[i].length;
            for (int t = 0; t < pred[i].length; t++) {
                if (correlation(pred[i][t], truth[i][t + 1]) < 0.8) {
                    a = t;
                    break;
                }
            }
            alive += a;
            // spectra over the correlated part of the rollout (the first 10 steps)
            int span = Math.min(10, pred[i].length);
            double[] bp = bandEnergy(java.util.Arrays.copyOfRange(pred[i], 0, span));
            double[] bt = bandEnergy(java.util.Arrays.copyOfRange(truth[i], 1, 1 + span));
            for (int k = 0; k < 2; k++) {
                ep[k] += bp[k];
                et[k] += bt[k];
            }
        }
        return new RefinerRow(name, alive / pred.length, Math.abs(ep[0] - et[0]) / et[0],
                Math.abs(ep[1] - et[1]) / et[1], ms);
    }

    private static Tensor states(double[][][] traj, int[] ids, int[] times) {
        int n = traj[0][0].length;
        float[] f = new float[ids.length * n];
        for (int k = 0; k < ids.length; k++) {
            for (int j = 0; j < n; j++) {
                f[k * n + j] = (float) traj[ids[k]][times[k]][j];
            }
        }
        return Tensor.of(Shape.of(ids.length, n), f);
    }

    /**
     * 8.3.7: PDE-Refiner on Kuramoto-Sivashinsky rollouts.
     *
     * @param trajectories the trajectories; the last {@code test} are held out
     * @param test test trajectories
     * @param refinements refinement steps {@code K}
     * @param iterations training iterations
     * @param seed the seed
     * @return rows for PDE-Refiner, the one-step model, FNO and MP-PDE
     */
    public static List<RefinerRow> refiner(double[][][] trajectories, int test, int refinements, int iterations,
            long seed) {
        int train = trajectories.length - test;
        int steps = trajectories[0].length - 1;
        int n = trajectories[0][0].length;
        List<RefinerRow> rows = new ArrayList<>();
        double[][][] truth = java.util.Arrays.copyOfRange(trajectories, train, trajectories.length);
        int[] testIds = new int[test];
        for (int i = 0; i < test; i++) {
            testIds[i] = train + i;
        }
        Tensor init = states(trajectories, testIds, new int[test]);
        for (int k : new int[] {refinements, 0}) {
            PdeRefiner model = PdeRefiner.of(n, 12, 8, 2, k, 2e-3, seed);
            Optimizer opt = Optimizer.adam(2e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            for (int it = 0; it < iterations; it++) {
                int[] ids = new int[16];
                int[] t0 = new int[16];
                int[] t1 = new int[16];
                for (int b = 0; b < 16; b++) {
                    ids[b] = r.nextInt(train);
                    t0[b] = r.nextInt(steps);
                    t1[b] = t0[b] + 1;
                }
                Var loss = model.loss(states(trajectories, ids, t0), states(trajectories, ids, t1), r);
                loss.backward();
                opt.step(model.parameters());
            }
            long start = System.nanoTime();
            float[][] roll = model.rollout(init, steps, new SplittableRandom(seed + 1));
            double ms = (System.nanoTime() - start) / 1e6;
            rows.add(refinerScore(k > 0 ? "PDE-Refiner (K=" + k + ")" : "one-step MSE (same network)",
                    unpack(roll, test, n), truth, ms));
        }
        // FNO one-step on the same transitions
        FourierNeuralOperator1d f = new FourierNeuralOperator1d(1, 1, 12, 8, 2, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            int[] ids = new int[16];
            int[] t0 = new int[16];
            int[] t1 = new int[16];
            for (int b = 0; b < 16; b++) {
                ids[b] = r.nextInt(train);
                t0[b] = r.nextInt(steps);
                t1[b] = t0[b] + 1;
            }
            Tensor a = states(trajectories, ids, t0);
            Tensor b = states(trajectories, ids, t1);
            Var loss = f.forward(Var.of(a.reshape(16, n, 1))).reshape(16, n).add(a).sub(b).square().mean();
            loss.backward();
            opt.step(f.parameters());
        }
        long start = System.nanoTime();
        float[][] roll = new float[steps][];
        Tensor u = init;
        for (int t = 0; t < steps; t++) {
            Tensor cur = u;
            u = Var.noGrad(() -> f.forward(Var.of(cur.reshape(test, n, 1)))).value().reshape(test, n).add(cur);
            roll[t] = u.toArray();
        }
        rows.add(refinerScore("FNO autoregressive", unpack(roll, test, n), truth, (System.nanoTime() - start) / 1e6));
        // MP-PDE, one-step, pushforward
        MessagePassingPdeSolver mp = MessagePassingPdeSolver.of(1, 1, 0, 16, 2, 2, seed);
        Optimizer mo = Optimizer.adam(2e-3f);
        for (int it = 0; it < iterations; it++) {
            int[] ids = new int[16];
            int[] t0 = new int[16];
            for (int b = 0; b < 16; b++) {
                ids[b] = r.nextInt(train);
                t0[b] = r.nextInt(steps - 1);
            }
            float[] w = new float[16 * n * 3];
            for (int b = 0; b < 16; b++) {
                for (int j = 0; j < n; j++) {
                    for (int t = 0; t < 3; t++) {
                        w[(b * n + j) * 3 + t] = (float) trajectories[ids[b]][t0[b] + t][j];
                    }
                }
            }
            Tensor win = Tensor.of(Shape.of(16, n, 3), w);
            boolean pf = r.nextBoolean();
            Var loss = mp.loss(pf ? win : win.slice(2, 1, 3), null, pf);
            loss.backward();
            mo.step(mp.parameters());
        }
        start = System.nanoTime();
        float[] out = mp.rollout(init.reshape(test, n, 1), null, steps).toArray();
        double ms = (System.nanoTime() - start) / 1e6;
        double[][][] pred = new double[test][steps][n];
        for (int i = 0; i < test; i++) {
            for (int t = 0; t < steps; t++) {
                for (int j = 0; j < n; j++) {
                    pred[i][t][j] = out[(i * n + j) * steps + t];
                }
            }
        }
        rows.add(refinerScore("MP-PDE (pushforward)", pred, truth, ms));
        return rows;
    }

    private static double[][][] unpack(float[][] roll, int test, int n) {
        double[][][] pred = new double[test][roll.length][n];
        for (int t = 0; t < roll.length; t++) {
            for (int i = 0; i < test; i++) {
                for (int j = 0; j < n; j++) {
                    pred[i][t][j] = roll[t][i * n + j];
                }
            }
        }
        return pred;
    }

    // ---------------------------------------------------------------- 9.5.16 size, data and grid sweep

    /**
     * Sample positions for the 9.5.16 sweep: the uniform grid of {@code m} points with a share
     * of them dropped at random, an irregular point cloud on which FNO needs interpolation.
     *
     * @param m the uniform grid size
     * @param drop the share of points dropped, 0 for the uniform grid
     * @param seed the seed
     * @return the sorted kept positions in {@code [0, 1)}
     */
    public static double[] subsampledPositions(int m, double drop, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int keep = (int) Math.round(m * (1 - drop));
        int[] idx = new int[m];
        for (int j = 0; j < m; j++) {
            idx[j] = j;
        }
        for (int j = m - 1; j > 0; j--) {
            int k = r.nextInt(j + 1);
            int t = idx[j];
            idx[j] = idx[k];
            idx[k] = t;
        }
        double[] p = new double[keep];
        for (int j = 0; j < keep; j++) {
            p[j] = idx[j] / (double) m;
        }
        java.util.Arrays.sort(p);
        return p;
    }

    /**
     * One cell of the 9.5.16 sweep.
     *
     * @param model the model
     * @param width the channel width
     * @param train training pairs
     * @param drop the share of grid points dropped
     * @param error relative L2 test error at the kept points
     * @param millis training time
     */
    public record SweepRow(String model, int width, int train, double drop, double error, double millis) {
    }

    /**
     * 9.5.16: Transolver and LANO against FNO (interpolated onto the uniform grid when points
     * are dropped) and DeepONet, across channel width, training-set size and grid regularity.
     * Every model sees the same kept points; FNO's width is the same {@code width} and it keeps
     * a quarter of the grid's modes. Each model gets the same number of Adam steps (batches of
     * 20), so a larger training set is seen fewer times rather than buying more compute.
     *
     * @param data pairs on a fine grid; the last {@code test} are held out
     * @param test held-out pairs
     * @param points the uniform grid the points are drawn from
     * @param widths channel widths
     * @param trains training-set sizes
     * @param drops shares of points dropped
     * @param steps Adam steps per model
     * @param seed the seed
     * @return one row per model and cell
     */
    public static List<SweepRow> sweep(Data data, int test, int points, int[] widths, int[] trains, double[] drops,
            int steps, long seed) {
        Data te = data.range(data.size() - test, data.size());
        List<SweepRow> rows = new ArrayList<>();
        for (double drop : drops) {
            double[] pos = drop == 0 ? positions(points, false, seed) : subsampledPositions(points, drop, seed + 7);
            Tensor xTe = sampleAt(te.inputs(), pos);
            Tensor yTe = sampleAt(te.outputs(), pos);
            for (int n : trains) {
                Data tr = data.range(0, Math.min(n, data.size() - test));
                Tensor xTr = sampleAt(tr.inputs(), pos);
                Tensor yTr = sampleAt(tr.outputs(), pos);
                int epochs = Math.max(1, steps * 20 / tr.size());
                for (int c : widths) {
                    for (Map.Entry<String, GridModel> e : sweepModels(c, points, pos, drop, seed).entrySet()) {
                        double ms = train(e.getValue(), xTr, yTr, epochs, 20, 2e-3f, seed);
                        float[] pred = Var.noGrad(() -> e.getValue().apply(Var.of(xTe))).value().toArray();
                        rows.add(new SweepRow(e.getKey(), c, tr.size(), drop, relative(pred, yTe), ms));
                    }
                }
            }
        }
        return rows;
    }

    private static Map<String, GridModel> sweepModels(int c, int points, double[] pos, double drop, long seed) {
        Map<String, GridModel> models = new LinkedHashMap<>();
        models.put("Transolver", new PointTransformer(i -> PhysicsAttention.of(PhysicsAttention.Mode.TRANSOLVER, c,
                8, seed + i), c, 1, pos, seed));
        models.put("LANO", new PointTransformer(i -> PhysicsAttention.of(PhysicsAttention.Mode.LANO, c, 8, seed + i),
                c, 1, pos, seed));
        FourierNeuralOperator1d f = new FourierNeuralOperator1d(1, 1, c, Math.max(4, points / 4), 2,
                new SplittableRandom(seed));
        GridModel grid = new GridModel() {
            @Override
            public Var apply(Var u) {
                int b = u.shape().dim(0);
                return f.forward(u.reshape(b, points, 1)).reshape(b, points);
            }

            @Override
            public List<Var> parameters() {
                return f.parameters();
            }
        };
        models.put("FNO", drop == 0 ? grid : onGridViaInterpolation(grid, pos, points));
        DeepOnet onet = DeepOnet.of(pos.length, 1, 2 * c, 2, c, seed);
        float[] q = new float[pos.length];
        for (int j = 0; j < pos.length; j++) {
            q[j] = (float) pos[j];
        }
        Tensor yq = Tensor.of(Shape.of(pos.length, 1), q);
        models.put("DeepONet", new GridModel() {
            @Override
            public Var apply(Var u) {
                return onet.forward(u, Var.of(yq));
            }

            @Override
            public List<Var> parameters() {
                return onet.parameters();
            }
        });
        return models;
    }
}
