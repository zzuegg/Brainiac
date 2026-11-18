package io.github.zzuegg.brainiac.experiments.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.EquivariantGraphLayer;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.sciml.EquivariantMlp;
import io.github.zzuegg.brainiac.experimental.sciml.FrameAveraging;
import io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems;
import io.github.zzuegg.brainiac.experimental.sciml.SchNet;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Symmetry-aware models (wave 8, items 8.3.14-8.3.15):
 *
 * <ul>
 *   <li>8.3.14: an {@link EquivariantMlp} (solved equivariant basis), {@link FrameAveraging}
 *       around a plain MLP and an EGNN on an {@code E(3)}-invariant point-cloud regression,
 *       against an MLP with and without random-rotation augmentation, at several data
 *       sizes; test clouds are randomly rotated and reflected.</li>
 *   <li>8.3.15: {@link SchNet} trained on energies and forces against energies only, an EGNN
 *       energy model and an MLP on sorted pair distances, on Lennard-Jones clusters; plus the
 *       true-energy drift of short molecular dynamics driven by each learned force field.</li>
 * </ul>
 */
public final class SymmetricModels {

    private SymmetricModels() {
    }

    // ------------------------------------------------------------------ helpers

    /**
     * A random orthogonal matrix (Haar on SO(3), times a random reflection if asked).
     *
     * @param random the source
     * @param reflections whether to include improper rotations
     * @return {@code [3][3]}
     */
    public static double[][] randomRotation(SplittableRandom random, boolean reflections) {
        double[][] q = new double[3][3];
        for (int c = 0; c < 3; c++) {
            double[] v = {random.nextGaussian(), random.nextGaussian(), random.nextGaussian()};
            for (int p = 0; p < c; p++) {
                double dot = 0;
                for (int k = 0; k < 3; k++) {
                    dot += v[k] * q[k][p];
                }
                for (int k = 0; k < 3; k++) {
                    v[k] -= dot * q[k][p];
                }
            }
            double norm = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            for (int k = 0; k < 3; k++) {
                q[k][c] = v[k] / norm;
            }
        }
        if (reflections && random.nextBoolean()) {
            for (int k = 0; k < 3; k++) {
                q[k][0] = -q[k][0];
            }
        }
        return q;
    }

    static double[] rotate(double[] cloud, double[][] q) {
        double[] out = new double[cloud.length];
        for (int i = 0; i < cloud.length / 3; i++) {
            for (int r = 0; r < 3; r++) {
                double s = 0;
                for (int c = 0; c < 3; c++) {
                    s += q[r][c] * cloud[3 * i + c];
                }
                out[3 * i + r] = s;
            }
        }
        return out;
    }

    static double[] centre(double[] cloud) {
        int n = cloud.length / 3;
        double[] c = new double[3];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 3; k++) {
                c[k] += cloud[3 * i + k] / n;
            }
        }
        double[] out = cloud.clone();
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 3; k++) {
                out[3 * i + k] -= c[k];
            }
        }
        return out;
    }

    static Tensor tensor(double[][] x) {
        return DynamicsLearning.tensor(x);
    }

    private static double distance(double[] cloud, int i, int j) {
        double s = 0;
        for (int k = 0; k < 3; k++) {
            double d = cloud[3 * i + k] - cloud[3 * j + k];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    private static Layer layer(SimpleMlp mlp) {
        return new Layer() {
            @Override
            public Var forward(Var input) {
                return mlp.forward(input);
            }

            @Override
            public List<Var> parameters() {
                return mlp.parameters();
            }
        };
    }

    /** A small EGNN energy model: two equivariant layers on constant node features, sum readout. */
    private static final class Egnn {
        private final int features;
        private final EquivariantGraphLayer l1;
        private final EquivariantGraphLayer l2;
        private final Var readout;
        private final Var bias;

        Egnn(int features, int hidden, long seed) {
            this.features = features;
            this.l1 = EquivariantGraphLayer.of(3, features, hidden, seed);
            this.l2 = EquivariantGraphLayer.of(3, features, hidden, seed + 1);
            this.readout = Var.param(Tensor.normal(Shape.of(features, 1), 0, (float) (1 / Math.sqrt(features)),
                    new SplittableRandom(seed + 2)));
            this.bias = Var.param(Tensor.zeros(Shape.of(1)));
        }

        /** @param x flat clouds {@code [b, n * 3]} @return per-cloud outputs {@code [b, 1]} */
        Var forward(Var x) {
            int b = x.shape().dim(0);
            int n = x.shape().dim(1) / 3;
            Var pos = x.reshape(b, n, 3);
            Var h = Var.of(Tensor.full(Shape.of(b, n, features), 1f));
            Var out = l2.forward(l1.forward(Var.concat(2, pos, h)));
            Var feat = out.slice(2, 3, 3 + features).sum(1, false);   // [b, features]
            return feat.matmul(readout).add(bias);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(l1.parameters());
            p.addAll(l2.parameters());
            p.add(readout);
            p.add(bias);
            return p;
        }
    }

    // ------------------------------------------------------------------ 8.3.14

    /**
     * An invariant-regression row.
     *
     * @param model the model
     * @param samples training clouds
     * @param testMse mean squared error on randomly rotated and reflected test clouds,
     *     relative to the target variance
     * @param invarianceGap mean absolute change of the prediction under a random O(3) map
     * @param millis training time
     */
    public record InvariantRow(String model, int samples, double testMse, double invarianceGap, double millis) {
    }

    /**
     * The {@code E(3)}-invariant target: {@code sum_(i<j) cos(1.5 |x_i - x_j|)} on centred clouds.
     *
     * @param cloud the flat cloud
     * @return the target
     */
    public static double invariantTarget(double[] cloud) {
        int n = cloud.length / 3;
        double s = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                s += Math.cos(1.5 * distance(cloud, i, j));
            }
        }
        return s;
    }

    /**
     * 8.3.14: equivariant MLP and frame averaging against an MLP with and without augmentation.
     *
     * @param points points per cloud
     * @param sizes training set sizes
     * @param steps Adam steps per model (minibatches of 32)
     * @param seed the seed
     * @return rows for each model and size
     */
    public static List<InvariantRow> invariantRegression(int points, int[] sizes, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int max = Arrays.stream(sizes).max().orElseThrow();
        double[][] all = new double[max][];
        double[][] ys = new double[max][1];
        for (int i = 0; i < max; i++) {
            double[] c = new double[3 * points];
            for (int k = 0; k < c.length; k++) {
                c[k] = r.nextGaussian();
            }
            // training clouds in a canonical-ish pose: no rotation applied (the MLP sees one pose)
            all[i] = centre(c);
            ys[i][0] = invariantTarget(all[i]);
        }
        int test = 200;
        double[][] tx = new double[test][];
        double[][] ty = new double[test][1];
        double[][] txRot = new double[test][];
        double mean = 0;
        double var = 0;
        for (int i = 0; i < test; i++) {
            double[] c = new double[3 * points];
            for (int k = 0; k < c.length; k++) {
                c[k] = r.nextGaussian();
            }
            tx[i] = rotate(centre(c), randomRotation(r, true));
            ty[i][0] = invariantTarget(tx[i]);
            txRot[i] = rotate(tx[i], randomRotation(r, true));
            mean += ty[i][0] / test;
        }
        for (int i = 0; i < test; i++) {
            var += (ty[i][0] - mean) * (ty[i][0] - mean) / test;
        }
        Tensor testX = tensor(tx);
        Tensor testXr = tensor(txRot);
        List<InvariantRow> rows = new ArrayList<>();
        for (int size : sizes) {
            double[][] x = Arrays.copyOf(all, size);
            double[][] y = Arrays.copyOf(ys, size);
            for (String name : List.of("MLP", "MLP + rotation augmentation", "frame averaging (MLP)",
                    "equivariant MLP (O(3) basis)", "EGNN")) {
                long t0 = System.nanoTime();
                Function<Var, Var> model;
                List<Var> params;
                switch (name) {
                    case "MLP", "MLP + rotation augmentation" -> {
                        SimpleMlp mlp = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), 3 * points, 64, 64, 1);
                        model = mlp::forward;
                        params = mlp.parameters();
                    }
                    case "frame averaging (MLP)" -> {
                        SimpleMlp mlp = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), 3 * points, 64, 64, 1);
                        FrameAveraging fa = FrameAveraging.invariant(layer(mlp), points, false);
                        model = fa::forward;
                        params = fa.parameters();
                    }
                    case "equivariant MLP (O(3) basis)" -> {
                        EquivariantMlp e = EquivariantMlp.invariant(EquivariantMlp.rotations3d(true), points, 32, 64,
                                1, seed);
                        model = e::forward;
                        params = e.parameters();
                    }
                    default -> {
                        Egnn e = new Egnn(8, 16, seed);
                        model = e::forward;
                        params = e.parameters();
                    }
                }
                boolean augment = name.equals("MLP + rotation augmentation");
                Optimizer opt = Optimizer.adam(3e-3f);
                SplittableRandom ar = new SplittableRandom(seed + 7);
                int batch = Math.min(size, 32);
                for (int s = 0; s < steps; s++) {
                    double[][] bx = new double[batch][];
                    double[][] by = new double[batch][];
                    for (int i = 0; i < batch; i++) {
                        int k = batch == size ? i : ar.nextInt(size);
                        bx[i] = augment ? rotate(x[k], randomRotation(ar, true)) : x[k];
                        by[i] = y[k];
                    }
                    Var loss = model.apply(Var.of(tensor(bx))).sub(tensor(by)).square().mean();
                    loss.backward();
                    opt.step(params);
                }
                double millis = (System.nanoTime() - t0) / 1e6;
                float[] p = Var.noGrad(() -> model.apply(Var.of(testX))).value().toArray();
                float[] pr = Var.noGrad(() -> model.apply(Var.of(testXr))).value().toArray();
                double mse = 0;
                double gap = 0;
                for (int i = 0; i < test; i++) {
                    mse += (p[i] - ty[i][0]) * (p[i] - ty[i][0]) / test;
                    gap += Math.abs(p[i] - pr[i]) / test;
                }
                rows.add(new InvariantRow(name, size, mse / var, gap, millis));
            }
        }
        return rows;
    }

    // ------------------------------------------------------------------ 8.3.15

    /**
     * An interatomic-potential row.
     *
     * @param model the model
     * @param energyMae energy mean absolute error per atom on test clusters
     * @param forceMae force component mean absolute error on test clusters (forces of the
     *     learned energy, by central differences for every model alike)
     * @param energyDrift absolute true-total-energy drift per atom over the learned-force MD
     *     run (NaN where not run)
     * @param millis training time
     */
    public record PotentialRow(String model, double energyMae, double forceMae, double energyDrift, double millis) {
    }

    private static double[] flatten(double[][] x) {
        double[] f = new double[x.length * 3];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, f, 3 * i, 3);
        }
        return f;
    }

    private static double[][] unflatten(double[] f) {
        double[][] x = new double[f.length / 3][3];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(f, 3 * i, x[i], 0, 3);
        }
        return x;
    }

    /** Sorted pair distances, the permutation- and E(3)-invariant MLP input. */
    static double[] sortedDistances(double[] cloud) {
        int n = cloud.length / 3;
        double[] d = new double[n * (n - 1) / 2];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                d[k++] = distance(cloud, i, j);
            }
        }
        Arrays.sort(d);
        return d;
    }

    /** A learned energy on flat clouds, in double precision, for finite-difference forces. */
    private interface Energy {
        double[] of(double[][] clouds);
    }

    private static double[][] forcesFd(Energy e, double[][] clouds) {
        int b = clouds.length;
        int dim = clouds[0].length;
        double h = 1e-3;
        double[][] shifted = new double[b * dim * 2][];
        for (int s = 0; s < b; s++) {
            for (int k = 0; k < dim; k++) {
                double[] p = clouds[s].clone();
                double[] m = clouds[s].clone();
                p[k] += h;
                m[k] -= h;
                shifted[(s * dim + k) * 2] = p;
                shifted[(s * dim + k) * 2 + 1] = m;
            }
        }
        double[] v = e.of(shifted);
        double[][] f = new double[b][dim];
        for (int s = 0; s < b; s++) {
            for (int k = 0; k < dim; k++) {
                f[s][k] = -(v[(s * dim + k) * 2] - v[(s * dim + k) * 2 + 1]) / (2 * h);
            }
        }
        return f;
    }

    private static double[] column(float[] f) {
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /**
     * 8.3.15: SchNet with and without force matching against an EGNN and a sorted-distance
     * MLP on Lennard-Jones clusters.
     *
     * @param atoms atoms per cluster
     * @param train training clusters
     * @param steps Adam steps per model
     * @param mdSteps molecular-dynamics steps (size 0.002) for the drift measurement
     * @param seed the seed
     * @return rows for SchNet (E+F), SchNet (E), EGNN (E) and the MLP (E)
     */
    public static List<PotentialRow> potentials(int atoms, int train, int steps, int mdSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int test = 50;
        double[][] x = new double[train + test][];
        double[] e = new double[train + test];
        double[][] f = new double[train + test][];
        for (int i = 0; i < train + test; i++) {
            double[][] c = MechanicalSystems.lennardJonesCluster(atoms, 0.08, r);
            x[i] = centre(flatten(c));
            e[i] = MechanicalSystems.lennardJonesEnergy(c);
            f[i] = flatten(MechanicalSystems.lennardJonesForces(c));
        }
        double mean = 0;
        double std = 0;
        for (int i = 0; i < train; i++) {
            mean += e[i] / train;
        }
        for (int i = 0; i < train; i++) {
            std += (e[i] - mean) * (e[i] - mean) / train;
        }
        std = Math.sqrt(std) + 1e-9;
        double eMean = mean;
        double eStd = std;
        // normalised targets: energy (E - mean) / std, forces F / std
        float[] pos = new float[train * atoms * 3];
        float[] en = new float[train];
        float[] fo = new float[train * atoms * 3];
        for (int i = 0; i < train; i++) {
            for (int k = 0; k < atoms * 3; k++) {
                pos[i * atoms * 3 + k] = (float) x[i][k];
                fo[i * atoms * 3 + k] = (float) (f[i][k] / eStd);
            }
            en[i] = (float) ((e[i] - eMean) / eStd);
        }
        Tensor posT = Tensor.of(Shape.of(train, atoms, 3), pos);
        Tensor flatT = Tensor.of(Shape.of(train, atoms * 3), pos);
        Tensor enT = Tensor.of(Shape.of(train, 1), en);
        Tensor foT = Tensor.of(Shape.of(train, atoms, 3), fo);
        double[][] testX = Arrays.copyOfRange(x, train, train + test);
        List<PotentialRow> rows = new ArrayList<>();
        for (String name : List.of("SchNet (energy + forces)", "SchNet (energy only)", "EGNN (energy only)",
                "MLP on sorted distances (energy only)")) {
            long t0 = System.nanoTime();
            Energy energy;
            Optimizer opt = Optimizer.adam(3e-3f);
            if (name.startsWith("SchNet")) {
                SchNet net = SchNet.of(24, 2, 16, 3.0, seed);
                float w = name.contains("forces") ? 1f : 0f;
                for (int s = 0; s < steps; s++) {
                    Var loss = net.loss(posT, enT, w > 0 ? foT : null, w);
                    loss.backward();
                    opt.step(net.parameters());
                }
                energy = c -> {
                    Tensor t = tensor(c);
                    return column(Var.noGrad(() -> net.forward(Var.of(t))).value().toArray());
                };
            } else if (name.startsWith("EGNN")) {
                Egnn net = new Egnn(8, 32, seed);
                for (int s = 0; s < steps; s++) {
                    Var loss = net.forward(Var.of(flatT)).sub(enT).square().mean();
                    loss.backward();
                    opt.step(net.parameters());
                }
                energy = c -> {
                    Tensor t = tensor(c);
                    return column(Var.noGrad(() -> net.forward(Var.of(t))).value().toArray());
                };
            } else {
                SimpleMlp mlp = SimpleMlp.of(seed, v -> v.mul(v.sigmoid()), atoms * (atoms - 1) / 2, 64, 64, 1);
                double[][] d = new double[train][];
                for (int i = 0; i < train; i++) {
                    d[i] = sortedDistances(x[i]);
                }
                Tensor dt = tensor(d);
                for (int s = 0; s < steps; s++) {
                    Var loss = mlp.forward(Var.of(dt)).sub(enT).square().mean();
                    loss.backward();
                    opt.step(mlp.parameters());
                }
                energy = c -> {
                    double[][] dd = new double[c.length][];
                    for (int i = 0; i < c.length; i++) {
                        dd[i] = sortedDistances(c[i]);
                    }
                    return column(mlp.predict(tensor(dd)).toArray());
                };
            }
            double millis = (System.nanoTime() - t0) / 1e6;
            Energy physical = c -> {
                double[] v = energy.of(c);
                for (int i = 0; i < v.length; i++) {
                    v[i] = v[i] * eStd + eMean;
                }
                return v;
            };
            double[] pe = physical.of(testX);
            double[][] pf = forcesFd(physical, testX);
            double emae = 0;
            double fmae = 0;
            for (int i = 0; i < test; i++) {
                emae += Math.abs(pe[i] - e[train + i]) / (test * atoms);
                for (int k = 0; k < atoms * 3; k++) {
                    fmae += Math.abs(pf[i][k] - f[train + i][k]) / (test * atoms * 3);
                }
            }
            double drift = mdSteps > 0 ? md(physical, testX[0], mdSteps) / atoms : Double.NaN;
            rows.add(new PotentialRow(name, emae, fmae, drift, millis));
        }
        return rows;
    }

    /**
     * Velocity-Verlet MD with learned forces from rest; returns the largest deviation of the
     * true total energy (LJ potential plus kinetic) from its start.
     */
    private static double md(Energy model, double[] start, int steps) {
        double dt = 0.002;
        double[] x = start.clone();
        double[] v = new double[x.length];
        double e0 = MechanicalSystems.lennardJonesEnergy(unflatten(x));
        double[] a = forcesFd(model, new double[][] {x})[0];
        double worst = 0;
        for (int s = 0; s < steps; s++) {
            for (int k = 0; k < x.length; k++) {
                v[k] += 0.5 * dt * a[k];
                x[k] += dt * v[k];
            }
            a = forcesFd(model, new double[][] {x})[0];
            double kinetic = 0;
            for (int k = 0; k < x.length; k++) {
                v[k] += 0.5 * dt * a[k];
                kinetic += 0.5 * v[k] * v[k];
            }
            double total = MechanicalSystems.lennardJonesEnergy(unflatten(x)) + kinetic;
            worst = Math.max(worst, Math.abs(total - e0));
            if (!Double.isFinite(total)) {
                return Double.MAX_VALUE;
            }
        }
        return worst;
    }
}
