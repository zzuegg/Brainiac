package io.github.zzuegg.brainiac.experiments.communications;

import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.data.ShapeClips;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.panel.TriplyRobustPanel;
import io.github.zzuegg.brainiac.experimental.signal.DenseMatrices;
import io.github.zzuegg.brainiac.experimental.signal.DictionaryLearning;
import io.github.zzuegg.brainiac.experimental.signal.RobustPca;
import io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.Method;
import io.github.zzuegg.brainiac.experimental.signal.UnrolledSolvers;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Sparse recovery, unrolled solvers, dictionary learning and robust PCA (wave 12, items
 * 12.4.11-12.4.14).
 *
 * <ul>
 *   <li>{@link #phaseDiagram} and {@link #convergence}: OMP, ISTA, FISTA and AMP against
 *       minimum-norm least squares.</li>
 *   <li>{@link #unrolled}: LISTA, LISTA-CP, ALISTA and LAMP against ISTA, FISTA and AMP at equal
 *       layers, in distribution, at doubled sparsity and with measurement noise.</li>
 *   <li>{@link #plantedAtoms} and {@link #denoising}: K-SVD and online dictionary learning
 *       against an overcomplete DCT and PCA truncation.</li>
 *   <li>{@link #corruptedMatrices} and {@link #video}: principal component pursuit and
 *       alternating projections against truncated PCA, the panel matrix completion oracle and a
 *       median background.</li>
 * </ul>
 */
public final class SparseRecovery {

    private static final List<Method> METHODS = List.of(Method.OMP, Method.ISTA, Method.FISTA, Method.AMP,
            Method.MIN_NORM_LS);

    private SparseRecovery() {
    }

    /**
     * The empirical phase diagram per method.
     *
     * @param n the signal length
     * @param deltas sampling ratios
     * @param rhos sparsity ratios
     * @param trials trials per cell
     * @param iterations iterations of the iterative methods
     * @param seed the seed
     * @return method name to {@code [delta][rho]} success rates
     */
    public static Map<String, double[][]> phaseDiagram(int n, double[] deltas, double[] rhos, int trials,
            int iterations, long seed) {
        Map<String, double[][]> out = new LinkedHashMap<>();
        for (Method m : METHODS) {
            out.put(m.name(), io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.phaseTransition(m, n,
                    deltas, rhos, trials, iterations, seed));
        }
        return out;
    }

    /**
     * Convergence on noiseless problems.
     *
     * @param m measurements
     * @param n signal length
     * @param s sparsity
     * @param iterations the iteration budget
     * @param trials problems averaged
     * @param seed the seed
     * @return method name to {@code {NMSE in dB after 20 iterations, mean iterations to NMSE 1e-4
     *     (the budget when never reached)}}
     */
    public static Map<String, double[]> convergence(int m, int n, int s, int iterations, int trials, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Method method : METHODS) {
            out.put(method.name(), new double[2]);
        }
        for (int t = 0; t < trials; t++) {
            double[][] a = io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.gaussianMatrix(m, n, rnd);
            double[] x = io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.sparseVector(n, s, rnd);
            double[] y = DenseMatrices.multiply(a, x);
            for (Method method : METHODS) {
                double[] trace = new double[iterations];
                double lambda = method == Method.AMP ? 1.5
                        : 1e-3 * io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.maxAbs(
                                DenseMatrices.multiplyTransposed(a, y));
                double[] est = io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.solve(method, a, y, s,
                        lambda, iterations, x, trace);
                if (method == Method.MIN_NORM_LS) {
                    Arrays.fill(trace, io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.nmse(est, x));
                }
                int reach = iterations;
                for (int i = 0; i < iterations; i++) {
                    if (trace[i] < 1e-4) {
                        reach = i + 1;
                        break;
                    }
                }
                double[] row = out.get(method.name());
                row[0] += db(trace[Math.min(19, iterations - 1)]) / trials;
                row[1] += reach / (double) trials;
            }
        }
        return out;
    }

    private static double db(double v) {
        return 10 * Math.log10(Math.max(v, 1e-12));
    }

    /**
     * Unrolled solvers against their iterative counterparts at equal depth.
     *
     * @param m measurements
     * @param n signal length
     * @param s training sparsity
     * @param layers layers and iterations
     * @param steps training steps
     * @param batch vectors per step
     * @param tests test vectors per condition
     * @param seed the seed
     * @return name to {@code {NMSE dB in distribution, at sparsity 2s, at noise 0.02, learned
     *     parameters}}
     */
    public static Map<String, double[]> unrolled(int m, int n, int s, int layers, int steps, int batch, int tests,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] a = io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.gaussianMatrix(m, n, rnd);
        Map<String, double[]> out = new LinkedHashMap<>();
        UnrolledSolvers[] nets = new UnrolledSolvers[UnrolledSolvers.Kind.values().length];
        for (UnrolledSolvers.Kind kind : UnrolledSolvers.Kind.values()) {
            UnrolledSolvers net = new UnrolledSolvers(kind, a, layers, 0.1);
            net.train(s, 0, steps, batch, kind == UnrolledSolvers.Kind.ALISTA ? 0.01f : 2e-3f, seed + kind.ordinal());
            nets[kind.ordinal()] = net;
            out.put(kind.name(), new double[] {0, 0, 0, net.learnedParameters()});
        }
        Method[] classical = {Method.ISTA, Method.FISTA, Method.AMP};
        for (Method c : classical) {
            out.put(c.name() + " (" + layers + " iterations)", new double[4]);
        }
        int[] sparsities = {s, 2 * s, s};
        double[] noises = {0, 0, 0.02};
        for (int cond = 0; cond < 3; cond++) {
            double[] err = new double[out.size()];
            double energy = 0;
            for (int t = 0; t < tests; t++) {
                double[] x = io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.sparseVector(n,
                        sparsities[cond], rnd);
                double[] y = DenseMatrices.multiply(a, x);
                for (int i = 0; i < m; i++) {
                    y[i] += noises[cond] * ChannelCodes.gaussian(rnd);
                }
                for (double v : x) {
                    energy += v * v;
                }
                int k = 0;
                for (UnrolledSolvers net : nets) {
                    err[k++] += sq(net.solve(y), x);
                }
                for (Method c : classical) {
                    double lambda = c == Method.AMP ? 1.5 : 0.1;
                    err[k++] += sq(io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.solve(c, a, y, 0, lambda,
                            layers, null, null), x);
                }
            }
            int k = 0;
            for (double[] row : out.values()) {
                row[cond] = db(err[k++] / energy);
            }
        }
        return out;
    }

    private static double sq(double[] est, double[] x) {
        double e = 0;
        for (int i = 0; i < x.length; i++) {
            e += (est[i] - x[i]) * (est[i] - x[i]);
        }
        return e;
    }

    /**
     * Recovery of a planted dictionary from noisy sparse combinations.
     *
     * @param dim the signal dimension
     * @param atoms planted atoms
     * @param signals training signals
     * @param sparsity atoms per signal
     * @param iterations learning iterations (K-SVD) or passes (online)
     * @param seed the seed
     * @return {@code {K-SVD recovered share, K-SVD ms, online recovered share, online ms}}
     */
    public static double[] plantedAtoms(int dim, int atoms, int signals, int sparsity, int iterations, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] d = new double[dim][atoms];
        for (int k = 0; k < atoms; k++) {
            double norm = 0;
            for (int i = 0; i < dim; i++) {
                d[i][k] = ChannelCodes.gaussian(rnd);
                norm += d[i][k] * d[i][k];
            }
            for (int i = 0; i < dim; i++) {
                d[i][k] /= Math.sqrt(norm);
            }
        }
        double[][] data = new double[signals][dim];
        for (double[] sig : data) {
            double[] code = io.github.zzuegg.brainiac.experimental.signal.SparseRecovery.sparseVector(atoms, sparsity,
                    rnd);
            double[] v = DenseMatrices.multiply(d, code);
            for (int i = 0; i < dim; i++) {
                sig[i] = v[i] + 0.01 * ChannelCodes.gaussian(rnd);
            }
        }
        double[] out = new double[4];
        int k = 0;
        for (DictionaryLearning.Method method : DictionaryLearning.Method.values()) {
            long t0 = System.nanoTime();
            double[][] learned = DictionaryLearning.learn(method, data, atoms, sparsity, iterations, seed + 1);
            out[k + 1] = (System.nanoTime() - t0) / 1e6;
            out[k] = DictionaryLearning.recovered(d, learned, 0.99);
            k += 2;
        }
        return out;
    }

    /**
     * A mosaic of small {@link ShapeImages}.
     *
     * @param tiles tiles per axis
     * @param seed the seed
     * @return the image, {@code 8 * tiles} square
     */
    public static double[][] mosaic(int tiles, long seed) {
        float[] img = new ShapeImages(8, 0).generate(tiles * tiles, seed)[0].toArray();
        double[][] out = new double[8 * tiles][8 * tiles];
        for (int t = 0; t < tiles * tiles; t++) {
            int r0 = 8 * (t / tiles);
            int c0 = 8 * (t % tiles);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    out[r0 + i][c0 + j] = img[t * 64 + i * 8 + j];
                }
            }
        }
        return out;
    }

    /**
     * Patch denoising PSNR in dB.
     *
     * @param sigmas noise levels
     * @param tiles mosaic tiles per axis
     * @param atoms learned dictionary size
     * @param iterations learning iterations
     * @param seed the seed
     * @return name to PSNR per noise level
     */
    public static Map<String, double[]> denoising(double[] sigmas, int tiles, int atoms, int iterations, long seed) {
        double[][] clean = mosaic(tiles, seed);
        SplittableRandom rnd = new SplittableRandom(seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"noisy", "overcomplete DCT (121)", "PCA truncation (top 16)", "K-SVD (" + atoms + ")",
            "online DL (" + atoms + ")"};
        for (String name : names) {
            out.put(name, new double[sigmas.length]);
        }
        double[][] dct = DictionaryLearning.overcompleteDct(8, 11);
        for (int si = 0; si < sigmas.length; si++) {
            double sigma = sigmas[si];
            double[][] noisy = new double[clean.length][clean.length];
            for (int i = 0; i < clean.length; i++) {
                for (int j = 0; j < clean.length; j++) {
                    noisy[i][j] = clean[i][j] + sigma * ChannelCodes.gaussian(rnd);
                }
            }
            double[][] train = DictionaryLearning.patches(noisy, 8, 2);
            ThinSvd svd = ThinSvd.of(train);
            double[][] pca = new double[64][16];
            for (int i = 0; i < 64; i++) {
                for (int k = 0; k < 16; k++) {
                    pca[i][k] = svd.v()[i][k];
                }
            }
            out.get(names[0])[si] = DictionaryLearning.psnr(noisy, clean);
            out.get(names[1])[si] = DictionaryLearning.psnr(DictionaryLearning.denoise(noisy, dct, 8, sigma, 16), clean);
            out.get(names[2])[si] = DictionaryLearning.psnr(DictionaryLearning.denoise(noisy, pca, 8, 0, 16), clean);
            double[][] ksvd = DictionaryLearning.learn(DictionaryLearning.Method.K_SVD, train, atoms, 4, iterations,
                    seed);
            out.get(names[3])[si] = DictionaryLearning.psnr(DictionaryLearning.denoise(noisy, ksvd, 8, sigma, 16),
                    clean);
            double[][] online = DictionaryLearning.learn(DictionaryLearning.Method.ONLINE_DL, train, atoms, 4,
                    iterations, seed);
            out.get(names[4])[si] = DictionaryLearning.psnr(DictionaryLearning.denoise(noisy, online, 8, sigma, 16),
                    clean);
        }
        return out;
    }

    /**
     * Low-rank plus sparse matrices.
     *
     * @param size the matrix side
     * @param rank the rank of the low-rank part
     * @param fractions corrupted fractions
     * @param seed the seed
     * @return name to relative error of {@code L} per fraction, plus "... support F1" rows
     */
    public static Map<String, double[]> corruptedMatrices(int size, int rank, double[] fractions, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        String[] names = {"PCP", "AltProj", "PCA (true rank)", "matrix completion oracle", "PCP support F1",
            "AltProj support F1"};
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String name : names) {
            out.put(name, new double[fractions.length]);
        }
        for (int f = 0; f < fractions.length; f++) {
            double[][] u = new double[size][rank];
            double[][] v = new double[size][rank];
            for (int i = 0; i < size; i++) {
                for (int k = 0; k < rank; k++) {
                    u[i][k] = ChannelCodes.gaussian(rnd);
                    v[i][k] = ChannelCodes.gaussian(rnd);
                }
            }
            double[][] l = new double[size][size];
            double[][] s = new double[size][size];
            double[][] mat = new double[size][size];
            double[][] w = new double[size][size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < rank; k++) {
                        l[i][j] += u[i][k] * v[j][k];
                    }
                    if (rnd.nextDouble() < fractions[f]) {
                        s[i][j] = (rnd.nextBoolean() ? 1 : -1) * (3 + 7 * rnd.nextDouble());
                    }
                    mat[i][j] = l[i][j] + s[i][j];
                    w[i][j] = s[i][j] == 0 ? 1 : 0;
                }
            }
            RobustPca.Decomposition pcp = RobustPca.pcp(mat, Double.NaN, 200, 1e-7);
            RobustPca.Decomposition alt = RobustPca.alternatingProjections(mat, rank, 30);
            double[][] oracle = TriplyRobustPanel.matrixCompletion(1, 150).solve(mat, w);
            out.get(names[0])[f] = RobustPca.relativeError(pcp.low(), l);
            out.get(names[1])[f] = RobustPca.relativeError(alt.low(), l);
            out.get(names[2])[f] = RobustPca.relativeError(RobustPca.truncatedPca(mat, rank), l);
            out.get(names[3])[f] = RobustPca.relativeError(oracle, l);
            out.get(names[4])[f] = RobustPca.supportF1(pcp.sparse(), s, 0.5);
            out.get(names[5])[f] = RobustPca.supportF1(alt.sparse(), s, 0.5);
        }
        return out;
    }

    /**
     * Background subtraction on {@link ShapeClips} over a static textured background whose
     * illumination drifts over the clip (a rank-one background a median cannot follow).
     *
     * @param clips clips averaged
     * @param drift the illumination amplitude (0 for a constant background)
     * @param seed the seed
     * @return {@code {PCP IoU, AltProj (rank 1) IoU, median background IoU}}
     */
    public static double[] video(int clips, double drift, long seed) {
        int side = 24;
        int length = 20;
        int pixels = side * side;
        List<ShapeClips.Clip> list = new ShapeClips(side, length, 0).generate(clips, seed);
        SplittableRandom rnd = new SplittableRandom(seed + 3);
        double[] out = new double[3];
        double tau = 0.15;
        for (ShapeClips.Clip clip : list) {
            float[] f = clip.frames();
            double[] background = new double[pixels];
            double fx = 0.3 + 0.4 * rnd.nextDouble();
            double fy = 0.3 + 0.4 * rnd.nextDouble();
            for (int p = 0; p < pixels; p++) {
                int y = p / side;
                int x = p % side;
                background[p] = 0.2 + 0.1 * Math.sin(fx * x) * Math.cos(fy * y) + 0.05 * rnd.nextDouble();
            }
            double[][] mat = new double[pixels][length];
            boolean[][] truth = new boolean[pixels][length];
            for (int p = 0; p < pixels; p++) {
                boolean always = true;
                for (int t = 0; t < length; t++) {
                    always &= f[t * pixels + p] > 0;
                }
                for (int t = 0; t < length; t++) {
                    double gain = 1 + drift * Math.sin(2 * Math.PI * t / length);
                    double v = f[t * pixels + p];
                    mat[p][t] = (v > 0 ? v : gain * background[p]) + 0.01 * ChannelCodes.gaussian(rnd);
                    truth[p][t] = v > 0 && !always;
                }
            }
            RobustPca.Decomposition pcp = RobustPca.pcp(mat, Double.NaN, 200, 1e-7);
            RobustPca.Decomposition alt = RobustPca.alternatingProjections(mat, 1, 20);
            boolean[][] mPcp = new boolean[pixels][length];
            boolean[][] mAlt = new boolean[pixels][length];
            boolean[][] mMed = new boolean[pixels][length];
            for (int p = 0; p < pixels; p++) {
                double[] row = mat[p].clone();
                Arrays.sort(row);
                double median = row[length / 2];
                for (int t = 0; t < length; t++) {
                    mPcp[p][t] = Math.abs(pcp.sparse()[p][t]) > tau;
                    mAlt[p][t] = Math.abs(alt.sparse()[p][t]) > tau;
                    mMed[p][t] = Math.abs(mat[p][t] - median) > tau;
                }
            }
            out[0] += iou(mPcp, truth) / clips;
            out[1] += iou(mAlt, truth) / clips;
            out[2] += iou(mMed, truth) / clips;
        }
        return out;
    }

    private static double iou(boolean[][] a, boolean[][] b) {
        int inter = 0;
        int union = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                inter += a[i][j] && b[i][j] ? 1 : 0;
                union += a[i][j] || b[i][j] ? 1 : 0;
            }
        }
        return union == 0 ? 1 : inter / (double) union;
    }
}
