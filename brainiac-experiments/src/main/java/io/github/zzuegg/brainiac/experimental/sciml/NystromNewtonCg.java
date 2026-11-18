package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

/**
 * Nyström-preconditioned Newton-CG (NNCG, Rathore et al., ICML 2024,
 * <a href="https://arxiv.org/abs/2402.01868">arXiv:2402.01868</a>), the finishing stage of
 * their recommended PINN recipe Adam, then L-BFGS, then NNCG. The PINN loss is badly
 * conditioned because its Hessian inherits the differential operator's spectrum: a few huge
 * eigenvalues and a long tail. Each iteration solves the damped Newton system
 * {@code (H + mu I) d = -g} by preconditioned conjugate gradients on exact Hessian-vector
 * products ({@link CurvatureProducts#hessianVector}, double backpropagation), stopping at
 * negative curvature, and takes an Armijo-backtracked step.
 *
 * <p>The preconditioner is a randomized Nyström approximation of rank {@code s}: with a
 * random orthonormal test matrix {@code Omega} and {@code Y = H Omega} ({@code s} products),
 * shift {@code nu}, {@code Y_nu = Y + nu Omega}, Cholesky {@code Omega^T Y_nu = C C^T},
 * {@code B = Y_nu C^-T} and its thin SVD {@code B = U Sigma V^T}, the approximation is
 * {@code H ~ U Lambda U^T} with {@code Lambda = max(0, Sigma^2 - nu)}, and
 * {@code P^-1 = (lambda_s + mu) U (Lambda + mu I)^-1 U^T + (I - U U^T)}: the top of the
 * spectrum is flattened, which is exactly where PINN conditioning fails. It is refreshed every
 * {@code refresh} iterations.
 */
public final class NystromNewtonCg {

    private final int rank;
    private final double damping;
    private final int cgIterations;
    private final int refresh;
    private double[][] u;
    private double[] scale;
    private double tail = 1;

    private NystromNewtonCg(int rank, double damping, int cgIterations, int refresh) {
        this.rank = rank;
        this.damping = damping;
        this.cgIterations = cgIterations;
        this.refresh = refresh;
    }

    /**
     * Creates the optimiser.
     *
     * @param rank the Nyström rank {@code s}, for example 20 to 60
     * @param damping the damping {@code mu}, for example 1e-3 relative to the loss scale
     * @param cgIterations the conjugate-gradient iterations per Newton step
     * @param refresh iterations between preconditioner rebuilds
     * @return the optimiser
     */
    public static NystromNewtonCg of(int rank, double damping, int cgIterations, int refresh) {
        return new NystromNewtonCg(rank, damping, cgIterations, refresh);
    }

    /**
     * Trains a PINN (usually after Adam and L-BFGS).
     *
     * @param pinn the network, trained in place
     * @param problem the problem
     * @param collocation collocation points (drawn once)
     * @param iterations Newton iterations
     * @param monitor evaluated after every iteration, or {@code null}
     * @param seed the seed
     * @return the outcome
     */
    public PinnNaturalGradient.Result train(PhysicsInformedNetwork pinn, PinnTraining.Problem problem,
            int collocation, int iterations, ToDoubleFunction<PhysicsInformedNetwork> monitor, long seed) {
        long start = System.nanoTime();
        SplittableRandom random = new SplittableRandom(seed);
        Tensor points = CollocationSampler.tensor(problem.domain().latinHypercube(collocation, random));
        List<Var> params = pinn.parameters();
        Supplier<Var> loss = () -> PinnNaturalGradient.residuals(pinn, problem, points).square().sum();
        double[] losses = new double[iterations];
        double[] mon = new double[monitor == null ? 0 : iterations];
        double[] times = new double[iterations];
        double l0 = Double.NaN;
        for (int it = 0; it < iterations; it++) {
            Var lv = loss.get();
            l0 = lv.value().item();
            double[] g = CurvatureProducts.gradient(lv, params);
            if (it % refresh == 0) {
                buildPreconditioner(loss, params, random);
            }
            double[] d = pcg(v -> {
                double[] hv = CurvatureProducts.hessianVector(loss, params, v);
                return CurvatureProducts.axpy(hv, damping, v);
            }, negate(g));
            double slope = CurvatureProducts.dot(g, d);
            if (!(slope < 0)) {
                d = negate(g);
                slope = -CurvatureProducts.dot(g, g);
            }
            double[] theta = CurvatureProducts.values(params);
            double eta = 1;
            double accepted = l0;
            boolean moved = false;
            for (int k = 0; k < 20; k++) {
                CurvatureProducts.assign(params, CurvatureProducts.axpy(theta, eta, d));
                double l = loss.get().value().item();
                if (Double.isFinite(l) && l <= l0 + 1e-4 * eta * slope) {
                    accepted = l;
                    moved = true;
                    break;
                }
                eta /= 2;
            }
            if (!moved) {
                CurvatureProducts.assign(params, theta);
            }
            l0 = accepted;
            losses[it] = accepted;
            if (monitor != null) {
                mon[it] = monitor.applyAsDouble(pinn);
            }
            times[it] = (System.nanoTime() - start) / 1e6;
        }
        return new PinnNaturalGradient.Result(l0, (System.nanoTime() - start) / 1e6, losses, mon, times);
    }

    private static double[] negate(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = -v[i];
        }
        return r;
    }

    /** Builds the rank-s Nyström approximation of the Hessian. */
    private void buildPreconditioner(Supplier<Var> loss, List<Var> params, SplittableRandom random) {
        int p = CurvatureProducts.size(params);
        int s = Math.min(rank, p);
        double[][] omega = new double[s][p];
        for (int i = 0; i < s; i++) {
            for (int a = 0; a < p; a++) {
                omega[i][a] = random.nextGaussian();
            }
            for (int j = 0; j < i; j++) {
                double dot = CurvatureProducts.dot(omega[i], omega[j]);
                for (int a = 0; a < p; a++) {
                    omega[i][a] -= dot * omega[j][a];
                }
            }
            double norm = Math.sqrt(CurvatureProducts.dot(omega[i], omega[i]));
            for (int a = 0; a < p; a++) {
                omega[i][a] /= norm;
            }
        }
        double[][] y = new double[s][];
        double frob = 0;
        for (int i = 0; i < s; i++) {
            y[i] = CurvatureProducts.hessianVector(loss, params, omega[i]);
            frob += CurvatureProducts.dot(y[i], y[i]);
        }
        double nu = Math.sqrt(p) * 1e-7 * Math.sqrt(frob) + 1e-12;
        for (int i = 0; i < s; i++) {
            for (int a = 0; a < p; a++) {
                y[i][a] += nu * omega[i][a];
            }
        }
        double[][] m = new double[s][s];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j <= i; j++) {
                double v = 0.5 * (CurvatureProducts.dot(omega[i], y[j]) + CurvatureProducts.dot(omega[j], y[i]));
                m[i][j] = v;
                m[j][i] = v;
            }
        }
        double[][] c = cholesky(m);
        if (c == null) {
            u = null;
            return;
        }
        // B = Y_nu C^-T: row i of B^T (a p-vector) solves the triangular system per coordinate
        double[][] bt = new double[s][p];
        for (int a = 0; a < p; a++) {
            for (int i = 0; i < s; i++) {
                double v = y[i][a];
                for (int j = 0; j < i; j++) {
                    v -= c[i][j] * bt[j][a];
                }
                bt[i][a] = v / c[i][i];
            }
        }
        double[][] gram = new double[s][s];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j <= i; j++) {
                gram[i][j] = CurvatureProducts.dot(bt[i], bt[j]);
                gram[j][i] = gram[i][j];
            }
        }
        double[][] vecs = new double[s][s];
        double[] vals = jacobiEigen(gram, vecs);
        u = new double[s][p];
        scale = new double[s];
        double smallest = Double.MAX_VALUE;
        double[] lambda = new double[s];
        for (int k = 0; k < s; k++) {
            double sigma = Math.sqrt(Math.max(vals[k], 1e-300));
            for (int i = 0; i < s; i++) {
                double w = vecs[i][k] / sigma;
                for (int a = 0; a < p; a++) {
                    u[k][a] += bt[i][a] * w;
                }
            }
            lambda[k] = Math.max(0, vals[k] - nu);
            smallest = Math.min(smallest, lambda[k]);
        }
        tail = smallest + damping;
        for (int k = 0; k < s; k++) {
            scale[k] = tail / (lambda[k] + damping);
        }
    }

    /** {@code P^-1 v}. */
    private double[] precondition(double[] v) {
        if (u == null) {
            return v.clone();
        }
        double[] out = v.clone();
        for (int k = 0; k < u.length; k++) {
            double c = CurvatureProducts.dot(u[k], v);
            double w = c * (scale[k] - 1);
            for (int a = 0; a < out.length; a++) {
                out[a] += w * u[k][a];
            }
        }
        return out;
    }

    /** Preconditioned CG for {@code A x = b}, stopping at negative curvature. */
    private double[] pcg(java.util.function.UnaryOperator<double[]> a, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        double[] r = b.clone();
        double[] z = precondition(r);
        double[] p = z.clone();
        double rz = CurvatureProducts.dot(r, z);
        double bNorm = Math.sqrt(CurvatureProducts.dot(b, b));
        for (int it = 0; it < cgIterations; it++) {
            double[] ap = a.apply(p);
            double curv = CurvatureProducts.dot(p, ap);
            if (curv <= 0) {
                return it == 0 ? b.clone() : x;
            }
            double alpha = rz / curv;
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            if (Math.sqrt(CurvatureProducts.dot(r, r)) <= 1e-6 * bNorm) {
                break;
            }
            z = precondition(r);
            double rzNew = CurvatureProducts.dot(r, z);
            double beta = rzNew / rz;
            rz = rzNew;
            for (int i = 0; i < n; i++) {
                p[i] = z[i] + beta * p[i];
            }
        }
        return x;
    }

    /** Lower Cholesky factor, with growing jitter; null if it keeps failing. */
    static double[][] cholesky(double[][] m) {
        int n = m.length;
        double jitter = 0;
        double diag = 0;
        for (int i = 0; i < n; i++) {
            diag = Math.max(diag, Math.abs(m[i][i]));
        }
        for (int attempt = 0; attempt < 8; attempt++) {
            double[][] l = new double[n][n];
            boolean ok = true;
            for (int i = 0; i < n && ok; i++) {
                for (int j = 0; j <= i; j++) {
                    double s = m[i][j] + (i == j ? jitter : 0);
                    for (int k = 0; k < j; k++) {
                        s -= l[i][k] * l[j][k];
                    }
                    if (i == j) {
                        if (!(s > 0)) {
                            ok = false;
                            break;
                        }
                        l[i][i] = Math.sqrt(s);
                    } else {
                        l[i][j] = s / l[j][j];
                    }
                }
            }
            if (ok) {
                return l;
            }
            jitter = jitter == 0 ? 1e-10 * Math.max(diag, 1e-12) : jitter * 100;
        }
        return null;
    }

    /**
     * Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations.
     *
     * @param a the matrix, destroyed
     * @param vectors receives the eigenvectors as columns
     * @return the eigenvalues
     */
    static double[] jacobiEigen(double[][] a, double[][] vectors) {
        int n = a.length;
        for (int i = 0; i < n; i++) {
            java.util.Arrays.fill(vectors[i], 0);
            vectors[i][i] = 1;
        }
        for (int sweep = 0; sweep < 60; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    off += a[i][j] * a[i][j];
                }
            }
            if (off < 1e-22) {
                break;
            }
            for (int pIdx = 0; pIdx < n; pIdx++) {
                for (int q = pIdx + 1; q < n; q++) {
                    if (Math.abs(a[pIdx][q]) < 1e-300) {
                        continue;
                    }
                    double theta = (a[q][q] - a[pIdx][pIdx]) / (2 * a[pIdx][q]);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    if (theta == 0) {
                        t = 1;
                    }
                    double c = 1 / Math.sqrt(t * t + 1);
                    double s = t * c;
                    for (int k = 0; k < n; k++) {
                        double akp = a[k][pIdx];
                        double akq = a[k][q];
                        a[k][pIdx] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        double apk = a[pIdx][k];
                        double aqk = a[q][k];
                        a[pIdx][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = vectors[k][pIdx];
                        double vkq = vectors[k][q];
                        vectors[k][pIdx] = c * vkp - s * vkq;
                        vectors[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            vals[i] = a[i][i];
        }
        return vals;
    }

    @Override
    public String toString() {
        return "NystromNewtonCg(rank " + rank + ", damping " + damping + ")";
    }
}
