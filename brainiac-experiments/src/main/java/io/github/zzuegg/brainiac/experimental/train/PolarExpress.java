package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Polar Express (Amsel, Persson, Musco and Gower 2025, "The Polar Express: Optimal Matrix
 * Sign Methods and Their Application to the Muon Algorithm",
 * <a href="https://arxiv.org/abs/2505.16932">arXiv:2505.16932</a>, ICLR 2026): a drop-in
 * replacement for Muon's Newton-Schulz iteration whose odd quintic
 * {@code p_t(x) = a_t x + b_t x^3 + c_t x^5} changes from iteration to iteration.
 *
 * <p>Each {@code p_t} is the minimax-optimal odd quintic on the interval {@code [l_t, u_t]}
 * that the singular values are known to lie in: it minimises {@code max |1 - p(x)|} there.
 * The next interval is then {@code [p_t(l_t), 2 - p_t(l_t)]}, so the iteration is optimal in a greedy sense and converges to the exact polar factor
 * (super-exponentially once the interval is small), while Muon's fixed coefficients
 * stall with singular values spread over roughly {@code [0.7, 1.2]}. Early iterations
 * have a very steep slope at 0 (about 8 for {@code l = 1e-3}), so small singular values
 * grow faster than with any fixed polynomial.
 *
 * <p>This class computes the coefficients itself with the Remez exchange algorithm (the
 * Chebyshev alternation theorem holds for {@code {x, x^3, x^5}} on positive intervals, so
 * the optimum equioscillates at the two interval ends and the two positive critical points
 * of {@code p}), for any lower bound and iteration count. Like the paper, it divides
 * {@code a, b, c} of every iteration except the last by {@code 1.01, 1.01^3, 1.01^5} for
 * float32 rounding headroom and normalises the input by {@code 1.01} times its Frobenius
 * norm.
 */
public final class PolarExpress implements Orthogonalizer {

    /**
     * The paper's cushion: each polynomial is fitted on {@code [max(l, 0.024 u), u]} and
     * then re-centred on {@code [l, u]}, so that tiny singular values are not mapped to
     * nearly 0 where the polynomial is flat.
     */
    public static final double CUSHION = 0.02407327424182761;

    private final double[][] coefficients;
    private final float safety;

    /**
     * @param coefficients {@code [steps][3]} the {@code (a, b, c)} of each iteration
     * @param safety the factor the Frobenius norm is multiplied by before dividing, 1.01
     */
    public PolarExpress(double[][] coefficients, float safety) {
        if (coefficients.length == 0 || safety < 1) {
            throw new IllegalArgumentException("need at least one iteration and safety >= 1");
        }
        this.coefficients = new double[coefficients.length][];
        for (int i = 0; i < coefficients.length; i++) {
            if (coefficients[i].length != 3) {
                throw new IllegalArgumentException("three coefficients per iteration");
            }
            this.coefficients[i] = coefficients[i].clone();
        }
        this.safety = safety;
    }

    /**
     * The paper's settings: lower bound {@code 1e-3}, safety factor 1.01.
     *
     * @param steps iterations
     * @return the orthogonaliser
     */
    public static PolarExpress of(int steps) {
        return of(steps, 1e-3, 1.01);
    }

    /**
     * @param steps iterations
     * @param lower the assumed smallest singular value after normalisation, for example 1e-3
     * @param safety the safety factor, for example 1.01; 1 for none
     * @return the orthogonaliser with coefficients computed for these settings
     */
    public static PolarExpress of(int steps, double lower, double safety) {
        double[][] c = coefficients(steps, lower);
        for (int t = 0; t < c.length - 1; t++) {
            c[t][0] /= safety;
            c[t][1] /= Math.pow(safety, 3);
            c[t][2] /= Math.pow(safety, 5);
        }
        return new PolarExpress(c, (float) safety);
    }

    /**
     * The greedy minimax-optimal coefficients, without the safety factor.
     *
     * @param steps iterations
     * @param lower the lower end of the first interval, the upper end is 1
     * @return {@code [steps][3]} the {@code (a, b, c)} of each iteration
     */
    public static double[][] coefficients(int steps, double lower) {
        if (steps < 1 || !(lower > 0) || lower >= 1) {
            throw new IllegalArgumentException("need steps >= 1 and 0 < lower < 1");
        }
        double[][] out = new double[steps][];
        double l = lower;
        double u = 1;
        for (int t = 0; t < steps; t++) {
            double[] abcE = minimaxQuintic(Math.max(l, CUSHION * u), u);
            double a = abcE[0];
            double b = abcE[1];
            double c = abcE[2];
            // re-centre around 1 on the true interval [l, u]: 1 - k p(l) = k p(u) - 1
            double k = 2 / (poly(a, b, c, l) + poly(a, b, c, u));
            out[t] = new double[] {a * k, b * k, c * k};
            l = poly(a * k, b * k, c * k, l);
            u = 2 - l;
        }
        return out;
    }

    /**
     * The odd quintic {@code a x + b x^3 + c x^5} that minimises {@code max |1 - p(x)|} on
     * {@code [l, u]}, by Remez exchange.
     *
     * @param l the lower end, positive
     * @param u the upper end
     * @return {@code {a, b, c, E}} with {@code E} the minimax error
     */
    public static double[] minimaxQuintic(double l, double u) {
        if (u - l < 1e-4) {   // a point: the classic contraction toward 1
            return new double[] {15 / 8.0, -10 / 8.0, 3 / 8.0, 0};
        }
        double[] x = new double[4];
        for (int i = 0; i < 4; i++) {   // Chebyshev-like start
            x[i] = l + (u - l) * (1 - Math.cos(Math.PI * i / 3)) / 2;
        }
        double[] sol = null;
        for (int iter = 0; iter < 100; iter++) {
            double[][] m = new double[4][4];
            double[] rhs = new double[4];
            for (int i = 0; i < 4; i++) {
                m[i][0] = x[i];
                m[i][1] = x[i] * x[i] * x[i];
                m[i][2] = m[i][1] * x[i] * x[i];
                m[i][3] = i % 2 == 0 ? 1 : -1;
                rhs[i] = 1;
            }
            sol = solve(m, rhs);
            double a = sol[0];
            double b = sol[1];
            double c = sol[2];
            // the interior extrema of 1 - p: roots of p'(x) = a + 3b x^2 + 5c x^4 in x^2
            double[] next = {l, Double.NaN, Double.NaN, u};
            double disc = 9 * b * b - 20 * a * c;
            if (disc >= 0 && c != 0) {
                double r1 = (-3 * b - Math.sqrt(disc)) / (10 * c);
                double r2 = (-3 * b + Math.sqrt(disc)) / (10 * c);
                double lo = Math.min(r1, r2);
                double hi = Math.max(r1, r2);
                if (lo > 0) {
                    next[1] = Math.sqrt(lo);
                }
                if (hi > 0) {
                    next[2] = Math.sqrt(hi);
                }
            }
            if (!(next[1] > l && next[1] < u && next[2] > next[1] && next[2] < u)) {
                break;   // no clean exchange: keep the current reference
            }
            double change = 0;
            for (int i = 0; i < 4; i++) {
                change = Math.max(change, Math.abs(next[i] - x[i]));
            }
            x = next;
            if (change < 1e-15 * u) {
                break;
            }
        }
        double[] m = new double[4];
        for (int i = 0; i < 4; i++) {
            m[i] = sol[i];
        }
        m[3] = Math.abs(maxError(sol[0], sol[1], sol[2], l, u));
        return m;
    }

    private static double poly(double a, double b, double c, double x) {
        return a * x + b * x * x * x + c * x * x * x * x * x;
    }

    private static double maxError(double a, double b, double c, double l, double u) {
        double worst = 0;
        for (int i = 0; i <= 4000; i++) {
            double x = l + (u - l) * i / 4000.0;
            worst = Math.max(worst, Math.abs(1 - (a * x + b * x * x * x + c * x * x * x * x * x)));
        }
        return worst;
    }

    private static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][];
        for (int i = 0; i < n; i++) {
            m[i] = new double[n + 1];
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++) {
                piv = Math.abs(m[r][col]) > Math.abs(m[piv][col]) ? r : piv;
            }
            double[] tmp = m[col];
            m[col] = m[piv];
            m[piv] = tmp;
            for (int r = 0; r < n; r++) {
                if (r != col) {
                    double f = m[r][col] / m[col][col];
                    for (int k = col; k <= n; k++) {
                        m[r][k] -= f * m[col][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = m[i][n] / m[i][i];
        }
        return x;
    }

    /** @return the iterations */
    public int steps() {
        return coefficients.length;
    }

    /**
     * @return {@code [steps][3]} the coefficients used, safety factor included; a copy
     */
    public double[][] coefficients() {
        double[][] c = new double[coefficients.length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = coefficients[i].clone();
        }
        return c;
    }

    @Override
    public Tensor orthogonalize(Tensor g) {
        boolean tall = g.shape().dim(0) > g.shape().dim(1);
        Tensor x = Orthogonalizer.normalize(tall ? g.transpose() : g, safety);
        for (double[] abc : coefficients) {
            Tensor a = x.matmul(x.transpose());
            Tensor b = a.mul((float) abc[1]).add(a.matmul(a).mul((float) abc[2]));
            x = x.mul((float) abc[0]).add(b.matmul(x));
        }
        return tall ? x.transpose() : x;
    }

    @Override
    public String toString() {
        return "PolarExpress(" + coefficients.length + " steps)";
    }
}
