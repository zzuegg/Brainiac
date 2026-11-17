package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 * Activations that recent models use but the core lacks, as ordinary {@link Activation}s
 * (drop-ins for {@code Activation.RELU} and friends):
 *
 * <ul>
 *   <li>{@link #SQUARED_RELU}: {@code relu(x)^2}, found by architecture search in Primer and
 *       used in many speed-run LLMs; sparse like ReLU but smooth at the kink's right side.
 *       So et al., <a href="https://arxiv.org/abs/2109.08668">arXiv:2109.08668</a>.</li>
 *   <li>{@link #snake()}: {@code x + sin^2(a x) / a} with a learnable frequency {@code a}
 *       per unit: monotone, yet periodic, so networks extrapolate periodic signals (and it is
 *       the activation of BigVGAN and other audio vocoders). Ziyin et al.,
 *       <a href="https://arxiv.org/abs/2006.08195">arXiv:2006.08195</a>.</li>
 *   <li>{@link #rational()}: the Padé activation unit, a learnable ratio of polynomials
 *       {@code P(x) / (1 + |b1 x| + ... + |bn x^n|)} that starts as a fit of leaky ReLU and is
 *       then shaped by training. Molina et al.,
 *       <a href="https://arxiv.org/abs/1907.06732">arXiv:1907.06732</a>.</li>
 * </ul>
 */
public final class ModernActivations {

    /** {@code relu(x)^2} (Primer). */
    public static final Activation SQUARED_RELU = Activation.named("squared_relu", Init.HE, x -> x.relu().square());

    private ModernActivations() {
    }

    /**
     * Snake with a learnable frequency per unit that starts at 1.
     *
     * @return the activation template
     */
    public static Activation snake() {
        return snake(1f);
    }

    /**
     * Snake with a learnable frequency per unit.
     *
     * @param initialAlpha the starting frequency, positive; larger means more wiggles
     * @return the activation template
     */
    public static Activation snake(float initialAlpha) {
        if (!(initialAlpha > 0)) {
            throw new IllegalArgumentException("alpha must be positive, got " + initialAlpha);
        }
        return Activation.withParameters("snake", Init.HE,
                width -> List.of(Tensor.full(Shape.of(width), initialAlpha)),
                (x, p) -> snake(x, p.get(0)));
    }

    /**
     * The Padé activation unit with numerator degree 5 and denominator degree 4, initialized
     * to approximate leaky ReLU (slope 0.01) on {@code [-3, 3]}; its 10 coefficients are
     * shared by the whole layer, as in the paper.
     *
     * @return the activation template
     */
    public static Activation rational() {
        return rational(x -> x >= 0 ? x : 0.01 * x, 5, 4);
    }

    /**
     * A Padé activation unit initialized to approximate {@code target} on {@code [-3, 3]}.
     *
     * @param target the function to start from
     * @param numerator the numerator degree
     * @param denominator the denominator degree
     * @return the activation template
     */
    public static Activation rational(DoubleUnaryOperator target, int numerator, int denominator) {
        double[][] init = fitRational(target, numerator, denominator);
        float[] a = toFloats(init[0]);
        float[] b = toFloats(init[1]);
        return Activation.withParameters("rational", Init.HE,
                width -> List.of(Tensor.of(Shape.of(a.length), a), Tensor.of(Shape.of(b.length), b)),
                (x, p) -> rationalForward(x, p.get(0), p.get(1)));
    }

    /**
     * Evaluates {@code P(x) / (1 + sum_k |b_k| |x|^k)}.
     *
     * @param x the input
     * @param a the numerator coefficients {@code a0..am}
     * @param b the denominator coefficients {@code b1..bn}
     * @return the activation
     */
    public static Var rationalForward(Var x, Var a, Var b) {
        int m = a.shape().dim(0);
        int n = b.shape().dim(0);
        Var power = null;
        Var num = a.slice(0, 0, 1).expand(x.shape());
        for (int k = 1; k < m; k++) {
            power = power == null ? x : power.mul(x);
            num = num.add(power.mul(a.slice(0, k, k + 1)));
        }
        Var ax = x.abs();
        Var apower = null;
        Var den = null;
        for (int k = 1; k <= n; k++) {
            apower = apower == null ? ax : apower.mul(ax);
            Var term = apower.mul(b.slice(0, k - 1, k).abs());
            den = den == null ? term : den.add(term);
        }
        return den == null ? num : num.div(den.add(1f));
    }

    /**
     * Fits rational coefficients to a function by linearized least squares:
     * {@code P(x) - f(x) * sum_k b_k |x|^k = f(x)} on a grid over {@code [-3, 3]}.
     *
     * @return {@code {a, b}}
     */
    static double[][] fitRational(DoubleUnaryOperator f, int m, int n) {
        int unknowns = m + 1 + n;
        double[][] ata = new double[unknowns][unknowns];
        double[] atb = new double[unknowns];
        double[] row = new double[unknowns];
        for (int i = 0; i <= 600; i++) {
            double x = -3 + 6.0 * i / 600;
            double y = f.applyAsDouble(x);
            for (int k = 0; k <= m; k++) {
                row[k] = Math.pow(x, k);
            }
            for (int k = 1; k <= n; k++) {
                row[m + k] = -y * Math.pow(Math.abs(x), k);
            }
            for (int r = 0; r < unknowns; r++) {
                atb[r] += row[r] * y;
                for (int c = 0; c < unknowns; c++) {
                    ata[r][c] += row[r] * row[c];
                }
            }
        }
        for (int r = 0; r < unknowns; r++) {
            ata[r][r] += 1e-6;
        }
        double[] sol = QuantMath.solve(ata, atb);
        double[] a = new double[m + 1];
        double[] b = new double[n];
        System.arraycopy(sol, 0, a, 0, m + 1);
        System.arraycopy(sol, m + 1, b, 0, n);
        return new double[][] {a, b};
    }

    private static float[] toFloats(double[] d) {
        float[] f = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }

    /**
     * Snake on autograd's sine: {@code x + sin^2(a x) / a}.
     *
     * @param x the input {@code [..., width]}
     * @param alpha the frequencies {@code [width]}
     * @return the activation
     */
    static Var snake(Var x, Var alpha) {
        return x.add(x.mul(alpha).sin().square().div(alpha));
    }
}
