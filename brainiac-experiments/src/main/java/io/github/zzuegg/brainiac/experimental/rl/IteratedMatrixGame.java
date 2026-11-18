package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * An infinitely iterated two-action matrix game between memory-one policies, with an exact,
 * differentiable discounted value. Each player's policy is five logits: the probability of
 * cooperating (action 0) in the first round and after each of the four previous joint
 * outcomes CC, CD, DC, DD (own action first). The two policies define a Markov chain over the
 * joint outcomes, and the normalised discounted value
 * {@code (1 - gamma) sum_t gamma^t E[r_t]} is computed in closed form as
 * {@code (1 - gamma) d0 (I - gamma P)^-1 r}, the inverse being summed by repeated squaring of
 * {@code gamma P} so that every operation is an autograd {@link Var} op; gradients and the
 * second-order gradients opponent shaping needs come from {@link Var#gradients}.
 *
 * <p>Foerster et al. 2018, "Learning with Opponent-Learning Awareness",
 * <a href="https://arxiv.org/abs/1709.04326">arXiv:1709.04326</a>, use exactly this setting
 * (the iterated prisoner's dilemma with {@code gamma = 0.96}).
 */
public final class IteratedMatrixGame {

    private final String name;
    private final double[] reward1;
    private final double[] reward2;
    private final double gamma;

    private IteratedMatrixGame(String name, double[] reward1, double[] reward2, double gamma) {
        this.name = name;
        this.reward1 = reward1.clone();
        this.reward2 = reward2.clone();
        this.gamma = gamma;
    }

    /**
     * @param name a label
     * @param reward1 player 1's rewards for CC, CD, DC, DD (player 1's action first)
     * @param reward2 player 2's rewards for the same joint outcomes
     * @param gamma the discount
     * @return the game
     */
    public static IteratedMatrixGame of(String name, double[] reward1, double[] reward2, double gamma) {
        return new IteratedMatrixGame(name, reward1, reward2, gamma);
    }

    /**
     * The iterated prisoner's dilemma: mutual cooperation -1 each, mutual defection -2 each,
     * the sucker -3 and the defector 0.
     *
     * @param gamma the discount, 0.96 in the LOLA paper
     * @return the game
     */
    public static IteratedMatrixGame prisonersDilemma(double gamma) {
        return of("iterated prisoner's dilemma", new double[] {-1, -3, 0, -2}, new double[] {-1, 0, -3, -2}, gamma);
    }

    /**
     * Iterated matching pennies: player 1 wins 1 when the actions match.
     *
     * @param gamma the discount
     * @return the game
     */
    public static IteratedMatrixGame matchingPennies(double gamma) {
        return of("iterated matching pennies", new double[] {1, -1, -1, 1}, new double[] {-1, 1, 1, -1}, gamma);
    }

    /**
     * The iterated stag hunt: hunting the stag together (action 0) pays 4, a hare 3 whatever
     * the other does, and hunting the stag alone 0.
     *
     * @param gamma the discount
     * @return the game
     */
    public static IteratedMatrixGame stagHunt(double gamma) {
        return of("iterated stag hunt", new double[] {4, 0, 3, 3}, new double[] {4, 3, 0, 3}, gamma);
    }

    /** @return the discount */
    public double gamma() {
        return gamma;
    }

    /**
     * The normalised discounted values of both players, differentiable in both policies.
     *
     * @param theta1 player 1's logits, shape {@code [1, 5]}
     * @param theta2 player 2's logits, shape {@code [1, 5]}
     * @return {@code {V1, V2}}, scalars
     */
    public Var[] values(Var theta1, Var theta2) {
        Var p = theta1.sigmoid();
        Var q = theta2.sigmoid();
        // player 2's view of CD is player 1's DC
        Var qv = Var.concat(1, q.slice(1, 1, 2), q.slice(1, 3, 4), q.slice(1, 2, 3), q.slice(1, 4, 5)).transpose();
        Var pv = p.slice(1, 1, 5).transpose();
        Var pn = pv.neg().add(1f);
        Var qn = qv.neg().add(1f);
        Var transition = Var.concat(1, pv.mul(qv), pv.mul(qn), pn.mul(qv), pn.mul(qn)).mul((float) gamma);
        Var p0 = p.slice(1, 0, 1);
        Var q0 = q.slice(1, 0, 1);
        Var start = Var.concat(1, p0.mul(q0), p0.mul(q0.neg().add(1f)), p0.neg().add(1f).mul(q0),
                p0.neg().add(1f).mul(q0.neg().add(1f)));
        // sum_t (gamma P)^t by repeated squaring: S <- S + A S, A <- A A
        Var sum = Var.of(identity());
        Var power = transition;
        for (int k = 0; k < 9; k++) {
            sum = sum.add(power.matmul(sum));
            power = power.matmul(power);
        }
        Var occupancy = start.matmul(sum).mul((float) (1 - gamma));
        Var v1 = occupancy.matmul(Var.of(column(reward1))).sum();
        Var v2 = occupancy.matmul(Var.of(column(reward2))).sum();
        return new Var[] {v1, v2};
    }

    /**
     * The same values in double precision without autograd (a 4x4 linear solve), for
     * black-box use such as evolution strategies.
     *
     * @param theta1 player 1's logits
     * @param theta2 player 2's logits
     * @return {@code {V1, V2}}
     */
    public double[] values(double[] theta1, double[] theta2) {
        double[] p = cooperation(theta1);
        double[] q = cooperation(theta2);
        double[] qv = {q[1], q[3], q[2], q[4]};
        double[][] a = new double[4][4];
        for (int s = 0; s < 4; s++) {
            double ps = p[s + 1];
            double qs = qv[s];
            double[] next = {ps * qs, ps * (1 - qs), (1 - ps) * qs, (1 - ps) * (1 - qs)};
            for (int j = 0; j < 4; j++) {
                a[j][s] = (j == s ? 1 : 0) - gamma * next[j];
            }
        }
        double[] d0 = {p[0] * q[0], p[0] * (1 - q[0]), (1 - p[0]) * q[0], (1 - p[0]) * (1 - q[0])};
        // occupancy x solves (I - gamma P)^T x = d0
        double[] x = solve(a, d0);
        double v1 = 0;
        double v2 = 0;
        for (int s = 0; s < 4; s++) {
            v1 += (1 - gamma) * x[s] * reward1[s];
            v2 += (1 - gamma) * x[s] * reward2[s];
        }
        return new double[] {v1, v2};
    }

    /**
     * The gradient of one player's value with respect to one player's logits, by central
     * differences on {@link #values(double[], double[])}.
     *
     * @param theta1 player 1's logits
     * @param theta2 player 2's logits
     * @param valueOf whose value, 0 or 1
     * @param withRespectTo whose logits, 0 or 1
     * @return the gradient, five values
     */
    public double[] gradient(double[] theta1, double[] theta2, int valueOf, int withRespectTo) {
        double[] g = new double[5];
        double h = 1e-5;
        for (int i = 0; i < 5; i++) {
            double[] a = (withRespectTo == 0 ? theta1 : theta2).clone();
            double[] b = a.clone();
            a[i] += h;
            b[i] -= h;
            double up = withRespectTo == 0 ? values(a, theta2)[valueOf] : values(theta1, a)[valueOf];
            double down = withRespectTo == 0 ? values(b, theta2)[valueOf] : values(theta1, b)[valueOf];
            g[i] = (up - down) / (2 * h);
        }
        return g;
    }

    private static double[] solve(double[][] m, double[] rhs) {
        int n = rhs.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n] = rhs[i];
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) {
                    piv = r;
                }
            }
            double[] t = a[c];
            a[c] = a[piv];
            a[piv] = t;
            for (int r = 0; r < n; r++) {
                if (r != c) {
                    double f = a[r][c] / a[c][c];
                    for (int k = c; k <= n; k++) {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = a[i][n] / a[i][i];
        }
        return x;
    }

    /**
     * @param logits five logits
     * @return the five cooperation probabilities
     */
    public static double[] cooperation(double[] logits) {
        double[] p = new double[logits.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = 1 / (1 + Math.exp(-logits[i]));
        }
        return p;
    }

    /**
     * @param values a row
     * @return a {@code [1, n]} tensor
     */
    public static Tensor row(double[] values) {
        float[] f = new float[values.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) values[i];
        }
        return Tensor.of(Shape.of(1, f.length), f);
    }

    private static Tensor column(double[] values) {
        return row(values).reshape(values.length, 1);
    }

    private static Tensor identity() {
        float[] f = new float[16];
        for (int i = 0; i < 4; i++) {
            f[5 * i] = 1;
        }
        return Tensor.of(Shape.of(4, 4), f);
    }

    @Override
    public String toString() {
        return name;
    }
}
