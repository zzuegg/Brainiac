package io.github.zzuegg.brainiac.experimental.logic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * ANFIS, the adaptive-network-based fuzzy inference system (Jang, IEEE SMC 1993): a first-order
 * Takagi–Sugeno {@link FuzzyInference} written as a five-layer network (memberships, product
 * firing, normalisation, linear consequents, sum) and trained by the hybrid rule. Each epoch the
 * consequents are solved by least squares with the premises fixed (the forward pass), then the
 * membership centres and widths take gradient steps with the consequents fixed (the backward
 * pass, through autograd, Adam). With {@code hybrid = false} every parameter is learned by
 * gradient descent, the baseline that the least-squares step is meant to beat in epochs.
 */
public final class Anfis {

    private final FuzzyInference fis;

    /**
     * Wraps a Takagi–Sugeno system with Gaussian or bell memberships.
     *
     * @param fis the system to train (modified in place)
     */
    public Anfis(FuzzyInference fis) {
        if (fis.type != FuzzyInference.Type.TAKAGI_SUGENO || fis.membership == FuzzyInference.Membership.TRIANGULAR) {
            throw new IllegalArgumentException("ANFIS needs a Takagi-Sugeno system with smooth memberships");
        }
        this.fis = fis;
    }

    /** @return the trained system */
    public FuzzyInference system() {
        return fis;
    }

    /**
     * Training history.
     *
     * @param trainRmse per epoch the training RMSE after the epoch
     * @param millis wall time
     */
    public record History(double[] trainRmse, double millis) {
    }

    private static Tensor matrix(double[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] v = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) rows[i][j];
            }
        }
        return Tensor.of(Shape.of(n, d), v);
    }

    /**
     * Trains the system.
     *
     * @param x inputs
     * @param y targets
     * @param epochs passes (one least-squares solve and {@code gdSteps} gradient steps each)
     * @param gdSteps gradient steps per epoch
     * @param lr Adam's learning rate for the premises (and consequents when not hybrid)
     * @param ridge the ridge penalty of the least-squares step
     * @param hybrid whether consequents come from least squares
     * @return the history
     */
    public History train(double[][] x, double[] y, int epochs, int gdSteps, double lr, double ridge, boolean hybrid) {
        long t0 = System.nanoTime();
        int n = x.length;
        int d = fis.inputs;
        int r = fis.rules();
        double[][] xb = new double[n][d + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x[i], 0, xb[i], 0, d);
            xb[i][d] = 1;
        }
        Var xs = Var.of(matrix(x));
        Var x1 = Var.of(matrix(xb));
        float[] yv = new float[n];
        for (int i = 0; i < n; i++) {
            yv[i] = (float) y[i];
        }
        Tensor target = Tensor.of(Shape.of(n), yv);
        Var centres = Var.param(matrix(fis.center));
        // widths are learned in log space to stay positive
        double[][] logW = new double[r][d];
        for (int k = 0; k < r; k++) {
            for (int i = 0; i < d; i++) {
                logW[k][i] = Math.log(fis.width[k][i]);
            }
        }
        Var logWidths = Var.param(matrix(logW));
        Var cons = Var.param(matrix(fis.consequent));
        List<Var> params = new ArrayList<>(List.of(centres, logWidths));
        if (!hybrid) {
            params.add(cons);
        }
        Optimizer opt = Optimizer.adam((float) lr);
        double[] hist = new double[epochs];
        for (int e = 0; e < epochs; e++) {
            if (hybrid) {
                pull(centres, logWidths, null);
                fis.fitConsequents(x, y, ridge);
                cons.assign(matrix(fis.consequent));
            }
            for (int s = 0; s < gdSteps; s++) {
                Var pred = forward(xs, x1, centres, logWidths, cons, n, d, r);
                pred.sub(target).square().mean().backward();
                opt.step(params);
                if (hybrid) {
                    cons.zeroGrad();
                }
            }
            pull(centres, logWidths, hybrid ? null : cons);
            if (hybrid) {
                fis.fitConsequents(x, y, ridge);
                cons.assign(matrix(fis.consequent));
            }
            double se = 0;
            for (int i = 0; i < n; i++) {
                double err = fis.predict(x[i]) - y[i];
                se += err * err;
            }
            hist[e] = Math.sqrt(se / n);
        }
        return new History(hist, (System.nanoTime() - t0) / 1e6);
    }

    private void pull(Var centres, Var logWidths, Var cons) {
        float[] c = centres.value().toArray();
        float[] w = logWidths.value().toArray();
        int d = fis.inputs;
        for (int k = 0; k < fis.rules(); k++) {
            for (int i = 0; i < d; i++) {
                fis.center[k][i] = c[k * d + i];
                fis.width[k][i] = Math.max(1e-4, Math.exp(w[k * d + i]));
            }
        }
        if (cons != null) {
            float[] p = cons.value().toArray();
            for (int k = 0; k < fis.rules(); k++) {
                for (int i = 0; i <= d; i++) {
                    fis.consequent[k][i] = p[k * (d + 1) + i];
                }
            }
        }
    }

    private Var forward(Var xs, Var x1, Var centres, Var logWidths, Var cons, int n, int d, int r) {
        Shape s3 = Shape.of(n, r, d);
        Var diff = xs.reshape(n, 1, d).expand(s3).sub(centres.reshape(1, r, d).expand(s3));
        Var z2 = diff.div(logWidths.exp().reshape(1, r, d).expand(s3)).square();
        Var logMu = fis.membership == FuzzyInference.Membership.GAUSSIAN ? z2.mul(-0.5f)
            : z2.square().add(1f).log().neg();
        Var logFire = logMu.sum(2, false);                                  // [n, r]
        Var norm = logFire.sub(logFire.logsumexp(1, true)).exp();           // normalised firing
        Var f = x1.matmul(cons.permute(1, 0));                              // [n, r]
        return norm.mul(f).sum(1, false);
    }
}
