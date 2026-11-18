package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.experimental.geometry.GeometryLinalg;

/**
 * Numerical inverse kinematics for planar serial arms with joint limits:
 *
 * <ul>
 *   <li>{@link Method#PSEUDO_INVERSE}: {@code dq = J^+ e} (Whitney 1969); fast, but steps
 *       blow up near singular configurations (a stretched or folded arm);</li>
 *   <li>{@link Method#DAMPED_LEAST_SQUARES}: {@code dq = J^T (J J^T + lambda^2 I)^-1 e}
 *       (Wampler 1986; Nakamura &amp; Hanafusa 1986), a Levenberg-Marquardt step that trades a
 *       little accuracy for stability near singularities;</li>
 *   <li>{@link Method#CCD}: cyclic coordinate descent, rotating one joint at a time (from the
 *       tip) to point the end effector at the target;</li>
 *   <li>{@link Method#FABRIK}: forward and backward reaching on joint positions with no
 *       matrices (Aristidou &amp; Lasenby, Graphical Models 2011), converted back to angles.</li>
 * </ul>
 *
 * Joint angles are relative; joint limits are enforced by clamping after every update.
 */
public final class InverseKinematics {

    /** The solver. */
    public enum Method {
        /** Jacobian pseudo-inverse. */
        PSEUDO_INVERSE,
        /** Damped least squares. */
        DAMPED_LEAST_SQUARES,
        /** Cyclic coordinate descent. */
        CCD,
        /** Forward and backward reaching. */
        FABRIK
    }

    /**
     * A solution.
     *
     * @param angles the joint angles
     * @param error the final end-effector distance to the target
     * @param iterations the iterations used
     */
    public record Solution(double[] angles, double error, int iterations) {
    }

    private final double[] lengths;
    private final double[] lower;
    private final double[] upper;
    private double maxStep = Double.POSITIVE_INFINITY;

    /**
     * Creates the solver for an arm based at the origin.
     *
     * @param lengths the link lengths
     * @param lower per-joint lower limits
     * @param upper per-joint upper limits
     */
    public InverseKinematics(double[] lengths, double[] lower, double[] upper) {
        this.lengths = lengths.clone();
        this.lower = lower.clone();
        this.upper = upper.clone();
    }

    /**
     * Clamps the norm of every Jacobian step (Buss 2004), which keeps the pseudo-inverse and
     * damped least squares from overshooting into joint limits.
     *
     * @param radians the largest joint-space step
     * @return this solver
     */
    public InverseKinematics maxStep(double radians) {
        this.maxStep = radians;
        return this;
    }

    /** @return the number of joints */
    public int joints() {
        return lengths.length;
    }

    /** @return the link lengths */
    public double[] lengths() {
        return lengths.clone();
    }

    /** @return the lower joint limits */
    public double[] lower() {
        return lower.clone();
    }

    /** @return the upper joint limits */
    public double[] upper() {
        return upper.clone();
    }

    /**
     * Forward kinematics.
     *
     * @param q joint angles
     * @return the joint positions {@code [n + 1][2]}, base first, end effector last
     */
    public double[][] positions(double[] q) {
        double[][] p = new double[lengths.length + 1][2];
        double a = 0;
        for (int i = 0; i < lengths.length; i++) {
            a += q[i];
            p[i + 1][0] = p[i][0] + lengths[i] * Math.cos(a);
            p[i + 1][1] = p[i][1] + lengths[i] * Math.sin(a);
        }
        return p;
    }

    /**
     * @param q joint angles
     * @return the end-effector position
     */
    public double[] endEffector(double[] q) {
        return positions(q)[lengths.length];
    }

    /**
     * @param q joint angles
     * @return the {@code 2 x n} position Jacobian
     */
    public double[][] jacobian(double[] q) {
        double[][] p = positions(q);
        int n = lengths.length;
        double[] e = p[n];
        double[][] j = new double[2][n];
        for (int i = 0; i < n; i++) {
            j[0][i] = -(e[1] - p[i][1]);
            j[1][i] = e[0] - p[i][0];
        }
        return j;
    }

    private void clamp(double[] q) {
        for (int i = 0; i < q.length; i++) {
            q[i] = Math.max(lower[i], Math.min(upper[i], q[i]));
        }
    }

    /**
     * Solves for a target.
     *
     * @param method the solver
     * @param target the target position
     * @param initial the initial joint angles
     * @param maxIterations the iteration budget
     * @param tolerance the success distance
     * @param damping {@code lambda} for damped least squares
     * @return the solution
     */
    public Solution solve(Method method, double[] target, double[] initial, int maxIterations, double tolerance,
            double damping) {
        double[] q = initial.clone();
        clamp(q);
        int it = 0;
        double err = error(q, target);
        while (it < maxIterations && err > tolerance && Double.isFinite(err)) {
            switch (method) {
                case PSEUDO_INVERSE -> jacobianStep(q, target, 0);
                case DAMPED_LEAST_SQUARES -> jacobianStep(q, target, damping);
                case CCD -> ccdSweep(q, target);
                case FABRIK -> fabrikSweep(q, target);
                default -> throw new IllegalStateException();
            }
            clamp(q);
            err = error(q, target);
            it++;
        }
        return new Solution(q, err, it);
    }

    /**
     * One damped least-squares refinement step, as used after a learned proposal.
     *
     * @param q the joint angles (updated in place)
     * @param target the target
     * @param damping {@code lambda}
     */
    public void refine(double[] q, double[] target, double damping) {
        jacobianStep(q, target, damping);
        clamp(q);
    }

    /**
     * @param q joint angles
     * @param target a target
     * @return the end-effector distance to the target
     */
    public double error(double[] q, double[] target) {
        double[] e = endEffector(q);
        return Math.hypot(e[0] - target[0], e[1] - target[1]);
    }

    private void jacobianStep(double[] q, double[] target, double damping) {
        double[] e = endEffector(q);
        double[] err = {target[0] - e[0], target[1] - e[1]};
        double[][] j = jacobian(q);
        double[] dq;
        if (damping == 0) {
            double[][] pinv = GeometryLinalg.pseudoInverse(j, 1e-12);
            dq = GeometryLinalg.multiply(pinv, err);
        } else {
            double[][] jjt = GeometryLinalg.multiply(j, GeometryLinalg.transpose(j));
            jjt[0][0] += damping * damping;
            jjt[1][1] += damping * damping;
            double[] y = GeometryLinalg.solve(jjt, err);
            dq = GeometryLinalg.multiply(GeometryLinalg.transpose(j), y == null ? new double[2] : y);
        }
        double norm = 0;
        for (double v : dq) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        double scale = norm > maxStep ? maxStep / norm : 1;
        for (int i = 0; i < q.length; i++) {
            q[i] += scale * dq[i];
        }
    }

    private void ccdSweep(double[] q, double[] target) {
        for (int i = q.length - 1; i >= 0; i--) {
            double[][] p = positions(q);
            double[] e = p[q.length];
            double a1 = Math.atan2(e[1] - p[i][1], e[0] - p[i][0]);
            double a2 = Math.atan2(target[1] - p[i][1], target[0] - p[i][0]);
            q[i] += GeometryLinalg.wrap(a2 - a1);
            q[i] = Math.max(lower[i], Math.min(upper[i], q[i]));
        }
    }

    private void fabrikSweep(double[] q, double[] target) {
        int n = lengths.length;
        double[][] p = positions(q);
        double reach = 0;
        for (double l : lengths) {
            reach += l;
        }
        if (Math.hypot(target[0], target[1]) >= reach) {
            for (int i = 0; i < n; i++) {
                double d = Math.hypot(target[0] - p[i][0], target[1] - p[i][1]);
                double t = lengths[i] / Math.max(d, 1e-12);
                p[i + 1][0] = (1 - t) * p[i][0] + t * target[0];
                p[i + 1][1] = (1 - t) * p[i][1] + t * target[1];
            }
        } else {
            p[n][0] = target[0];
            p[n][1] = target[1];
            for (int i = n - 1; i >= 0; i--) {
                double d = Math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]);
                double t = lengths[i] / Math.max(d, 1e-12);
                p[i][0] = (1 - t) * p[i + 1][0] + t * p[i][0];
                p[i][1] = (1 - t) * p[i + 1][1] + t * p[i][1];
            }
            p[0][0] = 0;
            p[0][1] = 0;
            for (int i = 0; i < n; i++) {
                double d = Math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]);
                double t = lengths[i] / Math.max(d, 1e-12);
                p[i + 1][0] = (1 - t) * p[i][0] + t * p[i + 1][0];
                p[i + 1][1] = (1 - t) * p[i][1] + t * p[i + 1][1];
            }
        }
        double prev = 0;
        for (int i = 0; i < n; i++) {
            double a = Math.atan2(p[i + 1][1] - p[i][1], p[i + 1][0] - p[i][0]);
            q[i] = GeometryLinalg.wrap(a - prev);
            prev = a;
        }
    }
}
