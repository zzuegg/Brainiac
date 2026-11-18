package io.github.zzuegg.brainiac.experimental.geometry;

import io.github.zzuegg.brainiac.autograd.Decompositions;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.vision.EdgeConv;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep Closest Point (Wang &amp; Solomon, ICCV 2019, <a href="https://arxiv.org/abs/1905.03304">arXiv:1905.03304</a>):
 * both clouds are embedded by a shared DGCNN ({@link EdgeConv} layers), the embeddings exchange
 * information through one cross-attention block in each direction (the "transformer" step,
 * with residual connections), every source point gets a soft correspondence as the
 * attention-weighted average of the target points, and the rigid motion follows in closed
 * form from the weighted cross-covariance. The closed form here is Horn's quaternion method:
 * the optimal quaternion is the top eigenvector of a symmetric 4x4 matrix built from the
 * cross-covariance, found by shifted power iteration, so the whole pipeline stays
 * differentiable with plain autograd ops (in place of DCP's SVD backward). Trained on the
 * rotation and translation errors, it works from large initial rotations where ICP fails;
 * ICP then polishes its estimate.
 *
 * <p>{@link Procrustes#SVD} (wave 13, 13.4.17) replaces the power iteration with DCP's own
 * weighted Kabsch step through core {@link Decompositions#svd}: for {@code S = U Sigma V^T},
 * {@code R = V diag(1, 1, det(V U^T)) U^T}, with exact SVD gradients. The default stays the power
 * iteration.
 */
public final class DeepClosestPoint {

    /** How the rotation is solved from the weighted cross-covariance. */
    public enum Procrustes {
        /** Horn's quaternion method by shifted power iteration (wave 12). */
        HORN_POWER_ITERATION,
        /** Kabsch through the differentiable SVD with the reflection correction. */
        SVD
    }

    private final EdgeConv first;
    private final EdgeConv second;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final int width;
    private final Procrustes procrustes;

    /**
     * Creates an untrained network with the power-iteration Procrustes step.
     *
     * @param width the embedding width
     * @param k the EdgeConv neighbourhood size
     * @param seed the seed
     */
    public DeepClosestPoint(int width, int k, long seed) {
        this(width, k, seed, Procrustes.HORN_POWER_ITERATION);
    }

    /**
     * Creates an untrained network.
     *
     * @param width the embedding width
     * @param k the EdgeConv neighbourhood size
     * @param seed the seed
     * @param procrustes the rotation solver
     */
    public DeepClosestPoint(int width, int k, long seed, Procrustes procrustes) {
        this.procrustes = procrustes;
        SplittableRandom r = new SplittableRandom(seed);
        this.first = new EdgeConv(3, width, k, r);
        this.second = new EdgeConv(width, width, k, r);
        float std = (float) Math.sqrt(1.0 / width);
        this.wq = Var.param(Tensor.normal(Shape.of(width, width), 0, std, r));
        this.wk = Var.param(Tensor.normal(Shape.of(width, width), 0, std, r));
        this.wv = Var.param(Tensor.normal(Shape.of(width, width), 0, std, r));
        this.width = width;
    }

    /** @return every parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(first.parameters());
        p.addAll(second.parameters());
        p.add(wq);
        p.add(wk);
        p.add(wv);
        return p;
    }

    private Var embed(Var x) {
        Var a = first.forward(x);
        return second.forward(a).add(a);
    }

    private Var cross(Var a, Var b) {
        int bs = a.shape().dim(0);
        int n = a.shape().dim(1);
        int m = b.shape().dim(1);
        Var q = a.reshape(bs * n, width).matmul(wq).reshape(bs, n, width);
        Var k = b.reshape(bs * m, width).matmul(wk).reshape(bs, m, width);
        Var v = b.reshape(bs * m, width).matmul(wv).reshape(bs, m, width);
        Var att = q.matmul(k.permute(0, 2, 1)).mul((float) (1 / Math.sqrt(width))).softmax(2);
        return a.add(att.matmul(v));
    }

    /**
     * Estimates the motion taking each source to its target.
     *
     * @param source {@code [b, n, 3]}
     * @param target {@code [b, m, 3]}
     * @return {@code {R [b, 3, 3], t [b, 3]}}
     */
    public Var[] forward(Var source, Var target) {
        Var fx = embed(source);
        Var fy = embed(target);
        Var px = cross(fx, fy);
        Var py = cross(fy, fx);
        Var scores = px.matmul(py.permute(0, 2, 1)).mul((float) (1 / Math.sqrt(width))).softmax(2);
        Var corr = scores.matmul(target);
        Var xm = source.mean(1, true);
        Var ym = corr.mean(1, true);
        Var xc = source.sub(xm);
        Var yc = corr.sub(ym);
        Var s = xc.permute(0, 2, 1).matmul(yc);
        Var rot = procrustes == Procrustes.SVD ? kabsch(s) : horn(s);
        int b = source.shape().dim(0);
        Var t = ym.reshape(b, 3).sub(rot.matmul(xm.reshape(b, 3, 1)).reshape(b, 3));
        return new Var[] {rot, t};
    }

    /** Kabsch: the rotation maximising {@code sum y_i^T R x_i} for {@code S = sum x y^T}, by SVD. */
    static Var kabsch(Var s) {
        int b = s.shape().dim(0);
        Var[] usv = Decompositions.svd(s);
        Var v = usv[2].permute(0, 2, 1);
        Var ut = usv[0].permute(0, 2, 1);
        float[] vv = v.value().toArray();
        float[] uu = ut.value().toArray();
        float[] d = new float[b * 9];
        for (int i = 0; i < b; i++) {
            double[][] m = new double[3][3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    for (int k = 0; k < 3; k++) {
                        m[r][c] += vv[i * 9 + r * 3 + k] * uu[i * 9 + k * 3 + c];
                    }
                }
            }
            double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            d[i * 9] = 1;
            d[i * 9 + 4] = 1;
            d[i * 9 + 8] = det < 0 ? -1 : 1;
        }
        return v.matmul(Var.of(Tensor.of(Shape.of(b, 3, 3), d))).matmul(ut);
    }

    private static Var e(Var s, int i, int j) {
        return s.slice(1, i, i + 1).slice(2, j, j + 1);
    }

    /** Horn's method: the rotation maximising {@code sum y_i^T R x_i} for {@code S = sum x y^T}. */
    static Var horn(Var s) {
        int b = s.shape().dim(0);
        Var sxx = e(s, 0, 0);
        Var sxy = e(s, 0, 1);
        Var sxz = e(s, 0, 2);
        Var syx = e(s, 1, 0);
        Var syy = e(s, 1, 1);
        Var syz = e(s, 1, 2);
        Var szx = e(s, 2, 0);
        Var szy = e(s, 2, 1);
        Var szz = e(s, 2, 2);
        Var[][] n = {
            {sxx.add(syy).add(szz), syz.sub(szy), szx.sub(sxz), sxy.sub(syx)},
            {syz.sub(szy), sxx.sub(syy).sub(szz), sxy.add(syx), szx.add(sxz)},
            {szx.sub(sxz), sxy.add(syx), syy.sub(sxx).sub(szz), syz.add(szy)},
            {sxy.sub(syx), szx.add(sxz), syz.add(szy), szz.sub(sxx).sub(syy)}};
        Var[] rows = new Var[4];
        for (int i = 0; i < 4; i++) {
            rows[i] = Var.concat(2, n[i]);
        }
        Var mat = Var.concat(1, rows);
        Var shift = s.abs().sum(1, false).sum(1, false).reshape(b, 1, 1).mul(3f).add(1e-3f);
        float[] eye = new float[b * 16];
        for (int i = 0; i < b; i++) {
            for (int d = 0; d < 4; d++) {
                eye[i * 16 + d * 5] = 1;
            }
        }
        Var shifted = mat.add(shift.mul(Tensor.of(Shape.of(b, 4, 4), eye)));
        float[] q0 = new float[b * 4];
        for (int i = 0; i < b; i++) {
            q0[i * 4] = 1;
            q0[i * 4 + 1] = 0.1f;
            q0[i * 4 + 2] = 0.05f;
            q0[i * 4 + 3] = 0.02f;
        }
        Var q = Var.of(Tensor.of(Shape.of(b, 4, 1), q0));
        for (int it = 0; it < 40; it++) {
            q = shifted.matmul(q).normalize(1, 1e-12f);
        }
        Var w = q.slice(1, 0, 1).reshape(b, 1);
        Var x = q.slice(1, 1, 2).reshape(b, 1);
        Var y = q.slice(1, 2, 3).reshape(b, 1);
        Var z = q.slice(1, 3, 4).reshape(b, 1);
        Var r00 = w.square().add(x.square()).sub(y.square()).sub(z.square());
        Var r01 = x.mul(y).sub(w.mul(z)).mul(2f);
        Var r02 = x.mul(z).add(w.mul(y)).mul(2f);
        Var r10 = x.mul(y).add(w.mul(z)).mul(2f);
        Var r11 = w.square().sub(x.square()).add(y.square()).sub(z.square());
        Var r12 = y.mul(z).sub(w.mul(x)).mul(2f);
        Var r20 = x.mul(z).sub(w.mul(y)).mul(2f);
        Var r21 = y.mul(z).add(w.mul(x)).mul(2f);
        Var r22 = w.square().sub(x.square()).sub(y.square()).add(z.square());
        return Var.concat(1, r00, r01, r02, r10, r11, r12, r20, r21, r22).reshape(b, 3, 3);
    }

    /**
     * Trains on random rigid motions of clouds drawn by a generator.
     *
     * @param pairs supplies {@code {source [n][3], target [n][3], R [3][3], t [1][3]}}
     * @param steps gradient steps
     * @param batch pairs per step
     * @param learningRate Adam's rate
     * @return the last loss
     */
    public double fit(java.util.function.Supplier<double[][][]> pairs, int steps, int batch, float learningRate) {
        Optimizer opt = Optimizer.adam(learningRate);
        double last = 0;
        for (int s = 0; s < steps; s++) {
            double[][][][] data = new double[batch][][][];
            for (int i = 0; i < batch; i++) {
                data[i] = pairs.get();
            }
            int n = data[0][0].length;
            int m = data[0][1].length;
            float[] src = new float[batch * n * 3];
            float[] tgt = new float[batch * m * 3];
            float[] rot = new float[batch * 9];
            float[] tr = new float[batch * 3];
            for (int i = 0; i < batch; i++) {
                for (int p = 0; p < n; p++) {
                    for (int c = 0; c < 3; c++) {
                        src[(i * n + p) * 3 + c] = (float) data[i][0][p][c];
                    }
                }
                for (int p = 0; p < m; p++) {
                    for (int c = 0; c < 3; c++) {
                        tgt[(i * m + p) * 3 + c] = (float) data[i][1][p][c];
                    }
                }
                for (int a = 0; a < 3; a++) {
                    for (int c = 0; c < 3; c++) {
                        rot[i * 9 + a * 3 + c] = (float) data[i][2][a][c];
                    }
                    tr[i * 3 + a] = (float) data[i][3][0][a];
                }
            }
            Var[] out = forward(Var.of(Tensor.of(Shape.of(batch, n, 3), src)),
                Var.of(Tensor.of(Shape.of(batch, m, 3), tgt)));
            Var loss = out[0].sub(Var.of(Tensor.of(Shape.of(batch, 3, 3), rot))).square().sum(1, false).sum(1, false)
                .mean().add(out[1].sub(Var.of(Tensor.of(Shape.of(batch, 3), tr))).square().sum(1, false).mean());
            last = loss.value().item();
            loss.backward();
            opt.step(parameters());
        }
        return last;
    }

    /**
     * Estimates one motion.
     *
     * @param source the source cloud
     * @param target the target cloud
     * @return {@code {R rows 0-2, t row 3}}
     */
    public double[][] estimate(double[][] source, double[][] target) {
        int n = source.length;
        int m = target.length;
        float[] a = new float[n * 3];
        float[] b = new float[m * 3];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < 3; c++) {
                a[i * 3 + c] = (float) source[i][c];
            }
        }
        for (int i = 0; i < m; i++) {
            for (int c = 0; c < 3; c++) {
                b[i * 3 + c] = (float) target[i][c];
            }
        }
        Var[] out = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, n, 3), a)),
            Var.of(Tensor.of(Shape.of(1, m, 3), b))));
        float[] r = out[0].value().toArray();
        float[] t = out[1].value().toArray();
        double[][] rr = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rr[i][j] = r[i * 3 + j];
            }
        }
        rr = GeometryLinalg.nearestRotation(rr);
        return new double[][] {rr[0], rr[1], rr[2], {t[0], t[1], t[2]}};
    }
}
