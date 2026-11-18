package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Relational knowledge distillation, RKD (Park et al. 2019, <a href="https://arxiv.org/abs/1904.05068">arXiv:1904.05068</a>):
 * distils the <em>relations</em> between the examples of a batch instead of each example's
 * output, so student and teacher embeddings may differ in width and in rotation:
 * <ul>
 * <li>distance-wise: the pairwise distances of the student's embeddings, each normalised by
 *     its own batch mean, match the teacher's under a Huber loss;</li>
 * <li>angle-wise: the cosine of the angle at every anchor {@code j} between {@code e_i - e_j}
 *     and {@code e_k - e_j} matches the teacher's (all triplets of the batch).</li>
 * </ul>
 * The loss is {@code distanceWeight * L_D + angleWeight * L_A}, the paper uses 1 and 2. As a
 * core {@link Loss} the prediction is the student's embeddings {@code [n, d_s]} and the target
 * the teacher's {@code [n, d_t]}.
 */
public final class RelationalKdLoss implements Loss {

    private final float distanceWeight;
    private final float angleWeight;

    /**
     * @param distanceWeight the weight of the distance term, for example 1
     * @param angleWeight the weight of the angle term, for example 2
     */
    public RelationalKdLoss(float distanceWeight, float angleWeight) {
        this.distanceWeight = distanceWeight;
        this.angleWeight = angleWeight;
    }

    @Override
    public Var compute(Var student, Tensor teacher) {
        int n = student.shape().dim(0);
        Var loss = null;
        if (distanceWeight != 0) {
            Var ds = distances(student);
            Tensor dt = distances(Var.of(teacher)).value();
            Var dsn = ds.div(ds.sum().div((float) n * (n - 1)));
            Tensor dtn = dt.div(dt.sum().div((float) n * (n - 1)));
            loss = huber(dsn.sub(Var.of(dtn))).sum().mul(distanceWeight / ((float) n * n));
        }
        if (angleWeight != 0) {
            Var as = angles(student);
            Tensor at = angles(Var.of(teacher)).value();
            Var a = huber(as.sub(Var.of(at))).sum().mul(angleWeight / ((float) n * n * n));
            loss = loss == null ? a : loss.add(a);
        }
        return loss;
    }

    /** Pairwise Euclidean distances {@code [n, n]}, zero on the diagonal. */
    static Var distances(Var e) {
        int n = e.shape().dim(0);
        int d = e.shape().dim(1);
        Var diff = e.reshape(1, n, d).expand(Shape.of(n, n, d)).sub(e.reshape(n, 1, d).expand(Shape.of(n, n, d)));
        Tensor offDiag = Tensor.ones(Shape.of(n, n)).sub(eye(n));
        return diff.square().sum(2, false).add(Tensor.full(Shape.of(n, n), 1e-12f)).sqrt().mul(offDiag);
    }

    /** {@code [n, n, n]}: at anchor {@code j}, the cosine between {@code e_i - e_j} and {@code e_k - e_j}. */
    static Var angles(Var e) {
        int n = e.shape().dim(0);
        int d = e.shape().dim(1);
        // diff[j, i] = e_i - e_j
        Var diff = e.reshape(1, n, d).expand(Shape.of(n, n, d)).sub(e.reshape(n, 1, d).expand(Shape.of(n, n, d)));
        Var norm = diff.square().sum(2, true).add(1e-12f).sqrt();
        Var unit = diff.div(norm);
        return unit.matmul(unit.permute(0, 2, 1));
    }

    private static Var huber(Var x) {
        Var d = x.abs();
        Var small = d.minimum(1f);
        return small.square().mul(0.5f).add(d.sub(small));
    }

    private static Tensor eye(int n) {
        float[] v = new float[n * n];
        for (int i = 0; i < n; i++) {
            v[i * n + i] = 1;
        }
        return Tensor.of(Shape.of(n, n), v);
    }

    @Override
    public String toString() {
        return "RelationalKd(distance " + distanceWeight + ", angle " + angleWeight + ")";
    }
}
