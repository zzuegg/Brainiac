package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * PowerSGD (Vogels, Karimireddy and Jaggi 2019, <a href="https://arxiv.org/abs/1905.13727">arXiv:1905.13727</a>):
 * rank-{@code r} gradient compression by one step of power iteration that is itself
 * all-reducible. Each worker forms {@code M_i = G_i + e_i} (its gradient plus its error-feedback
 * residual) and sends {@code P_i = M_i Q}; the mean {@code P} is orthonormalized, each worker
 * sends {@code Q_i = M_i^T P}, and everyone applies {@code P Q^T} with {@code Q} the mean. The
 * residual {@code e_i = M_i - P Q^T} is kept, and {@code Q} warm-starts the next step's power
 * iteration. Vectors and small matrices are all-reduced exactly.
 */
public final class PowerSgd implements GradientCompressor {

    private final int rank;
    private final boolean errorFeedback;
    private final SplittableRandom random;
    private final Map<Var, Tensor> q = new IdentityHashMap<>();
    private final Map<Var, List<Tensor>> residuals = new IdentityHashMap<>();
    private long bytes;

    /**
     * @param rank the rank {@code r}, for example 1 to 4
     * @param errorFeedback whether to keep and re-add the residual (the paper always does)
     * @param seed the seed of the initial {@code Q}
     */
    public PowerSgd(int rank, boolean errorFeedback, long seed) {
        this.rank = rank;
        this.errorFeedback = errorFeedback;
        this.random = new SplittableRandom(seed);
    }

    @Override
    public Tensor aggregate(Var parameter, List<Tensor> grads) {
        int workers = grads.size();
        Tensor g0 = grads.get(0);
        if (g0.rank() != 2 || Math.min(g0.shape().dim(0), g0.shape().dim(1)) <= rank) {
            bytes += 8L * g0.size() * workers;
            return GradientCompressor.mean(grads);
        }
        int n = g0.shape().dim(0);
        int m = g0.shape().dim(1);
        List<Tensor> res = residuals.computeIfAbsent(parameter, k -> new ArrayList<>());
        List<Tensor> ms = new ArrayList<>(workers);
        for (int i = 0; i < workers; i++) {
            Tensor mi = grads.get(i);
            if (errorFeedback && res.size() > i) {
                mi = mi.add(res.get(i));
            }
            ms.add(mi);
        }
        Tensor qm = q.computeIfAbsent(parameter, k -> Tensor.normal(Shape.of(m, rank), 0, 1, random));
        List<Tensor> ps = new ArrayList<>(workers);
        for (Tensor mi : ms) {
            ps.add(mi.matmul(qm));
        }
        Tensor p = orthonormalize(GradientCompressor.mean(ps));
        List<Tensor> qs = new ArrayList<>(workers);
        for (Tensor mi : ms) {
            qs.add(mi.transpose().matmul(p));
        }
        Tensor qNew = GradientCompressor.mean(qs);
        q.put(parameter, qNew);
        bytes += 8L * (long) (n + m) * rank * workers;
        Tensor approx = p.matmul(qNew.transpose());
        if (errorFeedback) {
            res.clear();
            for (Tensor mi : ms) {
                res.add(mi.sub(approx));
            }
        }
        return approx;
    }

    /** Gram-Schmidt on the columns. */
    private static Tensor orthonormalize(Tensor p) {
        int n = p.shape().dim(0);
        int r = p.shape().dim(1);
        float[] v = p.toArray();
        for (int c = 0; c < r; c++) {
            for (int prev = 0; prev < c; prev++) {
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += v[i * r + c] * v[i * r + prev];
                }
                for (int i = 0; i < n; i++) {
                    v[i * r + c] -= (float) (dot * v[i * r + prev]);
                }
            }
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += v[i * r + c] * v[i * r + c];
            }
            norm = Math.sqrt(norm) + 1e-12;
            for (int i = 0; i < n; i++) {
                v[i * r + c] /= (float) norm;
            }
        }
        return Tensor.of(p.shape(), v);
    }

    @Override
    public long bytes() {
        return bytes;
    }

    @Override
    public String toString() {
        return "PowerSGD(rank=" + rank + (errorFeedback ? "" : ", no EF") + ")";
    }
}
