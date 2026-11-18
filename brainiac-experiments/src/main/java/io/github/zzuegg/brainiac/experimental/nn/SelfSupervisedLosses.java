package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The objectives of joint-embedding self-supervised learning. Each takes the embeddings of
 * two augmented views of the same batch, {@code [batch, dim]}, and returns a scalar to
 * minimize; none needs labels.
 *
 * <ul>
 *   <li>{@link #ntXent}: SimCLR's normalized-temperature cross-entropy (InfoNCE): each view
 *       must pick its partner out of all {@code 2N - 1} other embeddings. Chen et al.,
 *       <a href="https://arxiv.org/abs/2002.05709">arXiv:2002.05709</a>.</li>
 *   <li>{@link #byol}: BYOL's cosine regression of an online predictor onto a momentum
 *       target; no negatives. Grill et al., <a href="https://arxiv.org/abs/2006.07733">arXiv:2006.07733</a>.</li>
 *   <li>{@link #dino}: DINO's cross-entropy between a sharpened, centered teacher
 *       distribution and the student's. Caron et al.,
 *       <a href="https://arxiv.org/abs/2104.14294">arXiv:2104.14294</a>; Oquab et al., DINOv2,
 *       <a href="https://arxiv.org/abs/2304.07193">arXiv:2304.07193</a>.</li>
 *   <li>{@link #barlowTwins}: pushes the cross-correlation matrix of the two views to the
 *       identity: invariant yet decorrelated features. Zbontar et al.,
 *       <a href="https://arxiv.org/abs/2103.03230">arXiv:2103.03230</a>.</li>
 *   <li>{@link #vicreg}: variance-invariance-covariance regularization, three explicit
 *       terms against collapse. Bardes et al., <a href="https://arxiv.org/abs/2105.04906">arXiv:2105.04906</a>.</li>
 * </ul>
 *
 * <p>For masked joint-embedding prediction (I-JEPA; Assran et al. 2023,
 * <a href="https://arxiv.org/abs/2301.08243">arXiv:2301.08243</a>) it also gives the
 * paper's recipe pieces: {@link #multiBlockMask} (several rectangular target blocks and a
 * large context block with the targets removed), {@link #cosineMomentum} (the target
 * encoder's EMA momentum rising from 0.996 to 1 on a cosine), and {@link #effectiveRank}
 * (RankMe, a collapse check on the learned features).
 */
public final class SelfSupervisedLosses {

    private SelfSupervisedLosses() {
    }

    /**
     * Scales each row to unit length.
     *
     * @param z {@code [rows, dim]}
     * @return the normalized rows
     */
    public static Var normalize(Var z) {
        return z.div(z.square().sum(-1, true).add(1e-8f).sqrt());
    }

    /**
     * SimCLR's NT-Xent / InfoNCE loss, with every other embedding of both views as a
     * negative ({@code 2N - 2} of them); the core {@code Functional.infoNce} is the
     * CLIP-style variant that contrasts only across the two views.
     *
     * @param z1 the first views' embeddings {@code [n, dim]}
     * @param z2 the second views' embeddings {@code [n, dim]}
     * @param temperature the softmax temperature, 0.1 to 0.5 typically
     * @return the mean cross-entropy of finding each partner
     */
    public static Var ntXent(Var z1, Var z2, float temperature) {
        int n = z1.shape().dim(0);
        Var z = normalize(Var.concat(0, z1, z2));
        Var sim = z.matmul(z.transpose()).mul(1f / temperature);
        float[] self = new float[4 * n * n];
        float[] partner = new float[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            self[i * 2 * n + i] = TechniqueOps.MASKED;
            partner[i] = i < n ? i + n : i - n;
        }
        sim = sim.add(Tensor.of(Shape.of(2 * n, 2 * n), self, z1.value().device()));
        return Functional.crossEntropy(sim, Tensor.of(Shape.of(2 * n), partner, z1.value().device()));
    }

    /**
     * BYOL's loss: {@code 2 - 2 cos(p, z)} averaged over rows, with the target held fixed.
     *
     * @param prediction the online network's predictions {@code [n, dim]}
     * @param target the momentum network's projections {@code [n, dim]} (detached here)
     * @return the loss
     */
    public static Var byol(Var prediction, Var target) {
        Var cos = normalize(prediction).mul(normalize(target.detach())).sum(-1, false);
        return cos.mean().mul(-2f).add(2f);
    }

    /**
     * DINO's loss: {@code -sum softmax((t - center) / tT) * logSoftmax(s / tS)}, averaged over
     * rows, with the teacher held fixed.
     *
     * @param student the student's head outputs {@code [n, k]}
     * @param teacher the teacher's head outputs {@code [n, k]} (detached here)
     * @param center the running center of teacher outputs {@code [k]}
     * @param studentTemperature 0.1 in the paper
     * @param teacherTemperature 0.04 to 0.07 in the paper: sharper than the student's
     * @return the loss
     */
    public static Var dino(Var student, Var teacher, Tensor center, float studentTemperature,
            float teacherTemperature) {
        Tensor t = Activation.SOFTMAX.forward(Var.of(teacher.value().sub(center).div(teacherTemperature))).value();
        Var logS = Activation.LOG_SOFTMAX.forward(student.div(studentTemperature));
        return logS.mul(t).sum().mul(-1f / student.shape().dim(0));
    }

    /**
     * The running mean of teacher outputs that DINO subtracts to avoid collapse to one
     * dimension (centering), while the low teacher temperature avoids collapse to the
     * uniform distribution (sharpening).
     */
    public static final class Center {

        private final float momentum;
        private Tensor value;

        /**
         * Creates a zero center.
         *
         * @param dim the head output width
         * @param momentum the update momentum, 0.9 in the paper
         */
        public Center(int dim, float momentum) {
            this.momentum = momentum;
            this.value = Tensor.zeros(Shape.of(dim));
        }

        /**
         * Moves the center towards the batch mean of teacher outputs.
         *
         * @param teacher teacher outputs {@code [n, dim]}
         */
        public void update(Tensor teacher) {
            value = value.to(teacher.device()).mul(momentum).add(teacher.mean(0, false).mul(1 - momentum));
        }

        /** @return the current center {@code [dim]} */
        public Tensor value() {
            return value;
        }
    }

    /**
     * Barlow Twins: {@code sum_i (1 - C_ii)^2 + lambda sum_{i != j} C_ij^2} with {@code C} the
     * cross-correlation of the batch-standardized embeddings.
     *
     * @param z1 {@code [n, dim]}
     * @param z2 {@code [n, dim]}
     * @param lambda the off-diagonal weight, about {@code 1 / dim}
     * @return the loss
     */
    public static Var barlowTwins(Var z1, Var z2, float lambda) {
        int n = z1.shape().dim(0);
        int d = z1.shape().dim(1);
        Var c = standardizeColumns(z1).transpose().matmul(standardizeColumns(z2)).mul(1f / n);
        Tensor eye = eye(d, z1.value().device());
        Var onDiag = c.mul(eye).sum(-1, false).sub(1f).square().sum();
        Var offDiag = c.mul(eye.neg().add(1f)).square().sum();
        return onDiag.add(offDiag.mul(lambda));
    }

    /**
     * VICReg: {@code invariance * mse(z1, z2) + variance * (v(z1) + v(z2)) + covariance * (c(z1) + c(z2))},
     * where {@code v} is the mean hinge {@code relu(1 - std)} of each dimension and {@code c}
     * the sum of squared off-diagonal covariances divided by the dimension.
     *
     * @param z1 {@code [n, dim]}
     * @param z2 {@code [n, dim]}
     * @param invariance the invariance weight, 25 in the paper
     * @param variance the variance weight, 25 in the paper
     * @param covariance the covariance weight, 1 in the paper
     * @return the loss
     */
    public static Var vicreg(Var z1, Var z2, float invariance, float variance, float covariance) {
        Var inv = z1.sub(z2).square().mean();
        Var var = varianceHinge(z1).add(varianceHinge(z2));
        Var cov = covariancePenalty(z1).add(covariancePenalty(z2));
        return inv.mul(invariance).add(var.mul(variance)).add(cov.mul(covariance));
    }

    private static Var varianceHinge(Var z) {
        Var c = z.sub(z.mean(0, true));
        Var std = c.square().mean(0, false).add(1e-4f).sqrt();
        return std.neg().add(1f).relu().mean();
    }

    /**
     * I-JEPA's multi-block mask on a {@code rows x cols} patch grid. Each sample gets
     * {@code targets} rectangular target blocks, each covering a random share in
     * {@code [targetMin, targetMax]} of the grid with an aspect ratio in {@code [0.75, 1.5]},
     * and one context block covering {@code contextScale} of the grid, from which every
     * target patch is removed. Counts are cut to the smallest in the batch (random patches
     * dropped), as the reference implementation does, so the batch stays rectangular.
     *
     * @param batch the samples
     * @param rows grid rows
     * @param cols grid columns
     * @param targets target blocks per sample, such as 4
     * @param targetMin the smallest target block share, such as 0.15
     * @param targetMax the largest target block share, such as 0.2
     * @param contextScale the context block's share before removing targets, such as 0.85-1
     * @param random the random source
     * @return {@code keep} the context patches, {@code masked} the union of the target patches
     */
    public static PatchMasking.Mask multiBlockMask(int batch, int rows, int cols, int targets, double targetMin,
            double targetMax, double contextScale, java.util.random.RandomGenerator random) {
        int tokens = rows * cols;
        java.util.List<int[]> ctx = new java.util.ArrayList<>();
        java.util.List<int[]> tgt = new java.util.ArrayList<>();
        int minCtx = Integer.MAX_VALUE;
        int minTgt = Integer.MAX_VALUE;
        for (int b = 0; b < batch; b++) {
            boolean[] isTarget = new boolean[tokens];
            for (int k = 0; k < targets; k++) {
                block(rows, cols, targetMin + (targetMax - targetMin) * random.nextDouble(), random, isTarget);
            }
            boolean[] inContext = new boolean[tokens];
            block(rows, cols, contextScale, random, inContext);
            java.util.List<Integer> c = new java.util.ArrayList<>();
            java.util.List<Integer> t = new java.util.ArrayList<>();
            for (int i = 0; i < tokens; i++) {
                if (isTarget[i]) {
                    t.add(i);
                } else if (inContext[i]) {
                    c.add(i);
                }
            }
            if (c.isEmpty()) {   // keep at least one context patch
                for (int i = 0; i < tokens && c.isEmpty(); i++) {
                    if (!isTarget[i]) {
                        c.add(i);
                    }
                }
            }
            ctx.add(c.stream().mapToInt(Integer::intValue).toArray());
            tgt.add(t.stream().mapToInt(Integer::intValue).toArray());
            minCtx = Math.min(minCtx, c.size());
            minTgt = Math.min(minTgt, t.size());
        }
        int[][] keep = new int[batch][];
        int[][] masked = new int[batch][];
        for (int b = 0; b < batch; b++) {
            keep[b] = subsample(ctx.get(b), minCtx, random);
            masked[b] = subsample(tgt.get(b), minTgt, random);
        }
        return new PatchMasking.Mask(keep, masked);
    }

    private static void block(int rows, int cols, double scale, java.util.random.RandomGenerator random,
            boolean[] into) {
        double area = scale * rows * cols;
        double aspect = 0.75 + 0.75 * random.nextDouble();
        int h = (int) Math.max(1, Math.min(rows, Math.round(Math.sqrt(area * aspect))));
        int w = (int) Math.max(1, Math.min(cols, Math.round(area / h)));
        int r0 = random.nextInt(rows - h + 1);
        int c0 = random.nextInt(cols - w + 1);
        for (int r = r0; r < r0 + h; r++) {
            for (int c = c0; c < c0 + w; c++) {
                into[r * cols + c] = true;
            }
        }
    }

    private static int[] subsample(int[] idx, int n, java.util.random.RandomGenerator random) {
        int[] a = idx.clone();
        for (int i = a.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        int[] out = java.util.Arrays.copyOf(a, n);
        java.util.Arrays.sort(out);
        return out;
    }

    /**
     * The target encoder's momentum on a cosine schedule, as I-JEPA, BYOL and DINO use:
     * {@code end - (end - start) (cos(pi step / total) + 1) / 2}.
     *
     * @param step the current step
     * @param total the total steps
     * @param start the momentum at step 0, such as 0.996
     * @param end the momentum at the last step, such as 1
     * @return the momentum
     */
    public static float cosineMomentum(int step, int total, float start, float end) {
        double t = Math.min(1, step / (double) Math.max(1, total));
        return (float) (end - (end - start) * (Math.cos(Math.PI * t) + 1) / 2);
    }

    /**
     * RankMe's effective rank (Garrido et al. 2023, <a href="https://arxiv.org/abs/2210.02885">arXiv:2210.02885</a>):
     * {@code exp} of the entropy of the normalized singular values of a feature matrix. A
     * collapsed representation has an effective rank near 1.
     *
     * @param features {@code [n, d]}
     * @return a value in {@code [1, min(n, d)]}
     */
    public static double effectiveRank(Tensor features) {
        Tensor[] usv = io.github.zzuegg.brainiac.tensor.Linalg.svd(features.to(io.github.zzuegg.brainiac.tensor.DType.FLOAT64));
        double[] sv = usv[1].toDoubleArray();
        double sum = 0;
        for (double v : sv) {
            sum += Math.abs(v);
        }
        double h = 0;
        for (double v : sv) {
            double p = Math.abs(v) / sum + 1e-12;
            h -= p * Math.log(p);
        }
        return Math.exp(h);
    }

    private static Var covariancePenalty(Var z) {
        int n = z.shape().dim(0);
        int d = z.shape().dim(1);
        Var c = z.sub(z.mean(0, true));
        Var cov = c.transpose().matmul(c).mul(1f / Math.max(1, n - 1));
        return cov.mul(eye(d, z.value().device()).neg().add(1f)).square().sum().mul(1f / d);
    }

    private static Var standardizeColumns(Var z) {
        Var c = z.sub(z.mean(0, true));
        return c.div(c.square().mean(0, true).add(1e-5f).sqrt());
    }

    private static Tensor eye(int d, io.github.zzuegg.brainiac.tensor.Device device) {
        float[] e = new float[d * d];
        for (int i = 0; i < d; i++) {
            e[i * d + i] = 1;
        }
        return Tensor.of(Shape.of(d, d), e, device);
    }
}
