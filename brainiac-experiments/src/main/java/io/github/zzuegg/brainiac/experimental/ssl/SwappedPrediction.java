package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * SwAV, online clustering with swapped prediction (Caron et al., NeurIPS 2020). Each view's
 * normalised embedding is scored against {@code K} learnable unit-norm prototypes. The
 * <em>codes</em> of a view are a balanced soft assignment of the batch to the prototypes: an
 * entropic optimal-transport plan with uniform marginals over samples and prototypes (the
 * equipartition constraint that prevents collapse), found by three Sinkhorn-Knopp iterations
 * ({@link EntropicTransport#sinkhornScaling}) at {@code epsilon = 0.05}. Each view then predicts
 * the codes of the other views by a softmax over its prototype scores at temperature
 * {@code tau = 0.1}: {@code -sum_k q_s,k log softmax(scores_t / tau)_k} for {@code s != t}.
 * Codes are only computed for the global views; extra low-resolution or small
 * {@linkplain #localCrops local crops} (multi-crop) only predict.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2006.09882">arXiv:2006.09882</a>.
 */
public final class SwappedPrediction {

    private final Var prototypes;
    private final float temperature;
    private final double epsilon;
    private final int iterations;

    /**
     * Creates random prototypes.
     *
     * @param dim the embedding width
     * @param count the number of prototypes {@code K}
     * @param temperature the prediction softmax temperature, 0.1 in the paper
     * @param epsilon the Sinkhorn regularisation, 0.05 in the paper
     * @param iterations Sinkhorn iterations, 3 in the paper
     * @param seed the seed of the prototypes
     */
    public SwappedPrediction(int dim, int count, float temperature, double epsilon, int iterations, long seed) {
        this.prototypes = Var.param(Tensor.normal(Shape.of(dim, count), 0, 1, new SplittableRandom(seed)));
        this.temperature = temperature;
        this.epsilon = epsilon;
        this.iterations = iterations;
    }

    /** @return the trainable prototypes {@code [dim, K]} */
    public List<Var> parameters() {
        return List.of(prototypes);
    }

    /**
     * Cosine scores of normalised embeddings against normalised prototypes.
     *
     * @param z the embeddings {@code [n, dim]}
     * @return {@code [n, K]} in {@code [-1, 1]}
     */
    public Var scores(Var z) {
        Var c = prototypes.div(prototypes.square().sum(0, true).add(1e-8f).sqrt());
        return SelfSupervisedLosses.normalize(z).matmul(c);
    }

    /**
     * The balanced codes of a batch: Sinkhorn-Knopp on {@code exp(scores / epsilon)} with
     * uniform marginals, each row then rescaled to sum to one.
     *
     * @param scores {@code [n, K]} prototype scores (not differentiated)
     * @return {@code [n, K]} soft assignments
     */
    public Tensor codes(Tensor scores) {
        int n = scores.shape().dim(0);
        int k = scores.shape().dim(1);
        float[] s = scores.toArray();
        double[][] cost = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                cost[i][j] = -s[i * k + j];
            }
        }
        double[][] plan = EntropicTransport.sinkhornScaling(EntropicTransport.uniform(n), EntropicTransport.uniform(k),
                cost, epsilon, iterations).coupling();
        float[] q = new float[n * k];
        for (int i = 0; i < n; i++) {
            double row = 0;
            for (int j = 0; j < k; j++) {
                row += plan[i][j];
            }
            for (int j = 0; j < k; j++) {
                q[i * k + j] = (float) (plan[i][j] / (row + 1e-30));
            }
        }
        return Tensor.of(Shape.of(n, k), q, scores.device());
    }

    /**
     * The swapped-prediction loss over several views of the same batch.
     *
     * @param views each view's embeddings {@code [n, dim]}; the first {@code codeViews} are the
     *     global views that get codes
     * @param codeViews how many leading views get codes, at least 1
     * @return the mean cross-entropy over all (code view, other view) pairs
     */
    public Var loss(List<Var> views, int codeViews) {
        Var[] scores = new Var[views.size()];
        for (int v = 0; v < views.size(); v++) {
            scores[v] = scores(views.get(v));
        }
        Var total = null;
        int terms = 0;
        for (int s = 0; s < codeViews; s++) {
            Tensor q = codes(scores[s].value());
            for (int t = 0; t < views.size(); t++) {
                if (t == s) {
                    continue;
                }
                Var logp = Activation.LOG_SOFTMAX.forward(scores[t].div(temperature));
                Var l = logp.mul(q).sum(-1, false).mean().neg();
                total = total == null ? l : total.add(l);
                terms++;
            }
        }
        return total.div(terms);
    }

    /**
     * Hard prototype assignments (the arg-max of the scores), for cluster purity.
     *
     * @param z embeddings {@code [n, dim]}
     * @return the prototype of each row
     */
    public int[] assign(Tensor z) {
        return Var.noGrad(() -> scores(Var.of(z))).value().argmax(1).toIntArray();
    }

    /**
     * Multi-crop's small views for images {@code [n, c, h, w]}: each image keeps a random
     * {@code window x window} square and the rest is zeroed (a cheap stand-in for SwAV's
     * low-resolution crops on images as small as these).
     *
     * @param images the images
     * @param window the crop side
     * @param random the source of crop positions
     * @return the cropped images, same shape
     */
    public static Tensor localCrops(Tensor images, int window, RandomGenerator random) {
        int n = images.shape().dim(0);
        int c = images.shape().dim(1);
        int h = images.shape().dim(2);
        int w = images.shape().dim(3);
        float[] x = images.toArray();
        float[] out = new float[x.length];
        for (int i = 0; i < n; i++) {
            int y0 = random.nextInt(h - window + 1);
            int x0 = random.nextInt(w - window + 1);
            for (int ch = 0; ch < c; ch++) {
                for (int y = y0; y < y0 + window; y++) {
                    for (int xx = x0; xx < x0 + window; xx++) {
                        int idx = ((i * c + ch) * h + y) * w + xx;
                        out[idx] = x[idx];
                    }
                }
            }
        }
        return Tensor.of(images.shape(), out, images.device());
    }
}
