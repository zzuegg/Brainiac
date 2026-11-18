package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The discriminative embedding loss for instance segmentation (De Brabandere, Neven and Van
 * Gool, 2017): a network predicts an embedding per pixel, and the loss
 *
 * <ul>
 *   <li>pulls each instance's pixels towards their mean, with a hinge so pixels within
 *       {@code deltaVar} of it are left alone (variance term);</li>
 *   <li>pushes the means of the instances of one image apart until they are
 *       {@code 2 deltaDist} apart (distance term);</li>
 *   <li>keeps the means near the origin (a small regulariser).</li>
 * </ul>
 *
 * Instances are then read out by clustering ({@link EmbeddingClustering}), so the number of
 * objects need not be known in advance and touching objects of one class can still be told
 * apart, which semantic segmentation plus connected components cannot do. It is a core
 * {@link Loss}: the prediction is the embedding map {@code [batch, dims, h, w]} and the target
 * the instance ids {@code [batch, h, w]} with 0 for background (ignored).
 *
 * <p>Paper: De Brabandere, Neven and Van Gool, "Semantic Instance Segmentation with a
 * Discriminative Loss Function" (2017),
 * <a href="https://arxiv.org/abs/1708.02551">arXiv:1708.02551</a>.
 */
public final class DiscriminativeEmbeddingLoss implements Loss {

    private final float deltaVar;
    private final float deltaDist;
    private final float regularisation;

    /**
     * Creates the loss.
     *
     * @param deltaVar the pull margin, such as 0.5
     * @param deltaDist half the push margin, such as 1.5
     * @param regularisation the weight of the mean-norm term, such as 0.001
     */
    public DiscriminativeEmbeddingLoss(float deltaVar, float deltaDist, float regularisation) {
        this.deltaVar = deltaVar;
        this.deltaDist = deltaDist;
        this.regularisation = regularisation;
    }

    /** @return the pull margin */
    public float deltaVar() {
        return deltaVar;
    }

    /** @return half the push margin */
    public float deltaDist() {
        return deltaDist;
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        int b = prediction.shape().dim(0);
        int e = prediction.shape().dim(1);
        int h = prediction.shape().dim(2);
        int w = prediction.shape().dim(3);
        int hw = h * w;
        float[] ids = target.toArray();
        List<Integer> pixels = new ArrayList<>();
        List<Integer> owner = new ArrayList<>();
        List<Integer> imageOf = new ArrayList<>();
        List<Integer> counts = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            Map<Integer, Integer> local = new HashMap<>();
            for (int p = 0; p < hw; p++) {
                int id = (int) ids[i * hw + p];
                if (id <= 0) {
                    continue;
                }
                Integer c = local.get(id);
                if (c == null) {
                    c = counts.size();
                    local.put(id, c);
                    counts.add(0);
                    imageOf.add(i);
                }
                counts.set(c, counts.get(c) + 1);
                pixels.add(i * hw + p);
                owner.add(c);
            }
        }
        int instances = counts.size();
        if (instances == 0) {
            return prediction.sum().mul(0f);
        }
        int n = pixels.size();
        float[] pix = new float[n];
        float[] own = new float[n];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) {
            pix[i] = pixels.get(i);
            own[i] = owner.get(i);
            weight[i] = 1f / (counts.get(owner.get(i)) * instances);
        }
        float[] inv = new float[instances];
        for (int c = 0; c < instances; c++) {
            inv[c] = 1f / counts.get(c);
        }
        Tensor ownT = Tensor.of(Shape.of(n), own);
        Var x = prediction.permute(0, 2, 3, 1).reshape(b * hw, e).gather(0, Tensor.of(Shape.of(n), pix));
        Var means = x.scatterAdd(0, ownT, instances).mul(Tensor.of(Shape.of(instances, 1), inv));
        Var d = x.sub(means.gather(0, ownT)).square().sum(1, false).add(1e-8f).sqrt();
        Var variance = d.sub(deltaVar).maximum(0f).square().mul(Tensor.of(Shape.of(n), weight)).sum();
        List<int[]> pairs = new ArrayList<>();
        for (int a = 0; a < instances; a++) {
            for (int c = 0; c < instances; c++) {
                if (a != c && imageOf.get(a).equals(imageOf.get(c))) {
                    pairs.add(new int[] {a, c});
                }
            }
        }
        Var loss = variance;
        if (!pairs.isEmpty()) {
            float[] ia = new float[pairs.size()];
            float[] ib = new float[pairs.size()];
            for (int i = 0; i < ia.length; i++) {
                ia[i] = pairs.get(i)[0];
                ib[i] = pairs.get(i)[1];
            }
            Var gap = means.gather(0, Tensor.of(Shape.of(ia.length), ia))
                    .sub(means.gather(0, Tensor.of(Shape.of(ib.length), ib))).square().sum(1, false).add(1e-8f).sqrt();
            loss = loss.add(gap.neg().add(2 * deltaDist).maximum(0f).square().mean());
        }
        Var reg = means.square().sum(1, false).add(1e-8f).sqrt().mean();
        return loss.add(reg.mul(regularisation));
    }

    @Override
    public String toString() {
        return "DiscriminativeEmbeddingLoss(" + deltaVar + ", " + deltaDist + ")";
    }
}
