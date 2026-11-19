package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The prototype layer of ProtoPNet ("This looks like that", Chen et al., NeurIPS 2019,
 * <a href="https://arxiv.org/abs/1806.10574">arXiv:1806.10574</a>): {@code m} learned
 * prototype vectors in the channel space of a convolutional feature map {@code [batch, C, H, W]};
 * each output is the prototype's similarity to its best-matching {@code 1x1} patch, so
 * {@code [batch, m]}.
 *
 * <ul>
 *   <li>{@link Similarity#L2_LOG}, the original log-activation
 *       {@code log((d + 1) / (d + 1e-4))} of the squared distance {@code d}, maximised over
 *       positions.</li>
 *   <li>{@link Similarity#COSINE}, the cosine similarity of the patch and the prototype,
 *       maximised over positions, which Moffett et al. (WACV 2026) report to be less sensitive
 *       to hyperparameters and to give better prototypes.</li>
 * </ul>
 * {@link #patchSimilarities} exposes the per-position map (for "which patch looks like the
 * prototype"), {@link #distances} the per-position distances for ProtoPNet's cluster and
 * separation costs, and {@link #project} the projection step that snaps each prototype onto its
 * nearest training patch of its own class, making it a real case.
 */
public final class PrototypeLayer implements Layer {

    /** The similarity function. */
    public enum Similarity {
        /** ProtoPNet's log activation of the squared L2 distance. */
        L2_LOG,
        /** Cosine similarity. */
        COSINE
    }

    private final Var prototypes;
    private final Similarity similarity;
    private final int channels;

    /**
     * Creates the layer with random prototypes.
     *
     * @param count the number of prototypes
     * @param channels the feature-map channels
     * @param similarity the similarity
     * @param random the initial values
     */
    public PrototypeLayer(int count, int channels, Similarity similarity, SplittableRandom random) {
        this.prototypes = Var.param(Tensor.uniform(Shape.of(count, channels), 0f, 1f, random));
        this.similarity = similarity;
        this.channels = channels;
    }

    /**
     * A spec for {@code Model.sequential().add(...)} after a convolutional layer.
     *
     * @param count the number of prototypes
     * @param similarity the similarity
     * @return the spec
     */
    public static LayerSpec spec(int count, Similarity similarity) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                if (input.rank() != 3) {
                    throw new IllegalArgumentException("expects [channels, h, w], got " + input);
                }
                return Shape.of(count);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PrototypeLayer(count, input.dim(0), similarity, random);
            }
        };
    }

    /** @return the prototype vectors {@code [m, C]} */
    public Var prototypes() {
        return prototypes;
    }

    /** @return the similarity */
    public Similarity similarity() {
        return similarity;
    }

    /** @return the number of prototypes */
    public int count() {
        return prototypes.shape().dim(0);
    }

    /** {@code [batch * H * W, C]} rows of the feature map. */
    private Var patches(Var map) {
        Shape s = map.shape();
        int b = s.dim(0);
        int c = s.dim(1);
        int hw = s.dim(2) * s.dim(3);
        return map.reshape(b, c, hw).permute(0, 2, 1).reshape(b * hw, c);
    }

    /**
     * Per-position distances: squared L2, or {@code 1 - cos} for the cosine similarity.
     *
     * @param map the feature map {@code [batch, C, H, W]}
     * @return {@code [batch, H * W, m]}
     */
    public Var distances(Var map) {
        Shape s = map.shape();
        int b = s.dim(0);
        int hw = s.dim(2) * s.dim(3);
        int m = count();
        Var z = patches(map);
        if (similarity == Similarity.COSINE) {
            Var zn = z.normalize(1, 1e-6f);
            Var pn = prototypes.normalize(1, 1e-6f);
            return zn.matmul(pn.transpose()).neg().add(1f).reshape(b, hw, m);
        }
        Var zz = z.square().sum(1, true);
        Var pp = prototypes.square().sum(1, false).reshape(1, m);
        Var d = zz.add(pp).sub(z.matmul(prototypes.transpose()).mul(2f)).relu();
        return d.reshape(b, hw, m);
    }

    /**
     * Per-position similarities.
     *
     * @param map the feature map {@code [batch, C, H, W]}
     * @return {@code [batch, H * W, m]}
     */
    public Var patchSimilarities(Var map) {
        Var d = distances(map);
        if (similarity == Similarity.COSINE) {
            return d.neg().add(1f);
        }
        return d.add(1f).div(d.add(1e-4f)).log();
    }

    @Override
    public Var forward(Var input) {
        return patchSimilarities(input).max(1, false);
    }

    @Override
    public List<Var> parameters() {
        return List.of(prototypes);
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return Shape.of(count());
    }

    @Override
    public int outputSize(int inputSize) {
        return count();
    }

    /**
     * The projection step: each prototype becomes its nearest patch among the feature maps of
     * the images of its own class.
     *
     * @param maps feature maps of training images {@code [n, C, H, W]}
     * @param labels their classes
     * @param prototypeClass the class of each prototype
     */
    public void project(Tensor maps, int[] labels, int[] prototypeClass) {
        Tensor d = distances(Var.of(maps)).value();
        int n = maps.shape().dim(0);
        int hw = maps.shape().dim(2) * maps.shape().dim(3);
        int m = count();
        float[] dist = d.toArray();
        float[] z = maps.reshape(n, channels, hw).toArray();
        float[] proto = prototypes.value().toArray();
        for (int k = 0; k < m; k++) {
            float best = Float.POSITIVE_INFINITY;
            int bi = -1;
            int bp = -1;
            for (int i = 0; i < n; i++) {
                if (labels[i] != prototypeClass[k]) {
                    continue;
                }
                for (int p = 0; p < hw; p++) {
                    float v = dist[(i * hw + p) * m + k];
                    if (v < best) {
                        best = v;
                        bi = i;
                        bp = p;
                    }
                }
            }
            if (bi < 0) {
                continue;
            }
            for (int c = 0; c < channels; c++) {
                proto[k * channels + c] = z[(bi * channels + c) * hw + bp];
            }
        }
        prototypes.assign(Tensor.of(prototypes.shape(), proto));
    }
}
