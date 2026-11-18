package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A prototypical-part network (Chen et al., NeurIPS 2019,
 * <a href="https://arxiv.org/abs/1806.10574">arXiv:1806.10574</a>): a convolutional backbone,
 * a {@link PrototypeLayer} with a fixed number of prototypes per class and a linear last layer,
 * so that a decision reads "this patch looks like that prototype, which is a patch of a training
 * image of class k".
 *
 * <p>Training follows the paper's three stages: {@link #train} jointly optimises the backbone and
 * prototypes on cross-entropy plus the cluster cost (every image close to some prototype of its
 * class) minus the separation cost (far from prototypes of other classes); {@link #project}
 * replaces each prototype by its nearest training patch of its class; {@link #fitLastLayer}
 * refits only the last layer with an L1 penalty on the connections from prototypes to the
 * classes they do not belong to.
 */
public final class ProtoPNet {

    private final Model backbone;
    private final PrototypeLayer layer;
    private final Var last;
    private final int classes;
    private final int[] prototypeClass;

    /**
     * Creates the network.
     *
     * @param backbone a model whose output is a feature map {@code [C, H, W]}
     * @param classes the number of classes
     * @param perClass prototypes per class
     * @param similarity the prototype similarity
     * @param seed the seed of the prototypes
     */
    public ProtoPNet(Model backbone, int classes, int perClass, PrototypeLayer.Similarity similarity, long seed) {
        this.backbone = backbone;
        this.classes = classes;
        int m = classes * perClass;
        this.layer = new PrototypeLayer(m, backbone.outputShape().dim(0), similarity, new SplittableRandom(seed));
        this.prototypeClass = new int[m];
        float[] w = new float[m * classes];
        for (int k = 0; k < m; k++) {
            prototypeClass[k] = k / perClass;
            for (int c = 0; c < classes; c++) {
                w[k * classes + c] = c == prototypeClass[k] ? 1f : -0.5f;
            }
        }
        this.last = Var.param(Tensor.of(Shape.of(m, classes), w));
    }

    /** @return the backbone */
    public Model backbone() {
        return backbone;
    }

    /** @return the prototype layer */
    public PrototypeLayer prototypes() {
        return layer;
    }

    /** @return the class each prototype belongs to */
    public int[] prototypeClass() {
        return prototypeClass.clone();
    }

    /**
     * The class logits.
     *
     * @param images the inputs
     * @param training whether the backbone is in training mode
     * @return {@code [batch, classes]}
     */
    public Var logits(Var images, boolean training) {
        return layer.forward(backbone.forward(images, training)).matmul(last);
    }

    private Tensor classMask(Tensor labels) {
        int b = labels.shape().dim(0);
        float[] y = labels.toArray();
        int m = prototypeClass.length;
        float[] mask = new float[b * m];
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < m; k++) {
                mask[i * m + k] = prototypeClass[k] == (int) y[i] ? 1f : 0f;
            }
        }
        return Tensor.of(Shape.of(b, m), mask);
    }

    /**
     * The joint stage: cross-entropy plus {@code 0.8} times the cluster cost minus {@code 0.08}
     * times the separation cost, by Adam on random batches.
     *
     * @param images the training inputs
     * @param labels their classes as floats {@code [n]}
     * @param steps optimisation steps
     * @param learningRate the Adam step
     * @param batch the batch size
     * @param seed the batch seed
     */
    public void train(Tensor images, Tensor labels, int steps, float learningRate, int batch, long seed) {
        List<Var> params = new ArrayList<>(backbone.parameters());
        params.add(layer.prototypes());
        params.add(last);
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int n = images.shape().dim(0);
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(n - batch + 1);
            Tensor x = images.slice(0, from, from + batch);
            Tensor y = labels.slice(0, from, from + batch);
            Var map = backbone.forward(Var.of(x), true);
            Var dist = layer.distances(map);
            Var minDist = dist.neg().max(1, false).neg();
            Var logits = layer.forward(map).matmul(last);
            Tensor mask = classMask(y);
            Tensor big = mask.neg().add(1f).mul(1e4f);
            Var cluster = minDist.add(big).neg().max(1, false).neg().mean();
            Var separation = minDist.add(mask.mul(1e4f)).neg().max(1, false).neg().mean();
            Functional.crossEntropy(logits, y).add(cluster.mul(0.8f)).sub(separation.mul(0.08f)).backward();
            opt.step(params);
        }
    }

    /**
     * The projection stage.
     *
     * @param images training inputs
     * @param labels their classes
     */
    public void project(Tensor images, int[] labels) {
        Tensor maps = Var.noGrad(() -> backbone.forward(Var.of(images), false)).value();
        layer.project(maps, labels, prototypeClass);
    }

    /**
     * The last-layer stage: cross-entropy plus {@code l1} times the absolute weights from
     * prototypes to classes other than their own, only the last layer trained.
     *
     * @param images training inputs
     * @param labels their classes as floats
     * @param steps steps
     * @param learningRate the Adam step
     * @param l1 the sparsity weight
     */
    public void fitLastLayer(Tensor images, Tensor labels, int steps, float learningRate, float l1) {
        Tensor sims = Var.noGrad(() -> layer.forward(backbone.forward(Var.of(images), false))).value();
        int m = prototypeClass.length;
        float[] off = new float[m * classes];
        for (int k = 0; k < m; k++) {
            for (int c = 0; c < classes; c++) {
                off[k * classes + c] = c == prototypeClass[k] ? 0f : 1f;
            }
        }
        Tensor offMask = Tensor.of(Shape.of(m, classes), off);
        Optimizer opt = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            Functional.crossEntropy(Var.of(sims).matmul(last), labels).add(last.mul(offMask).abs().sum().mul(l1))
                    .backward();
            opt.step(List.of(last));
        }
    }

    /**
     * @param images inputs
     * @return the predicted classes
     */
    public int[] predict(Tensor images) {
        Tensor logits = Var.noGrad(() -> logits(Var.of(images), false)).value();
        Tensor arg = logits.argmax(1);
        float[] a = arg.toArray();
        int[] out = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (int) a[i];
        }
        return out;
    }

    /**
     * The most activated prototype of each image and the feature-map position where it
     * matches best: the "this patch looks like that prototype" of the explanation.
     *
     * @param images inputs
     * @return {@code [n][2]}: prototype index, position index in the {@code H * W} map
     */
    public int[][] topPatches(Tensor images) {
        Tensor sims = Var.noGrad(() -> layer.patchSimilarities(backbone.forward(Var.of(images), false))).value();
        int n = sims.shape().dim(0);
        int hw = sims.shape().dim(1);
        int m = sims.shape().dim(2);
        float[] v = sims.toArray();
        int[][] out = new int[n][2];
        for (int i = 0; i < n; i++) {
            float best = Float.NEGATIVE_INFINITY;
            for (int p = 0; p < hw; p++) {
                for (int k = 0; k < m; k++) {
                    float s = v[(i * hw + p) * m + k];
                    if (s > best) {
                        best = s;
                        out[i][0] = k;
                        out[i][1] = p;
                    }
                }
            }
        }
        return out;
    }
}
