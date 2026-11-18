package io.github.zzuegg.brainiac.experimental.synth;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.LongFunction;

/**
 * Dataset distillation by condensation: synthesise a tiny training set (one to ten images per
 * class) that trains a fresh network nearly as well as the full set (Wang et al., "Dataset
 * Distillation", 2018, <a href="https://arxiv.org/abs/1811.10959">arXiv:1811.10959</a>).
 * <ul>
 *   <li>{@link Mode#GRADIENT_MATCHING}: dataset condensation (Zhao et al., ICLR 2021,
 *       <a href="https://arxiv.org/abs/2006.05929">arXiv:2006.05929</a>): for a freshly
 *       initialised network, per class, the synthetic images are moved so that the gradient of
 *       the loss they induce matches the gradient of a real batch (layer-wise cosine distance),
 *       differentiating through the gradient ({@link Var#gradients(Var, List, boolean)} with a
 *       graph); the network then takes a few steps on the synthetic set, and a new network is
 *       drawn every outer iteration;</li>
 *   <li>{@link Mode#DISTRIBUTION_MATCHING}: Zhao and Bilen (WACV 2023,
 *       <a href="https://arxiv.org/abs/2110.04181">arXiv:2110.04181</a>): match the mean
 *       embedding of real and synthetic images of each class under randomly initialised
 *       networks, with no inner training loop;</li>
 *   <li>differentiable Siamese augmentation: the same random shift (±1 pixel, wrapping) and
 *       brightness scale is applied to the real and the synthetic batch inside the matching
 *       loss, so gradients reach the synthetic pixels through it.</li>
 * </ul>
 * Images are flat {@code side * side} rows. Networks come from a factory of {@link SimpleMlp}s.
 */
public final class DatasetCondenser {

    /** The matching objective. */
    public enum Mode {
        /** Gradient matching. */
        GRADIENT_MATCHING,
        /** Distribution (mean-embedding) matching. */
        DISTRIBUTION_MATCHING
    }

    private final Mode mode;
    private final int classes;
    private final int perClass;
    private final int side;
    private final boolean augment;
    private final Var images;
    private final Optimizer optimizer;
    private final SplittableRandom random;

    /**
     * Creates a condenser whose synthetic images start as random real images of their class.
     *
     * @param mode the matching objective
     * @param realX real images {@code [n][side * side]}
     * @param realY their classes
     * @param classes the number of classes
     * @param perClass synthetic images per class
     * @param side the image side
     * @param augment whether to use differentiable Siamese augmentation
     * @param lr the Adam rate on the synthetic pixels
     * @param seed the seed
     */
    public DatasetCondenser(Mode mode, float[][] realX, int[] realY, int classes, int perClass, int side,
            boolean augment, float lr, long seed) {
        this.mode = mode;
        this.classes = classes;
        this.perClass = perClass;
        this.side = side;
        this.augment = augment;
        this.random = new SplittableRandom(seed);
        float[] init = new float[classes * perClass * side * side];
        for (int c = 0; c < classes; c++) {
            List<Integer> idx = indicesOf(realY, c);
            for (int k = 0; k < perClass; k++) {
                float[] src = realX[idx.get(random.nextInt(idx.size()))];
                System.arraycopy(src, 0, init, (c * perClass + k) * side * side, side * side);
            }
        }
        this.images = Var.param(Tensor.of(Shape.of(classes * perClass, side * side), init));
        this.optimizer = Optimizer.adam(lr);
    }

    private static List<Integer> indicesOf(int[] y, int c) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < y.length; i++) {
            if (y[i] == c) {
                out.add(i);
            }
        }
        return out;
    }

    /** @return the synthetic images {@code [classes * perClass, side * side]}, class-major */
    public Tensor images() {
        return images.value();
    }

    /** @return the synthetic labels, class-major */
    public int[] labels() {
        int[] y = new int[classes * perClass];
        for (int i = 0; i < y.length; i++) {
            y[i] = i / perClass;
        }
        return y;
    }

    private Tensor shiftIndex(int dx, int dy) {
        float[] idx = new float[side * side];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                idx[y * side + x] = Math.floorMod(y - dy, side) * side + Math.floorMod(x - dx, side);
            }
        }
        return Tensor.of(Shape.of(side * side), idx);
    }

    private Var augmentWith(Var x, Tensor index, float scale) {
        int n = x.shape().dim(0);
        Var shifted = x.gather(1, index);
        return shifted.mul(scale).reshape(n, side * side);
    }

    private static Var crossEntropy(Var logits, int label, int classes) {
        int n = logits.shape().dim(0);
        float[] oh = new float[n * classes];
        for (int i = 0; i < n; i++) {
            oh[i * classes + label] = 1;
        }
        Var picked = logits.mul(Var.of(Tensor.of(Shape.of(n, classes), oh))).sum(1, false);
        return logits.logsumexp(1, false).sub(picked).mean();
    }

    /**
     * One outer iteration: a fresh network, a matching loss summed over classes, one Adam step on
     * the synthetic images (and, for gradient matching, {@code innerSteps} SGD steps of the
     * network on the synthetic set before the next class pass).
     *
     * @param realX real images
     * @param realY their classes
     * @param batch real images per class per iteration
     * @param nets a factory of freshly initialised networks
     * @param innerSteps network steps on the synthetic set between matching passes (gradient matching)
     * @return the matching loss
     */
    public double step(float[][] realX, int[] realY, int batch, LongFunction<SimpleMlp> nets, int innerSteps) {
        SimpleMlp net = nets.apply(random.nextLong());
        double total = 0;
        int passes = mode == Mode.GRADIENT_MATCHING ? Math.max(1, innerSteps) : 1;
        for (int pass = 0; pass < passes; pass++) {
            Var loss = null;
            for (int c = 0; c < classes; c++) {
                List<Integer> idx = indicesOf(realY, c);
                float[][] rb = new float[batch][];
                for (int i = 0; i < batch; i++) {
                    rb[i] = realX[idx.get(random.nextInt(idx.size()))];
                }
                Var real = Var.of(Tensor.of(rb));
                Var syn = images.slice(0, c * perClass, (c + 1) * perClass);
                if (augment) {
                    Tensor index = shiftIndex(random.nextInt(3) - 1, random.nextInt(3) - 1);
                    float scale = (float) (0.8 + 0.4 * random.nextDouble());
                    real = augmentWith(real, index, scale);
                    syn = augmentWith(syn, index, scale);
                }
                Var term = mode == Mode.GRADIENT_MATCHING ? gradientDistance(net, real, syn, c)
                        : embeddingDistance(net, real, syn);
                loss = loss == null ? term : loss.add(term);
            }
            images.zeroGrad();
            loss.backward();
            optimizer.step(List.of(images));
            total += loss.value().item() / passes;
            if (mode == Mode.GRADIENT_MATCHING && pass + 1 < passes) {
                for (Var p : net.parameters()) {
                    p.zeroGrad();
                }
                Var l = null;
                Tensor all = images.value();
                for (int c = 0; c < classes; c++) {
                    Var part = crossEntropy(net.forward(Var.of(all.slice(0, c * perClass, (c + 1) * perClass))), c,
                            classes);
                    l = l == null ? part : l.add(part);
                }
                List<Var> g = Var.gradients(l, net.parameters(), false);
                for (int k = 0; k < g.size(); k++) {
                    Var p = net.parameters().get(k);
                    p.assign(p.value().sub(g.get(k).value().mul(0.05f)));
                }
            }
        }
        return total;
    }

    private Var gradientDistance(SimpleMlp net, Var real, Var syn, int label) {
        List<Var> params = net.parameters();
        List<Var> gReal = Var.gradients(crossEntropy(net.forward(real), label, classes), params, false);
        List<Var> gSyn = Var.gradients(crossEntropy(net.forward(syn), label, classes), params, true);
        Var d = null;
        for (int k = 0; k < params.size(); k++) {
            Tensor gr = gReal.get(k).value();
            Var gs = gSyn.get(k);
            Var dot = gs.mul(Var.of(gr)).sum();
            Var norm = gs.square().sum().add(1e-8f).sqrt().mul((float) Math.sqrt(gr.square().sum().item() + 1e-8));
            Var term = Var.of(Tensor.scalar(1f)).sub(dot.div(norm));
            d = d == null ? term : d.add(term);
        }
        return d;
    }

    private static Var embeddingDistance(SimpleMlp net, Var real, Var syn) {
        Var er = net.layer(0, real).mean(0, false);
        Var es = net.layer(0, syn).mean(0, false);
        return es.sub(Var.of(er.value())).square().sum();
    }

    @Override
    public String toString() {
        return "DatasetCondenser(" + mode + ", " + classes + " x " + perClass + ")";
    }
}
