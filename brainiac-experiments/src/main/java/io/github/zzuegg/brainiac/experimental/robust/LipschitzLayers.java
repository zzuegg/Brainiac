package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * 1-Lipschitz layers (in the {@code ℓ2} norm), from which networks are built whose
 * classification margin {@code m = z_y - max_(j != y) z_j} certifies robustness within the
 * {@code ℓ2} radius {@code m / √2}, at no extra cost at test time.
 *
 * <ul>
 *   <li>{@link GroupSort}: sorts pairs of units, {@code (a, b) -> (max, min)}; a permutation
 *       on each input region, so it is 1-Lipschitz and gradient-norm preserving, unlike ReLU,
 *       which loses expressivity under norm constraints (Anil, Lucas and Grosse, ICML 2019,
 *       <a href="https://arxiv.org/abs/1811.05381">arXiv:1811.05381</a>).</li>
 *   <li>{@link BjorckDense}: the weight is made orthonormal on every forward pass by Björck
 *       iterations {@code W <- W (3/2 I - 1/2 Wᵀ W)} from a Frobenius-scaled start, so the
 *       layer is an isometry on its row space (same paper).</li>
 *   <li>{@link AolDense}: almost-orthogonal layers rescale the weight's input dimensions by
 *       {@code d_i = (Σ_j |W Wᵀ|_ij)^(-1/2)}, which makes {@code x ↦ x diag(d) W} 1-Lipschitz
 *       without any iteration (Prach and Lampert, ECCV 2022,
 *       <a href="https://arxiv.org/abs/2208.03160">arXiv:2208.03160</a>).</li>
 *   <li>{@link AolConv}: AOL for a convolution: {@code Σ_j |KᵀK|_ij} becomes the sum of the
 *       absolute cross-correlation of the kernel with itself over channel pairs and offsets,
 *       one factor per input channel.</li>
 * </ul>
 *
 * <p>All are core {@link Layer}s with {@link LayerSpec}s for {@code Model.sequential()}.
 */
public final class LipschitzLayers {

    private LipschitzLayers() {
    }

    /** @return a spec for {@link GroupSort} */
    public static LayerSpec groupSort() {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new GroupSort();
            }
        };
    }

    /**
     * @param units the output width
     * @param iterations Björck iterations per forward pass
     * @return a spec for {@link BjorckDense}
     */
    public static LayerSpec bjorckDense(int units, int iterations) {
        return denseSpec(units, (in, r) -> new BjorckDense(in, units, iterations, r));
    }

    /**
     * @param units the output width
     * @return a spec for {@link AolDense}
     */
    public static LayerSpec aolDense(int units) {
        return denseSpec(units, (in, r) -> new AolDense(in, units, r));
    }

    /**
     * @param filters output channels
     * @param kernel the odd kernel size (same padding, stride 1)
     * @return a spec for {@link AolConv}
     */
    public static LayerSpec aolConv(int filters, int kernel) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(filters, input.dim(1), input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new AolConv(input.dim(0), filters, kernel, random);
            }
        };
    }

    private interface DenseFactory {
        Layer create(int in, SplittableRandom random);
    }

    private static LayerSpec denseSpec(int units, DenseFactory f) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return f.create(input.dim(input.rank() - 1), random);
            }
        };
    }

    /**
     * The certified {@code ℓ2} radius of a 1-Lipschitz classifier's logits.
     *
     * @param logits {@code [n, classes]}
     * @param labels the true classes
     * @return per input the radius {@code margin / √2}, negative when misclassified
     */
    public static double[] certifiedRadius(Tensor logits, int[] labels) {
        float[] z = logits.toArray();
        int k = logits.shape().dim(1);
        double[] out = new double[labels.length];
        for (int i = 0; i < labels.length; i++) {
            double other = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) {
                if (j != labels[i]) {
                    other = Math.max(other, z[i * k + j]);
                }
            }
            out[i] = (z[i * k + labels[i]] - other) / Math.sqrt(2);
        }
        return out;
    }

    /** Pairwise sort of units: {@code (a, b) -> (max(a, b), min(a, b))} for each pair. */
    public static final class GroupSort implements Layer {

        /** Creates the activation. */
        public GroupSort() {
        }

        @Override
        public Var forward(Var input) {
            int[] d = input.shape().toArray();
            int last = d[d.length - 1];
            if (last % 2 != 0) {
                throw new IllegalArgumentException("GroupSort needs an even width, got " + last);
            }
            int rows = Math.toIntExact(input.value().size() / last);
            Var pairs = input.reshape(rows, last / 2, 2);
            Var a = pairs.slice(2, 0, 1);
            Var b = pairs.slice(2, 1, 2);
            return Var.concat(2, a.maximum(b), a.minimum(b)).reshape(d);
        }

        @Override
        public int outputSize(int inputSize) {
            return inputSize;
        }

        @Override
        public String toString() {
            return "GroupSort";
        }
    }

    /** A dense layer whose weight is orthonormalised by Björck iterations on every pass. */
    public static final class BjorckDense implements Layer {
        private final Var weight;
        private final Var bias;
        private final int iterations;

        /**
         * Creates the layer.
         *
         * @param in the input width
         * @param out the output width
         * @param iterations Björck iterations per forward pass
         * @param random the initialisation source
         */
        public BjorckDense(int in, int out, int iterations, SplittableRandom random) {
            this.weight = Var.param(Init.XAVIER.create(Shape.of(in, out), in, out, random));
            this.bias = Var.param(Tensor.zeros(Shape.of(out)));
            this.iterations = iterations;
        }

        /** @return the orthonormalised weight {@code [in, out]} */
        public Var orthonormal() {
            Var w = weight.div(weight.square().sum().add(1e-12f).sqrt());
            boolean tall = weight.shape().dim(0) >= weight.shape().dim(1);
            for (int i = 0; i < iterations; i++) {
                if (tall) {
                    w = w.mul(1.5f).sub(w.matmul(w.transpose().matmul(w)).mul(0.5f));
                } else {
                    w = w.mul(1.5f).sub(w.matmul(w.transpose()).matmul(w).mul(0.5f));
                }
            }
            return w;
        }

        @Override
        public Var forward(Var input) {
            return input.matmul(orthonormal()).add(bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return weight.shape().dim(1);
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }
    }

    /** A dense layer with AOL rescaling. */
    public static final class AolDense implements Layer {
        private final Var weight;
        private final Var bias;

        /**
         * Creates the layer.
         *
         * @param in the input width
         * @param out the output width
         * @param random the initialisation source
         */
        public AolDense(int in, int out, SplittableRandom random) {
            this.weight = Var.param(Init.XAVIER.create(Shape.of(in, out), in, out, random));
            this.bias = Var.param(Tensor.zeros(Shape.of(out)));
        }

        /** @return the rescaled weight {@code diag(d) W} */
        public Var rescaled() {
            int in = weight.shape().dim(0);
            Var gram = weight.matmul(weight.transpose()).abs().sum(1, true);
            return weight.mul(gram.add(1e-12f).pow(-0.5f).reshape(in, 1));
        }

        @Override
        public Var forward(Var input) {
            return input.matmul(rescaled()).add(bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return weight.shape().dim(1);
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }
    }

    /** A same-padded, stride-1 convolution with AOL rescaling per input channel. */
    public static final class AolConv implements Layer {
        private final Var kernel;
        private final Var bias;
        private final int size;

        /**
         * Creates the layer.
         *
         * @param channels input channels
         * @param filters output channels
         * @param size the odd kernel size
         * @param random the initialisation source
         */
        public AolConv(int channels, int filters, int size, SplittableRandom random) {
            if (size % 2 == 0) {
                throw new IllegalArgumentException("kernel size must be odd: " + size);
            }
            int fanIn = channels * size * size;
            this.kernel = Var.param(Init.HE.create(Shape.of(filters, channels, size, size), fanIn, filters * size * size,
                    random));
            this.bias = Var.param(Tensor.zeros(Shape.of(filters)));
            this.size = size;
        }

        /** @return the rescaled kernel */
        public Var rescaled() {
            int c = kernel.shape().dim(1);
            int p = size - 1;
            // full cross-correlation of the kernel with itself, summed over output channels:
            // [c, filters, k, k] (*) [c', filters, k, k] -> [c, c', 2k-1, 2k-1]
            Var asImages = kernel.permute(1, 0, 2, 3);
            Var corr = Functional.conv(asImages, asImages, null, new int[] {1, 1}, new int[] {p, p},
                    new int[] {p, p}, new int[] {1, 1}, 1);
            Var d = corr.abs().sum(3, false).sum(2, false).sum(1, false).add(1e-12f).pow(-0.5f);
            return kernel.mul(d.reshape(1, c, 1, 1));
        }

        @Override
        public Var forward(Var input) {
            int p = size / 2;
            return Functional.conv(input, rescaled(), bias, new int[] {1, 1}, new int[] {p, p}, new int[] {p, p},
                    new int[] {1, 1}, 1);
        }

        @Override
        public List<Var> parameters() {
            return List.of(kernel, bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return inputSize / kernel.shape().dim(1) * kernel.shape().dim(0);
        }
    }
}
