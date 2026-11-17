package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.MultiScaleVqTokenizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * VAR, next-scale prediction (Tian et al., <a href="https://arxiv.org/abs/2404.02905">2404.02905</a>),
 * on the 8x8 {@link ToyImages}: a {@link MultiScaleVqTokenizer} describes each image by
 * residual token maps of 1x1, 2x2 and 4x4 tiles, and a network predicts every token of a
 * scale at once from the reconstruction of the coarser scales. The network is the token MLP
 * of {@link MaskGit}, made block-causal: while it predicts scale {@code k}, the inputs of
 * finer scales are zero. A sample takes three network calls, one per scale.
 */
@SuppressWarnings("exports")
public final class NextScalePrediction {
    private final MultiScaleVqTokenizer tokenizer;
    private final int[] scales;
    private final int length;
    private final int vocab;
    private final int dim;
    private final Model net;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private final SplittableRandom random;
    private final double reconstructionError;

    private NextScalePrediction(int hidden, int patch, int[] scales, int codes, long seed) {
        tokenizer = MultiScaleVqTokenizer.of(ToyImages.SIDE, patch, scales, codes, seed);
        List<float[]> all = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            all.addAll(ToyImages.all(c));
        }
        reconstructionError = tokenizer.fit(all, 10);
        this.scales = tokenizer.scales();
        length = tokenizer.sequenceLength();
        vocab = tokenizer.vocabulary();
        dim = tokenizer.dim();
        net = Model.sequential().input(length * dim + scales.length).dense(hidden, Activation.SILU)
                .dense(hidden, Activation.SILU).dense(length * vocab).seed(seed).build();
        random = new SplittableRandom(seed + 1);
    }

    /**
     * Creates and fits the tokenizer and an untrained predictor.
     *
     * @param hidden the hidden width of the predictor
     * @param seed the seed
     * @return the experiment
     */
    public static NextScalePrediction of(int hidden, long seed) {
        return new NextScalePrediction(hidden, 2, new int[] {1, 2, 4}, 32, seed);
    }

    /**
     * Creates and fits a tokenizer of chosen scales and an untrained predictor.
     *
     * @param hidden the hidden width of the predictor
     * @param patch the tile side of the finest scale
     * @param scales the token map sizes, the last {@code 8 / patch}
     * @param codes codes per scale
     * @param seed the seed
     * @return the experiment
     */
    public static NextScalePrediction of(int hidden, int patch, int[] scales, int codes, long seed) {
        return new NextScalePrediction(hidden, patch, scales, codes, seed);
    }

    /** @return the tokenizer's mean squared reconstruction error per pixel on the data set */
    public double reconstructionError() {
        return reconstructionError;
    }

    /** @return the number of network calls per sample */
    public int steps() {
        return scales.length;
    }

    private float[] input(int[][] tokens, int scale) {
        float[] ctx = tokenizer.contextInputs(tokens, scale + 1);
        float[] in = new float[length * dim + scales.length];
        System.arraycopy(ctx, 0, in, 0, ctx.length);
        in[length * dim + scale] = 1;
        return in;
    }

    /**
     * Trains the predictor: per image a random scale, the context of the coarser scales, and
     * the cross-entropy of all tokens of that scale.
     *
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the final loss
     */
    public float train(int steps, int batch) {
        float last = 0;
        int width = length * dim + scales.length;
        for (int s = 0; s < steps; s++) {
            float[] in = new float[batch * width];
            float[] weight = new float[batch * length * vocab];
            for (int i = 0; i < batch; i++) {
                int[][] tokens = tokenizer.encode(ToyImages.sample(random.nextInt(ToyImages.CLASSES), random));
                int k = random.nextInt(scales.length);
                System.arraycopy(input(tokens, k), 0, in, i * width, width);
                for (int j = 0; j < tokens[k].length; j++) {
                    int pos = tokenizer.offset(k) + j;
                    weight[(i * length + pos) * vocab + tokens[k][j]] = 1f / (batch * tokens[k].length);
                }
            }
            Var logits = net.forward(Var.of(Tensor.of(Shape.of(batch, width), in)), true).reshape(batch * length, vocab);
            Var loss = Activation.LOG_SOFTMAX.forward(logits).mul(Tensor.of(Shape.of(batch * length, vocab), weight))
                    .sum().neg();
            last = loss.value().item();
            loss.backward();
            optimizer.step(net.parameters());
        }
        return last;
    }

    /**
     * Generates images coarse to fine, one network call per scale. The decoded pixels are
     * rounded to the data's two values.
     *
     * @param count the number of images
     * @param temperature the sampling temperature
     * @param seed the seed
     * @return the decoded images
     */
    public List<float[]> sample(int count, float temperature, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int width = length * dim + scales.length;
        int[][][] tokens = new int[count][scales.length][];
        for (int k = 0; k < scales.length; k++) {
            float[] in = new float[count * width];
            for (int i = 0; i < count; i++) {
                System.arraycopy(input(tokens[i], k), 0, in, i * width, width);
            }
            float[] logits = net.forward(Var.of(Tensor.of(Shape.of(count, width), in)), false).value().toArray();
            for (int i = 0; i < count; i++) {
                tokens[i][k] = new int[scales[k] * scales[k]];
                for (int j = 0; j < tokens[i][k].length; j++) {
                    int pos = tokenizer.offset(k) + j;
                    tokens[i][k][j] = TokenModels.draw(logits, (i * length + pos) * vocab, vocab, temperature, r);
                }
            }
        }
        List<float[]> out = new ArrayList<>();
        for (int[][] t : tokens) {
            float[] img = tokenizer.decode(t);
            for (int i = 0; i < img.length; i++) {
                img[i] = img[i] >= 0 ? 1 : -1;   // the data is binary; this removes the residual codebook error
            }
            out.add(img);
        }
        return out;
    }

    /**
     * Scores decoded images like {@link MaskGit#score}.
     *
     * @param images generated images
     * @return {@code {distance per pixel, real share, distinct real images, classes}}
     */
    public static double[] score(List<float[]> images) {
        return TokenModels.score(images);
    }
}
