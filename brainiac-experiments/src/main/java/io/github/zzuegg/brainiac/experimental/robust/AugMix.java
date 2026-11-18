package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.DoubleFunction;

/**
 * AugMix (Hendrycks et al., ICLR 2020, <a href="https://arxiv.org/abs/1912.02781">arXiv:1912.02781</a>):
 * {@code width} augmentation chains of depth 1 to 3, each a random composition of
 * operations at a fixed severity, are mixed with Dirichlet(1) weights and blended with the
 * original by a Beta(1, 1) weight, {@code x_aug = m x + (1 - m) Σ w_i chain_i(x)}, which keeps
 * the result close to the data manifold. Training adds the Jensen-Shannon consistency loss
 * {@link #jensenShannon} between the clean and two AugMix views. It is a core
 * {@link Augmentation}, and uses the same strength-parametrised operations as
 * {@link TrivialAugment} (by default {@link TrivialAugment#geometric}, which holds out the
 * photometric test corruptions).
 */
public final class AugMix implements Augmentation {

    private final List<DoubleFunction<Augmentation>> operations;
    private final int width;
    private final double severity;
    private final int sampleSize;

    /**
     * Creates AugMix.
     *
     * @param operations operations by strength
     * @param width chains mixed, 3 in the paper
     * @param severity the operations' strength in {@code [0, 1]}
     * @param sampleSize the values of one image, {@code channels * height * width}
     */
    public AugMix(List<DoubleFunction<Augmentation>> operations, int width, double severity, int sampleSize) {
        this.operations = List.copyOf(operations);
        this.width = width;
        this.severity = severity;
        this.sampleSize = sampleSize;
    }

    @Override
    public void apply(float[] features, int offset, SplittableRandom random) {
        mix(features, offset, sampleSize, random);
    }

    /**
     * Mixes one image in place.
     *
     * @param features the array holding the image
     * @param offset where it starts
     * @param size its length
     * @param random the random source
     */
    public void mix(float[] features, int offset, int size, SplittableRandom random) {
        double[] w = new double[width];
        double z = 0;
        for (int i = 0; i < width; i++) {
            w[i] = -Math.log(Math.max(random.nextDouble(), 1e-300));
            z += w[i];
        }
        double m = random.nextDouble();
        float[] mixed = new float[size];
        float[] chain = new float[size];
        for (int i = 0; i < width; i++) {
            System.arraycopy(features, offset, chain, 0, size);
            int depth = 1 + random.nextInt(3);
            for (int k = 0; k < depth; k++) {
                operations.get(random.nextInt(operations.size())).apply(severity).apply(chain, 0, random);
            }
            for (int j = 0; j < size; j++) {
                mixed[j] += (float) (w[i] / z * chain[j]);
            }
        }
        for (int j = 0; j < size; j++) {
            features[offset + j] = (float) (m * features[offset + j] + (1 - m) * mixed[j]);
        }
    }

    /**
     * An AugMix view of a batch of flat images.
     *
     * @param batch the images, not changed
     * @param random the random source
     * @return mixed copies
     */
    public float[][] view(float[][] batch, SplittableRandom random) {
        float[][] out = new float[batch.length][];
        for (int i = 0; i < batch.length; i++) {
            out[i] = batch[i].clone();
            mix(out[i], 0, out[i].length, random);
        }
        return out;
    }

    /**
     * The Jensen-Shannon consistency loss between the predictions of three views,
     * {@code (KL(p_c ‖ M) + KL(p_1 ‖ M) + KL(p_2 ‖ M)) / 3} with {@code M} their mean,
     * averaged over the batch.
     *
     * @param clean logits of the clean view
     * @param aug1 logits of the first AugMix view
     * @param aug2 logits of the second
     * @return the loss
     */
    public static Var jensenShannon(Var clean, Var aug1, Var aug2) {
        Var pc = clean.softmax(1);
        Var p1 = aug1.softmax(1);
        Var p2 = aug2.softmax(1);
        Var logM = pc.add(p1).add(p2).mul(1f / 3).clamp(1e-7f, 1f).log();
        int rows = clean.shape().dim(0);
        Var kl = kl(pc, logM).add(kl(p1, logM)).add(kl(p2, logM));
        return kl.mul(1f / (3 * rows));
    }

    private static Var kl(Var p, Var logM) {
        return p.mul(p.clamp(1e-7f, 1f).log().sub(logM)).sum();
    }
}
