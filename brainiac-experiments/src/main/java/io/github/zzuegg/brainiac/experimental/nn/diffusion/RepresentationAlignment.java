package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * REPA, representation alignment (Yu et al., "Representation Alignment for Generation:
 * Training Diffusion Transformers Is Easier Than You Think",
 * <a href="https://arxiv.org/abs/2410.06940">2410.06940</a>, ICLR 2025 oral): an auxiliary
 * loss that makes an early hidden state of the denoiser, seen on the <em>noisy</em> input,
 * predict (through a small projector MLP) the features a frozen self-supervised encoder
 * computes from the <em>clean</em> input. The loss is the negative cosine similarity,
 * added to the diffusion loss with weight {@code lambda}.
 *
 * <p>Why: a denoiser has to learn a good representation of its data before it can denoise
 * well, and learns it slowly from the denoising loss alone. Handing it the representation
 * of a pretrained encoder (DINOv2 in the paper; any frozen encoder from
 * {@link SelfSupervisedLosses} here) speeds training up many times. The projector is
 * thrown away after training; sampling is unchanged.
 */
public final class RepresentationAlignment {
    private final Dense first;
    private final Dense second;

    private RepresentationAlignment(int hiddenSize, int featureSize, int width, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        first = GenBlocks.linear(hiddenSize, width, random);
        second = GenBlocks.linear(width, featureSize, random);
    }

    /**
     * Creates the projector.
     *
     * @param hiddenSize the denoiser's hidden size
     * @param featureSize the encoder's feature size
     * @param width the projector's hidden width
     * @param seed the initialization seed
     * @return the alignment loss
     */
    public static RepresentationAlignment of(int hiddenSize, int featureSize, int width, long seed) {
        return new RepresentationAlignment(hiddenSize, featureSize, width, seed);
    }

    /**
     * The alignment loss.
     *
     * @param hidden the denoiser's hidden states {@code [batch, hiddenSize]}
     * @param features the frozen encoder's features of the clean inputs {@code [batch, featureSize]}
     * @return the mean negative cosine similarity, a scalar in [-1, 1]
     */
    public Var loss(Var hidden, Tensor features) {
        Var projected = second.forward(GenBlocks.silu(first.forward(hidden)));
        return SelfSupervisedLosses.normalize(projected).mul(SelfSupervisedLosses.normalize(Var.of(features)))
                .sum(-1, false).mean().neg();
    }

    /**
     * Diffusion loss plus alignment: core {@link Diffusion}'s training loss for the network,
     * computed in the same forward pass that yields the hidden state, plus {@code lambda}
     * times {@link #loss}.
     *
     * @param diffusion the core diffusion (schedule and prediction type)
     * @param net the denoiser (unconditional)
     * @param layer which hidden state to align (see {@link VectorConditionedDenoiser#applyWithHidden})
     * @param x0 clean data {@code [batch, dimension]}
     * @param encoder the frozen encoder, clean data to features
     * @param lambda the alignment weight (0.5 in the paper)
     * @param random the source of steps and noise
     * @return {@code {total loss, diffusion loss, alignment loss}}
     */
    public Var[] diffusionLoss(Diffusion diffusion, VectorConditionedDenoiser net, int layer, Tensor x0,
            UnaryOperator<Tensor> encoder, float lambda, RandomGenerator random) {
        NoiseSchedule schedule = diffusion.schedule();
        int batch = x0.shape().dim(0);
        int[] steps = new int[batch];
        float[] time = new float[batch];
        float[] a = new float[batch];
        float[] b = new float[batch];
        for (int i = 0; i < batch; i++) {
            steps[i] = random.nextInt(schedule.steps());
            time[i] = steps[i];
            a[i] = (float) Math.sqrt(schedule.alphaBar(steps[i]));
            b[i] = (float) Math.sqrt(1 - schedule.alphaBar(steps[i]));
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random);
        Tensor xt = diffusion.noisy(x0, noise, steps);
        Tensor target = switch (diffusion.prediction()) {
            case NOISE -> noise;
            case SAMPLE -> x0;
            case VELOCITY -> noise.mul(column(a)).sub(x0.mul(column(b)));
        };
        Var[] out = net.applyWithHidden(Var.of(xt), Tensor.of(Shape.of(batch), time), null, layer);
        Var denoise = out[0].sub(Var.of(target)).square().mean();
        Var align = loss(out[1], encoder.apply(x0));
        return new Var[] {denoise.add(align.mul(lambda)), denoise, align};
    }

    private static Tensor column(float[] v) {
        return Tensor.of(Shape.of(v.length, 1), v.clone());
    }

    /** @return the projector's parameters, to train with the denoiser */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(first.parameters());
        p.addAll(second.parameters());
        return p;
    }
}
