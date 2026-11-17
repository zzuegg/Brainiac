package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConditionalDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.RectifiedFlow;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Flow matching experiments (Lipman et al., <a href="https://arxiv.org/abs/2210.02747">2210.02747</a>;
 * Rectified Flow, <a href="https://arxiv.org/abs/2209.03003">2209.03003</a>) on top of core
 * {@link FlowMatching}: a class-conditional velocity network trained with label dropout,
 * sampled with Euler or Heun steps and classifier-free guidance, and straightened by
 * reflow so that one Euler step approaches the many-step result.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class RectifiedFlowExperiment {
    private final ConditionalDenoiser net;
    private final Optimizer optimizer;
    private final float labelDropout;
    private final SplittableRandom random;

    private RectifiedFlowExperiment(ConditionalDenoiser net, float learningRate, float labelDropout, long seed) {
        this.net = net;
        this.optimizer = Optimizer.adam(learningRate);
        this.labelDropout = labelDropout;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates the experiment.
     *
     * @param net the velocity network (time scale 1)
     * @param learningRate the Adam learning rate
     * @param labelDropout the share of labels dropped while training, for guidance
     * @param seed the seed for noise and times
     * @return the experiment
     */
    public static RectifiedFlowExperiment of(ConditionalDenoiser net, float learningRate, float labelDropout,
            long seed) {
        return new RectifiedFlowExperiment(net, learningRate, labelDropout, seed);
    }

    /**
     * Changes the learning rate, for example to anneal it.
     *
     * @param learningRate the new rate
     */
    public void learningRate(float learningRate) {
        optimizer.learningRate(learningRate);
    }

    private float step(Var loss) {
        float value = loss.value().item();
        loss.backward();
        Nets.clip(net.parameters(), 1f);
        optimizer.step(net.parameters());
        return value;
    }

    /**
     * One step of the core flow-matching loss, with label dropout.
     *
     * @param data data samples
     * @param labels their labels, or -1
     * @return the loss before the step
     */
    public float trainStep(float[][] data, int[] labels) {
        int[] l = ConditionalDenoiser.dropLabels(labels, labelDropout, random);
        return step(FlowMatching.loss(net.bind(l), Nets.matrix(data), random));
    }

    /**
     * One reflow step on (noise, sample) couplings.
     *
     * @param noise the noise ends
     * @param samples the data ends, produced by the model from that noise
     * @param labels the labels, or -1
     * @return the loss before the step
     */
    public float reflowStep(float[][] noise, float[][] samples, int[] labels) {
        return step(RectifiedFlow.couplingLoss(net.bind(labels), Nets.matrix(noise), Nets.matrix(samples), random));
    }

    /**
     * Draws the starting noise for a sampler.
     *
     * @param n the number of samples
     * @param seed the seed
     * @return standard normal points
     */
    public float[][] noise(int n, long seed) {
        return Nets.rows(Nets.gaussian(n, net.dimension(), new SplittableRandom(seed)));
    }

    /**
     * Euler sampling with guidance.
     *
     * @param start the noise
     * @param labels labels, or -1
     * @param steps the number of steps
     * @param guidance the guidance weight, 1 for none
     * @param trajectory receives the state after every step, or {@code null}
     * @return the samples
     */
    public float[][] sampleEuler(float[][] start, int[] labels, int steps, float guidance, List<float[][]> trajectory) {
        List<Tensor> t = trajectory == null ? null : new ArrayList<>();
        Tensor x = RectifiedFlow.sampleEuler(net.guided(labels, guidance), Nets.matrix(start), steps, t);
        copy(t, trajectory);
        return Nets.rows(x);
    }

    /**
     * Heun sampling with guidance.
     *
     * @param start the noise
     * @param labels labels, or -1
     * @param steps the number of steps
     * @param guidance the guidance weight, 1 for none
     * @param trajectory receives the state after every step, or {@code null}
     * @return the samples
     */
    public float[][] sampleHeun(float[][] start, int[] labels, int steps, float guidance, List<float[][]> trajectory) {
        List<Tensor> t = trajectory == null ? null : new ArrayList<>();
        Tensor x = RectifiedFlow.sampleHeun(net.guided(labels, guidance), Nets.matrix(start), steps, t);
        copy(t, trajectory);
        return Nets.rows(x);
    }

    private static void copy(List<Tensor> from, List<float[][]> to) {
        if (from != null) {
            for (Tensor x : from) {
                to.add(Nets.rows(x));
            }
        }
    }
}
