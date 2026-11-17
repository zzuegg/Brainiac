package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Running-statistics normalization of inputs (RSNorm in SimBa, Lee et al.,
 * <a href="https://arxiv.org/abs/2410.09754">2410.09754</a>): every training forward pass
 * folds its batch into a running mean and variance per feature (Chan et al.'s parallel
 * update), and every pass divides {@code x - mean} by the running standard deviation. RL
 * observations change scale as the policy explores; a network that sees them standardized
 * trains more stably. No parameters; the statistics are {@link #buffers()}.
 */
public final class RunningObservationNorm implements Layer {
    private final int features;
    private final Var mean;
    private final Var variance;
    private double count;

    /**
     * Creates the layer.
     *
     * @param features the input size
     */
    public RunningObservationNorm(int features) {
        this.features = features;
        this.mean = Var.of(Tensor.zeros(Shape.of(features)));
        this.variance = Var.of(Tensor.full(Shape.of(features), 1f));
    }

    /** @return a spec for {@code Model.sequential().add(...)} */
    public static LayerSpec spec() {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new RunningObservationNorm(input.dim(-1));
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (training) {
            update(input.value());
        }
        Tensor std = variance.value().add(1e-6f).sqrt();
        return input.sub(mean.value()).div(std);
    }

    private void update(Tensor batch) {
        int rows = Math.toIntExact(batch.size() / features);
        Tensor x = batch.reshape(Shape.of(rows, features));
        Tensor bMean = x.mean(0, false);
        Tensor bVar = x.sub(bMean).square().mean(0, false);
        double total = count + rows;
        Tensor delta = bMean.sub(mean.value());
        Tensor newMean = mean.value().add(delta.mul((float) (rows / total)));
        Tensor m2 = variance.value().mul((float) count).add(bVar.mul(rows))
                .add(delta.square().mul((float) (count * rows / total)));
        mean.assign(newMean);
        variance.assign(m2.div((float) total));
        count = total;
    }

    @Override
    public List<Var> buffers() {
        return List.of(mean, variance);
    }

    /** @return how many rows the statistics have seen */
    public double count() {
        return count;
    }

    @Override
    public String toString() {
        return "RunningObservationNorm(" + features + ")";
    }
}
