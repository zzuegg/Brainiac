package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KanLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The toy regression of the KAN paper (Liu et al., <a href="https://arxiv.org/abs/2404.19756">2404.19756</a>):
 * a [2, 5, 1] Kolmogorov-Arnold network fits {@code f(x, y) = exp(sin(pi x) + y^2)} on a
 * coarse grid, then grid extension moves it to a finer grid and training continues from
 * where it was rather than from scratch.
 */
public final class KolmogorovArnoldFit {
    private KanLayer first;
    private KanLayer second;
    private Optimizer optimizer;
    private final Tensor inputs;
    private final Tensor targets;
    private final Tensor testInputs;
    private final Tensor testTargets;
    private final float learningRate;

    private KolmogorovArnoldFit(int intervals, int hidden, float learningRate, long seed) {
        this.first = KanLayer.of(2, hidden, intervals, -1, 1, seed);
        this.second = KanLayer.of(hidden, 1, intervals, -1, 1, seed + 1);
        this.learningRate = learningRate;
        this.optimizer = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed + 2);
        Tensor[] train = data(256, r);
        Tensor[] test = data(512, r);
        this.inputs = train[0];
        this.targets = train[1];
        this.testInputs = test[0];
        this.testTargets = test[1];
    }

    /**
     * Creates the experiment.
     *
     * @param intervals the starting grid size
     * @param hidden the hidden width
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the experiment
     */
    public static KolmogorovArnoldFit of(int intervals, int hidden, float learningRate, long seed) {
        return new KolmogorovArnoldFit(intervals, hidden, learningRate, seed);
    }

    /**
     * The target function.
     *
     * @param x the first input
     * @param y the second input
     * @return {@code exp(sin(pi x) + y^2)}
     */
    public static double target(double x, double y) {
        return Math.exp(Math.sin(Math.PI * x) + y * y);
    }

    private static Tensor[] data(int n, SplittableRandom r) {
        float[] x = new float[2 * n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            x[2 * i] = (float) r.nextDouble(-1, 1);
            x[2 * i + 1] = (float) r.nextDouble(-1, 1);
            y[i] = (float) target(x[2 * i], x[2 * i + 1]);
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 2), x), Tensor.of(Shape.of(n, 1), y)};
    }

    private Var predict(Tensor x) {
        return second.forward(first.forward(Var.of(x)));
    }

    private List<Var> parameters() {
        List<Var> p = new ArrayList<>(first.parameters());
        p.addAll(second.parameters());
        return p;
    }

    /**
     * Trains full-batch.
     *
     * @param steps the optimizer steps
     * @return the last training loss
     */
    public float train(int steps) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Var loss = predict(inputs).sub(Var.of(targets)).square().mean();
            last = loss.value().item();
            loss.backward();
            optimizer.step(parameters());
        }
        return last;
    }

    /** @return the root mean squared error on held-out points */
    public float testRmse() {
        return (float) Math.sqrt(predict(testInputs).value().sub(testTargets).square().mean().item());
    }

    /**
     * Grid extension of both layers. The hidden layer's grid range follows the hidden
     * activations actually seen, with a margin.
     *
     * @param intervals the new number of intervals
     */
    public void refine(int intervals) {
        Tensor hidden = first.forward(Var.of(inputs)).value();
        float lo = hidden.min().item();
        float hi = hidden.max().item();
        float margin = 0.1f * (hi - lo);
        KanLayer f = first.refine(intervals, -1, 1, inputs);
        second = second.refine(intervals, lo - margin, hi + margin, hidden);
        first = f;
        optimizer = Optimizer.adam(learningRate);
    }

    /** @return the current grid size */
    public int intervals() {
        return first.intervals();
    }
}
