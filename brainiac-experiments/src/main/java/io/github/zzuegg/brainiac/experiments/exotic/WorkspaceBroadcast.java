package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GlobalWorkspaceLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The broadcast problem for a shared global workspace (Goyal et al.,
 * <a href="https://arxiv.org/abs/2103.01197">2103.01197</a>): several specialists each hold
 * a random vector, one of them is flagged as important, and every specialist must output the
 * flagged one's vector. With a workspace that only one specialist may write per pass, the
 * network has to learn to let the flagged specialist win the competition and then broadcast
 * its content to all.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class WorkspaceBroadcast {
    /** Number of specialists. */
    public static final int SPECIALISTS = 6;
    /** Width of each specialist's content. */
    public static final int CONTENT = 4;
    private final Dense encoder;
    private final GlobalWorkspaceLayer workspace;
    private final Dense decoder;
    private final Optimizer optimizer;
    private final SplittableRandom random;

    private WorkspaceBroadcast(int width, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.encoder = new Dense(CONTENT + 1, width, Activation.IDENTITY, Init.XAVIER, r);
        this.workspace = GlobalWorkspaceLayer.of(width, 1, 1, seed + 1);
        this.decoder = new Dense(width, CONTENT, Activation.IDENTITY, Init.XAVIER, r);
        this.optimizer = Optimizer.adam(learningRate);
        this.random = r;
    }

    /**
     * Creates the experiment with a one-slot, one-writer workspace.
     *
     * @param width the specialist width
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the experiment
     */
    public static WorkspaceBroadcast of(int width, float learningRate, long seed) {
        return new WorkspaceBroadcast(width, learningRate, seed);
    }

    /** A batch: inputs {@code [b, n, content + 1]}, targets {@code [b, n, content]}, flagged indices. */
    private record Batch(Tensor inputs, Tensor targets, int[] flagged) {
    }

    private static Batch batch(int size, SplittableRandom r) {
        int w = CONTENT + 1;
        float[] in = new float[size * SPECIALISTS * w];
        float[] out = new float[size * SPECIALISTS * CONTENT];
        int[] flagged = new int[size];
        for (int b = 0; b < size; b++) {
            flagged[b] = r.nextInt(SPECIALISTS);
            float[] chosen = new float[CONTENT];
            for (int s = 0; s < SPECIALISTS; s++) {
                for (int c = 0; c < CONTENT; c++) {
                    float v = (float) r.nextGaussian();
                    in[(b * SPECIALISTS + s) * w + c] = v;
                    if (s == flagged[b]) {
                        chosen[c] = v;
                    }
                }
                in[(b * SPECIALISTS + s) * w + CONTENT] = s == flagged[b] ? 1 : 0;
            }
            for (int s = 0; s < SPECIALISTS; s++) {
                System.arraycopy(chosen, 0, out, (b * SPECIALISTS + s) * CONTENT, CONTENT);
            }
        }
        return new Batch(Tensor.of(Shape.of(size, SPECIALISTS, w), in),
                Tensor.of(Shape.of(size, SPECIALISTS, CONTENT), out), flagged);
    }

    private Var predict(Tensor inputs) {
        return decoder.forward(workspace.forward(encoder.forward(Var.of(inputs))));
    }

    /**
     * Trains.
     *
     * @param steps optimizer steps on batches of 32
     * @return the last loss
     */
    public float train(int steps) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Batch b = batch(32, random);
            Var loss = predict(b.inputs()).sub(Var.of(b.targets())).square().mean();
            last = loss.value().item();
            loss.backward();
            List<Var> params = new ArrayList<>(encoder.parameters());
            params.addAll(workspace.parameters());
            params.addAll(decoder.parameters());
            optimizer.step(params);
        }
        return last;
    }

    /**
     * Evaluates on fresh problems.
     *
     * @param n the number of problems
     * @param seed the seed
     * @return {@code {mean squared error, share of problems where the flagged specialist wrote}}
     */
    public double[] evaluate(int n, long seed) {
        Batch b = batch(n, new SplittableRandom(seed));
        float mse = predict(b.inputs()).value().sub(b.targets()).square().mean().item();
        boolean[][] writers = workspace.lastWriters();
        int hit = 0;
        for (int i = 0; i < n; i++) {
            if (writers[i][b.flagged()[i]]) {
                hit++;
            }
        }
        return new double[] {mse, (double) hit / n};
    }
}
