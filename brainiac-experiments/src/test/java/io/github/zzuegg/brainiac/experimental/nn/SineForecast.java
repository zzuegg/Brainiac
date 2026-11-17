package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** Next-value forecasting of sines with random frequency and phase, for sequence layers. */
final class SineForecast {

    private SineForecast() {
    }

    /** {input [b, t, 1], target [b, t, 1]} with target[t] = input[t + 1]. */
    static Tensor[] batch(SplittableRandom r, int size, int steps) {
        float[] x = new float[size * steps];
        float[] y = new float[size * steps];
        for (int i = 0; i < size; i++) {
            double freq = 0.3 + r.nextDouble() * 0.5;
            double phase = r.nextDouble() * 2 * Math.PI;
            for (int t = 0; t < steps; t++) {
                x[i * steps + t] = (float) Math.sin(freq * t + phase);
                y[i * steps + t] = (float) Math.sin(freq * (t + 1) + phase);
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(size, steps, 1), x), Tensor.of(Shape.of(size, steps, 1), y)};
    }

    /**
     * Trains {@code in (1 -> width) -> layer -> out (outWidth -> 1)} and returns the MSE over the
     * second half of fresh sequences (where enough history exists to infer the frequency).
     */
    static double train(Layer layer, int width, int outWidth, int steps, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var in = MixerOps.weight(r, 1, width);
        Var out = MixerOps.weight(r, outWidth, 1);
        List<Var> params = new ArrayList<>(List.of(in, out));
        params.addAll(layer.parameters());
        Optimizer opt = Optimizer.adam(lr);
        int length = 16;
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(r, 16, length);
            Var pred = layer.forward(Var.of(b[0]).matmul(in), true).matmul(out);
            Var loss = pred.slice(1, 3, length).sub(Var.of(b[1]).slice(1, 3, length)).square().mean();
            loss.backward();
            opt.step(params);
        }
        Tensor[] test = batch(new SplittableRandom(seed + 7), 64, length);
        Var pred = layer.forward(Var.of(test[0]).matmul(in)).matmul(out);
        return pred.slice(1, length / 2, length).sub(Var.of(test[1]).slice(1, length / 2, length)).square().mean()
                .value().item();
    }
}
