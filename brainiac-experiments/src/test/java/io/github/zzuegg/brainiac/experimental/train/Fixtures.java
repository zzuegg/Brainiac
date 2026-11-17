package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/** Small seeded tasks for optimizer tests. */
final class Fixtures {

    private Fixtures() {
    }

    /** Three interleaved spirals: {x [n, 2], ids [n]}. */
    static Tensor[] spiral(int perClass, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = perClass * 3;
        float[] x = new float[n * 2];
        float[] y = new float[n];
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < perClass; i++) {
                double radius = (double) i / perClass;
                double angle = c * 2 * Math.PI / 3 + radius * 4 + r.nextGaussian() * 0.1;
                int k = c * perClass + i;
                x[2 * k] = (float) (radius * Math.sin(angle));
                x[2 * k + 1] = (float) (radius * Math.cos(angle));
                y[k] = c;
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 2), x), Tensor.of(Shape.of(n), y)};
    }

    static float loss(SimpleMlp net, Tensor[] data) {
        return Functional.crossEntropy(net.forward(Var.of(data[0])), data[1]).value().item();
    }

    /** Full-batch training; returns the final loss. */
    static float train(SimpleMlp net, Optimizer opt, Tensor[] data, int steps) {
        for (int i = 0; i < steps; i++) {
            Functional.crossEntropy(net.forward(Var.of(data[0])), data[1]).backward();
            opt.step(net.parameters());
        }
        return loss(net, data);
    }
}
