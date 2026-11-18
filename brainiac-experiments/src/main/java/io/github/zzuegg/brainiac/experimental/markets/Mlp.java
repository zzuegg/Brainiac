package io.github.zzuegg.brainiac.experimental.markets;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** A small tanh multilayer perceptron on autograd parameters, shared by the market blocks. */
final class Mlp {

    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();

    Mlp(int[] sizes, SplittableRandom rnd) {
        for (int l = 0; l + 1 < sizes.length; l++) {
            weights.add(Var.param(random(sizes[l], sizes[l + 1], rnd)));
            biases.add(Var.param(Tensor.zeros(Shape.of(sizes[l + 1]))));
        }
    }

    static Tensor random(int in, int out, SplittableRandom rnd) {
        float[] w = new float[in * out];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (ChannelCodes.gaussian(rnd) / Math.sqrt(in));
        }
        return Tensor.of(Shape.of(in, out), w);
    }

    Var forward(Var x) {
        Var h = x;
        for (int l = 0; l < weights.size(); l++) {
            h = h.matmul(weights.get(l)).add(biases.get(l));
            if (l + 1 < weights.size()) {
                h = h.tanh();
            }
        }
        return h;
    }

    List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        p.addAll(biases);
        return p;
    }

    long count() {
        long c = 0;
        for (Var v : parameters()) {
            c += v.value().size();
        }
        return c;
    }
}
