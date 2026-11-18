package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** A small ReLU perceptron over raw parameters, for the mixers' hypernetworks. */
final class MixerNet {

    private final List<Var> params = new ArrayList<>();

    MixerNet(SplittableRandom random, int... sizes) {
        for (int i = 0; i + 1 < sizes.length; i++) {
            float std = (float) Math.sqrt(2.0 / sizes[i]);
            params.add(Var.param(Tensor.normal(Shape.of(sizes[i], sizes[i + 1]), 0, std, random)));
            params.add(Var.param(Tensor.full(Shape.of(1, sizes[i + 1]), 0f)));
        }
    }

    List<Var> parameters() {
        return params;
    }

    Var apply(Var x) {
        Var h = x;
        for (int i = 0; i < params.size(); i += 2) {
            h = h.matmul(params.get(i)).add(params.get(i + 1));
            if (i + 2 < params.size()) {
                h = h.relu();
            }
        }
        return h;
    }
}
