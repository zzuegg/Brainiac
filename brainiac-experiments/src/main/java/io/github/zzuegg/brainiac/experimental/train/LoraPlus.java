package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Collection;
import java.util.function.Function;

/**
 * LoRA+ (Hayou et al. 2024, <a href="https://arxiv.org/abs/2402.12354">arXiv:2402.12354</a>):
 * train the {@code B} matrices of low-rank adapters with a learning rate {@code ratio}
 * times that of the {@code A} matrices (about 16). With equal rates, feature learning in
 * wide models is inefficient because {@code B} starts at zero; the ratio alone speeds up
 * fine-tuning by up to 2x at no cost. Built from the core's parameter groups.
 */
public final class LoraPlus {

    private LoraPlus() {
    }

    /**
     * @param bMatrices the adapters' {@code B} matrices
     * @param learningRate the rate of everything else (the {@code A} matrices)
     * @param ratio the factor for {@code B}, for example 16
     * @param optimizer makes an optimizer for a learning rate (for example {@code Optimizer::adam})
     * @return the grouped optimizer
     */
    public static Optimizer optimizer(Collection<Var> bMatrices, float learningRate, float ratio,
            Function<Float, Optimizer> optimizer) {
        return Optimizer.groups(optimizer.apply(learningRate),
                Optimizer.Group.of(bMatrices, optimizer.apply(learningRate * ratio)));
    }
}
