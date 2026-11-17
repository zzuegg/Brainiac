package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * Activation checkpointing: runs a block without keeping the values computed inside it, and
 * runs it again during the backward pass to get its gradients. Training then keeps one
 * value per block instead of every value inside it, which lets deep models train with much
 * less memory for about a third more compute.
 *
 * <pre>{@code
 * for (Block b : blocks) {
 *     x = Recompute.apply(b::forward, x, b.parameters());
 * }
 * }</pre>
 *
 * <p>The block must compute the same thing both times: no randomness (such as dropout)
 * unless it is seeded the same way, and no state that changes.
 */
public final class Recompute {

    private Recompute() {
    }

    /**
     * Runs a block with recomputation.
     *
     * @param block the block: a function of its input that may use {@code parameters}
     * @param x the block's input
     * @param parameters every parameter the block uses, so their gradients are computed
     * @return the block's output, as if {@code block.apply(x)}
     */
    public static Var apply(Function<Var, Var> block, Var x, List<Var> parameters) {
        Var[] inputs = new Var[1 + parameters.size()];
        inputs[0] = x;
        for (int i = 0; i < parameters.size(); i++) {
            inputs[1 + i] = parameters.get(i);
        }
        Op op = new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                Var out = block.apply(Var.of(in[0]));
                Tensor value = out.value();
                Tensor.evaluate(List.of(value));
                out.release();   // what the block computed on the way: not kept
                return value;
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor output, Tensor... in) {
                Tensor[] before = new Tensor[parameters.size()];
                for (int i = 0; i < before.length; i++) {
                    before[i] = parameters.get(i).grad();
                    parameters.get(i).zeroGrad();
                }
                Var input = Var.param(in[0]);
                Var out = block.apply(input);
                out.backward(grad);
                Tensor[] grads = new Tensor[in.length];
                grads[0] = input.grad();
                for (int i = 0; i < before.length; i++) {
                    Var p = parameters.get(i);
                    grads[1 + i] = p.grad();   // only what this block added; the caller adds it
                    p.setGrad(before[i]);
                }
                Tensor[] keep = java.util.Arrays.stream(grads).filter(java.util.Objects::nonNull)
                        .toArray(Tensor[]::new);
                Tensor.evaluate(List.of(keep));
                out.release(keep);
                return grads;
            }

            @Override
            public String name() {
                return "Recompute";
            }
        };
        return Var.apply(op, inputs);
    }
}
