package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * A forward hook: wraps a layer and passes its output through a function, which may read
 * it (record activations) or replace it (add a steering vector, patch in activations from
 * another run, ablate units). The wrapped layer's parameters and buffers are shared, so the
 * hooked layer can stand in for it in a model with no copying.
 *
 * <pre>{@code
 * List<Tensor> seen = new ArrayList<>();
 * Layer probe = HookedLayer.of(layer, out -> { seen.add(out.value()); return out; });
 * Layer steered = HookedLayer.of(layer, out -> out.add(direction.mul(4f)));
 * }</pre>
 */
public final class HookedLayer implements Layer {

    private final Layer inner;
    private final UnaryOperator<Var> hook;

    private HookedLayer(Layer inner, UnaryOperator<Var> hook) {
        this.inner = Objects.requireNonNull(inner, "inner");
        this.hook = Objects.requireNonNull(hook, "hook");
    }

    /**
     * Wraps a layer.
     *
     * @param layer the layer
     * @param hook applied to its output
     * @return the hooked layer
     */
    public static HookedLayer of(Layer layer, UnaryOperator<Var> hook) {
        return new HookedLayer(layer, hook);
    }

    /** @return the wrapped layer */
    public Layer inner() {
        return inner;
    }

    @Override
    public Var forward(Var input) {
        return hook.apply(inner.forward(input));
    }

    @Override
    public Var forward(Var input, boolean training) {
        return hook.apply(inner.forward(input, training));
    }

    @Override
    public List<Var> parameters() {
        return inner.parameters();
    }

    @Override
    public List<Var> buffers() {
        return inner.buffers();
    }

    @Override
    public int outputSize(int inputSize) {
        return inner.outputSize(inputSize);
    }

    @Override
    public Shape outputShape(Shape input) {
        return inner.outputShape(input);
    }

    @Override
    public String toString() {
        return "Hooked(" + inner + ")";
    }
}
