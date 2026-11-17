package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Wraps a layer and remembers the inputs it last saw, flattened to {@code [rows, features]}:
 * the calibration activations that one-shot compression (Wanda, SparseGPT, GPTQ,
 * activation-aware SVD) needs, collected by running a few batches through the wrapped
 * model. It changes nothing else.
 */
public final class InputRecorder implements Layer {

    private final Layer base;
    private Tensor last;

    /**
     * @param base the layer to watch
     */
    public InputRecorder(Layer base) {
        this.base = base;
    }

    /** @return the wrapped layer */
    public Layer base() {
        return base;
    }

    /** @return the last input {@code [rows, features]}, or null before any forward pass */
    public Tensor inputs() {
        return last;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        int features = s.dim(-1);
        last = input.value().reshape((int) (s.size() / features), features);
        return base.forward(input, training);
    }

    @Override
    public List<Var> parameters() {
        return base.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return base.outputSize(inputSize);
    }
}
