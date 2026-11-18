package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Physics-aware training (Wright et al., "Deep physical neural networks trained with
 * backpropagation", <a href="https://arxiv.org/abs/2104.13386">arXiv:2104.13386</a>,
 * Nature 2022): a layer whose forward pass runs through a <em>physical</em> system, which is
 * a black box with no gradient (an optical bench, a mechanical plate, an analog chip, here any
 * {@code Tensor -> Tensor} function), while the backward pass runs through a differentiable
 * <em>digital twin</em> of it.
 *
 * <pre>
 * y = physical(x)                      what the loss sees
 * dL/dtheta = (d twin(x) / dtheta)^T dL/dy, evaluated at the physical x
 * </pre>
 *
 * <p>This is the straight-through estimator with a model in the backward path: the loss and
 * the error signal come from the real hardware, so the weights are tuned to the real device
 * (fabrication errors, unmodelled nonlinearities included), while only the <em>direction</em>
 * of the update comes from the imperfect twin. Training the twin alone and then deploying it
 * ("in silico") suffers the full simulation-reality gap. In a stack of such layers each input
 * already holds the physical activations of the layer before, so every twin is linearized at
 * the real operating point, as in the paper.
 *
 * <p>The twin's parameters are the parameters the physical system is configured with (its
 * {@code physical} function reads them when called). At evaluation ({@code training = false})
 * only the physical system runs.
 */
public final class PhysicsAwareTraining implements Layer {
    private final UnaryOperator<Tensor> physical;
    private final Layer twin;
    private long physicalCalls;

    private PhysicsAwareTraining(UnaryOperator<Tensor> physical, Layer twin) {
        this.physical = Objects.requireNonNull(physical, "physical");
        this.twin = Objects.requireNonNull(twin, "twin");
    }

    /**
     * Wraps a physical system and its digital twin.
     *
     * @param physical the physical forward pass; reads the twin's current parameters
     * @param twin the differentiable model of it
     * @return the layer
     */
    public static PhysicsAwareTraining of(UnaryOperator<Tensor> physical, Layer twin) {
        return new PhysicsAwareTraining(physical, twin);
    }

    /**
     * The functional form: physical values, twin gradients.
     *
     * @param input the input (physical activations of earlier layers)
     * @param physical the physical system
     * @param twin the digital twin
     * @return a value equal to {@code physical(input)} whose gradient is that of {@code twin(input)}
     */
    public static Var apply(Var input, UnaryOperator<Tensor> physical, UnaryOperator<Var> twin) {
        Var digital = twin.apply(input);
        Tensor real = physical.apply(input.value());
        if (!real.shape().equals(digital.shape())) {
            throw new IllegalArgumentException("physical output " + real.shape() + " does not match the twin's "
                    + digital.shape());
        }
        return Functional.straightThrough(Var.of(real), digital);
    }

    @Override
    public Var forward(Var input) {
        physicalCalls++;
        return apply(input, physical, twin::forward);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (training) {
            return forward(input);
        }
        physicalCalls++;
        return Var.of(physical.apply(input.value()));
    }

    /** @return how often the physical system ran */
    public long physicalCalls() {
        return physicalCalls;
    }

    /** @return the digital twin */
    public Layer twin() {
        return twin;
    }

    @Override
    public List<Var> parameters() {
        return twin.parameters();
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return twin.outputShape(inputShape);
    }
}
