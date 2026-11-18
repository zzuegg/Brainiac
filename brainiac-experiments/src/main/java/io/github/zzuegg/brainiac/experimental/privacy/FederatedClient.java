package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.LocalWorker;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * One simulated federated client: a replica of the model's parameters, the size of its local
 * data set, and a closure that computes the loss of a fresh minibatch of its private data with
 * the replica's current parameters. The data never leaves the closure; algorithms only read
 * the parameters and gradients. It is a {@link LocalWorker}, so core-style
 * {@code FederatedAveraging} runs it unchanged (its {@link #train} is plain local SGD), while
 * {@link FederatedAlgorithms} and {@link PersonalizedFederated} drive {@link #gradients()}
 * themselves to add proximal terms, control variates or personal heads.
 */
public final class FederatedClient implements LocalWorker {

    private final List<Var> parameters;
    private final int samples;
    private final Supplier<Var> batchLoss;
    private final float learningRate;

    /**
     * Creates a client.
     *
     * @param parameters the replica's parameters, in the global model's order
     * @param samples the number of local examples (the averaging weight)
     * @param batchLoss computes a minibatch loss with the replica's parameters
     * @param learningRate the local SGD step size
     */
    public FederatedClient(List<Var> parameters, int samples, Supplier<Var> batchLoss, float learningRate) {
        this.parameters = List.copyOf(parameters);
        this.samples = samples;
        this.batchLoss = Objects.requireNonNull(batchLoss);
        this.learningRate = learningRate;
    }

    @Override
    public List<Var> parameters() {
        return parameters;
    }

    @Override
    public int samples() {
        return samples;
    }

    /** @return the local SGD step size */
    public float learningRate() {
        return learningRate;
    }

    /** @return the closure computing a minibatch loss */
    public Supplier<Var> batchLoss() {
        return batchLoss;
    }

    /**
     * The gradient of one fresh minibatch at the current parameters.
     *
     * @param loss receives the minibatch loss in element 0, may be {@code null}
     * @return one gradient per parameter
     */
    public Tensor[] gradients(float[] loss) {
        Var l = batchLoss.get();
        if (loss != null) {
            loss[0] = l.value().item();
        }
        List<Var> g = Var.gradients(l, parameters, false);
        Tensor[] out = new Tensor[g.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = g.get(i).value();
        }
        return out;
    }

    @Override
    public float train(int steps) {
        float[] l = new float[1];
        double sum = 0;
        for (int s = 0; s < steps; s++) {
            Tensor[] g = gradients(l);
            sum += l[0];
            for (int i = 0; i < g.length; i++) {
                Var p = parameters.get(i);
                p.assign(p.value().sub(g[i].mul(learningRate)));
            }
        }
        return (float) (sum / Math.max(1, steps));
    }

    /** @return a copy of the current parameter values */
    public List<Tensor> snapshot() {
        List<Tensor> out = new ArrayList<>(parameters.size());
        for (Var p : parameters) {
            out.add(p.value());
        }
        return out;
    }

    /**
     * Loads parameter values.
     *
     * @param values one value per parameter
     */
    public void load(List<Tensor> values) {
        for (int i = 0; i < parameters.size(); i++) {
            parameters.get(i).assign(values.get(i));
        }
    }
}
