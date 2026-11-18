package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * FedDF, ensemble distillation for model fusion (Lin, Kong, Stich and Jaggi, NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2006.07242">arXiv:2006.07242</a>), as a
 * {@link FederatedAlgorithms.Aggregator}: the server starts from the FedAvg average of the
 * client models and then distils the ensemble of client models into it on unlabelled server
 * data, minimising the KL divergence between the softened mean of the clients' logits and the
 * server model's. Averaging weights is a poor fusion when clients drift apart; distillation fuses
 * their functions instead.
 */
public final class EnsembleDistillationAggregator implements FederatedAlgorithms.Aggregator {

    private final List<Var> model;
    private final UnaryOperator<Var> forward;
    private final Tensor unlabelled;
    private final int steps;
    private final int batch;
    private final float learningRate;
    private final float temperature;
    private final SplittableRandom random;

    /**
     * Creates the aggregator.
     *
     * @param model the global model's parameters (the same {@code Var}s the algorithm updates)
     * @param forward the global model's logits as a function of an input batch
     * @param unlabelled the server's unlabelled inputs {@code [n, ...]}
     * @param steps distillation steps per round
     * @param batch the distillation minibatch
     * @param learningRate the Adam step size
     * @param temperature the softening temperature
     * @param seed the minibatch seed
     */
    public EnsembleDistillationAggregator(List<Var> model, UnaryOperator<Var> forward, Tensor unlabelled, int steps,
            int batch, float learningRate, float temperature, long seed) {
        this.model = List.copyOf(model);
        this.forward = forward;
        this.unlabelled = unlabelled;
        this.steps = steps;
        this.batch = batch;
        this.learningRate = learningRate;
        this.temperature = temperature;
        this.random = new SplittableRandom(seed);
    }

    @Override
    public Tensor[] aggregate(Tensor[] start, Tensor[][] updates, double[] weights, int[] localSteps) {
        int n = unlabelled.shape().dim(0);
        // the ensemble's mean logits on the server data
        Tensor teacher = null;
        for (Tensor[] u : updates) {
            for (int p = 0; p < start.length; p++) {
                model.get(p).assign(start[p].add(u[p]));
            }
            Tensor logits = Var.noGrad(() -> forward.apply(Var.of(unlabelled))).value();
            teacher = teacher == null ? logits : teacher.add(logits);
        }
        teacher = teacher.mul(1f / updates.length);
        Tensor[] avg = FedNova.average(updates, weights);
        for (int p = 0; p < start.length; p++) {
            model.get(p).assign(start[p].add(avg[p]));
        }
        float[] t = teacher.toArray();
        int classes = (int) (teacher.size() / n);
        int row = (int) (unlabelled.size() / n);
        float[] xs = unlabelled.toArray();
        int[] dims = unlabelled.shape().dims();
        Optimizer opt = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            int b = Math.min(batch, n);
            float[] bx = new float[b * row];
            float[] bt = new float[b * classes];
            for (int i = 0; i < b; i++) {
                int j = random.nextInt(n);
                System.arraycopy(xs, j * row, bx, i * row, row);
                System.arraycopy(t, j * classes, bt, i * classes, classes);
            }
            int[] bd = dims.clone();
            bd[0] = b;
            Tensor target = Var.of(Tensor.of(Shape.of(b, classes), bt)).div(temperature).softmax(1).value();
            Var student = forward.apply(Var.of(Tensor.of(Shape.of(bd), bx))).div(temperature).logSoftmax(1);
            Var loss = student.mul(target).sum().neg().div(b);
            loss.backward();
            opt.step(model);
        }
        Tensor[] delta = new Tensor[start.length];
        for (int p = 0; p < start.length; p++) {
            delta[p] = model.get(p).value().sub(start[p]);
            model.get(p).assign(start[p]);
        }
        return delta;
    }
}
