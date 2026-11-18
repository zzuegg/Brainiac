package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.StatefulOptimizer;

import java.util.List;
import java.util.Objects;

/**
 * Adaptive federated optimisation (FedOpt; Reddi et al., ICLR 2021,
 * <a href="https://arxiv.org/abs/2003.00295">arXiv:2003.00295</a>): the server treats the
 * negated average client update {@code -Δ = w - mean(w_i)} as a pseudo-gradient and hands it
 * to any core {@link Optimizer}. Plain SGD at rate 1 is exactly FedAvg; momentum gives
 * FedAvgM; Adam and {@link Yogi} give FedAdam and FedYogi, which need far less tuning of the
 * client learning rate and cope better with heterogeneous clients.
 */
public final class ServerOptimizer {

    private final Optimizer optimizer;
    private final String name;

    private ServerOptimizer(Optimizer optimizer, String name) {
        this.optimizer = Objects.requireNonNull(optimizer);
        this.name = name;
    }

    /**
     * Any core optimizer as the server step.
     *
     * @param optimizer the optimizer, stepping on the pseudo-gradient
     * @return the server optimizer
     */
    public static ServerOptimizer of(Optimizer optimizer) {
        return new ServerOptimizer(optimizer, optimizer.toString());
    }

    /** @return FedAvg: replace the global model by the clients' average */
    public static ServerOptimizer fedAvg() {
        return new ServerOptimizer(Optimizer.sgd(1), "FedAvg");
    }

    /**
     * @param learningRate the server rate
     * @param momentum the server momentum, such as 0.9
     * @return FedAvgM
     */
    public static ServerOptimizer fedAvgM(float learningRate, float momentum) {
        return new ServerOptimizer(Optimizer.momentum(learningRate, momentum), "FedAvgM");
    }

    /**
     * @param learningRate the server rate
     * @return FedAdam with {@code β = (0.9, 0.99)} and adaptivity {@code τ = 1e-3}
     */
    public static ServerOptimizer fedAdam(float learningRate) {
        return new ServerOptimizer(Optimizer.adam(learningRate, 0.9f, 0.99f, 1e-3f), "FedAdam");
    }

    /**
     * @param learningRate the server rate
     * @return FedYogi with {@code β = (0.9, 0.99)} and {@code τ = 1e-3}
     */
    public static ServerOptimizer fedYogi(float learningRate) {
        return new ServerOptimizer(new Yogi(learningRate, 0.9f, 0.99f, 1e-3f), "FedYogi");
    }

    /**
     * Applies the averaged client update.
     *
     * @param global the global parameters, updated in place
     * @param delta the aggregated update {@code mean(w_i) - w}, one per parameter
     */
    public void apply(List<Var> global, Tensor[] delta) {
        for (int i = 0; i < delta.length; i++) {
            global.get(i).setGrad(delta[i].neg());
        }
        optimizer.step(global);
    }

    @Override
    public String toString() {
        return name;
    }

    /**
     * The Yogi optimizer (Zaheer et al., NeurIPS 2018): Adam whose second moment moves
     * additively, {@code v -= (1 - β2) g² sign(v - g²)}, so it cannot grow too fast after a
     * burst of large gradients. Drop-in for any core {@link Optimizer}.
     */
    public static final class Yogi extends StatefulOptimizer {
        private final float beta1;
        private final float beta2;
        private final float epsilon;

        /**
         * Creates the optimizer.
         *
         * @param learningRate the step size
         * @param beta1 the first-moment decay
         * @param beta2 the second-moment rate
         * @param epsilon the adaptivity floor
         */
        public Yogi(float learningRate, float beta1, float beta2, float epsilon) {
            super(learningRate, "m", "v");
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
        }

        @Override
        protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
            Tensor g2 = g.square();
            Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
            Tensor v = s[1] == null ? g2 : s[1].sub(g2.mul(s[1].sub(g2).sign()).mul(1 - beta2));
            p.assign(p.value().sub(m.div(v.sqrt().add(epsilon)).mul(learningRate())));
            return new Tensor[] {m, v};
        }

        @Override
        public String toString() {
            return "Yogi(lr=" + learningRate() + ")";
        }
    }
}
