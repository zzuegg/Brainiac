package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Approximate posteriors over network weights from the SGD trajectory, for uncertainty
 * estimates from one training run. Each kind yields weight <em>samples</em>; predictions
 * average the model's output over them ({@link #average}).
 *
 * <ul>
 *   <li>{@link Swag} (Maddox et al. 2019, <a href="https://arxiv.org/abs/1902.02476">arXiv:1902.02476</a>):
 *       collects iterates of SGD at a constant, fairly high learning rate (like
 *       {@link StochasticWeightAveraging}, whose average is SWAG's mean) and fits a Gaussian
 *       to them: a diagonal variance from the first two moments plus a low-rank covariance
 *       from the last {@code rank} deviations,
 *       {@code w = mean + sqrt(1/2) diag(sigma) z1 + sqrt(1/(2(K-1))) D z2}.</li>
 *   <li>{@link Sgld} (Welling and Teh 2011): stochastic-gradient Langevin dynamics, an
 *       {@link Optimizer} that adds Gaussian noise of variance {@code 2 lr T / N} to each
 *       gradient step on the mean loss (with a Gaussian prior), so the iterates sample the
 *       posterior (tempered by {@code T}); snapshots of them are the samples.</li>
 * </ul>
 *
 * <p>The mean-field variational alternative, Bayes by backprop, is a layer:
 * {@code experimental.nn.BayesianDense}.
 */
public final class WeightPosterior {

    private WeightPosterior() {
    }

    /**
     * Averages a prediction over posterior samples: assigns each sample, predicts, and
     * restores the original weights.
     *
     * @param parameters the model's parameters, in the order the samples use
     * @param samples weight samples
     * @param predict computes the prediction (for example probabilities) with the current weights
     * @return the mean prediction and each sample's prediction: {@code {mean, sample 1, ...}}
     */
    public static List<Tensor> average(List<Var> parameters, List<List<Tensor>> samples, Supplier<Tensor> predict) {
        List<Tensor> saved = new ArrayList<>();
        for (Var p : parameters) {
            saved.add(p.value());
        }
        List<Tensor> out = new ArrayList<>();
        out.add(null);
        Tensor sum = null;
        try {
            for (List<Tensor> s : samples) {
                for (int i = 0; i < parameters.size(); i++) {
                    parameters.get(i).assign(s.get(i));
                }
                Tensor y = predict.get();
                out.add(y);
                sum = sum == null ? y : sum.add(y);
            }
        } finally {
            for (int i = 0; i < parameters.size(); i++) {
                parameters.get(i).assign(saved.get(i));
            }
        }
        out.set(0, sum.div(samples.size()));
        return out;
    }

    /**
     * SWAG: a Gaussian fitted to SGD iterates. Usable as a {@link TrainingListener} that
     * collects every {@code every} steps, or by calling {@link #collect()} directly.
     */
    public static final class Swag implements TrainingListener {

        private final List<Var> parameters;
        private final int rank;
        private final int every;
        private List<Tensor> mean;
        private List<Tensor> square;
        private final List<List<Tensor>> deviations = new ArrayList<>();
        private int count;

        /**
         * Creates the collector.
         *
         * @param parameters the parameters
         * @param rank how many recent deviations form the low-rank part
         * @param every collect every this many steps when used as a listener
         */
        public Swag(List<Var> parameters, int rank, int every) {
            this.parameters = List.copyOf(parameters);
            this.rank = rank;
            this.every = every;
        }

        /** Adds the current weights to the moments. */
        public void collect() {
            if (mean == null) {
                mean = new ArrayList<>();
                square = new ArrayList<>();
                for (Var p : parameters) {
                    mean.add(p.value());
                    square.add(p.value().square());
                }
            } else {
                for (int i = 0; i < parameters.size(); i++) {
                    Tensor w = parameters.get(i).value();
                    mean.set(i, mean.get(i).add(w.sub(mean.get(i)).div(count + 1)));
                    square.set(i, square.get(i).add(w.square().sub(square.get(i)).div(count + 1)));
                }
            }
            count++;
            List<Tensor> dev = new ArrayList<>();
            for (int i = 0; i < parameters.size(); i++) {
                dev.add(parameters.get(i).value().sub(mean.get(i)));
            }
            deviations.add(dev);
            if (deviations.size() > rank) {
                deviations.removeFirst();
            }
        }

        /** @return how many iterates were collected */
        public int collected() {
            return count;
        }

        /** @return the SWA mean of the iterates */
        public List<Tensor> mean() {
            return List.copyOf(mean);
        }

        /**
         * Draws weight samples from the fitted Gaussian.
         *
         * @param n how many
         * @param random the source
         * @return the samples, each one tensor per parameter
         */
        public List<List<Tensor>> sample(int n, SplittableRandom random) {
            if (count < 2) {
                throw new IllegalStateException("collect at least two iterates first");
            }
            int k = deviations.size();
            List<List<Tensor>> out = new ArrayList<>();
            for (int s = 0; s < n; s++) {
                double[] z2 = new double[k];
                for (int j = 0; j < k; j++) {
                    z2[j] = random.nextGaussian();
                }
                List<Tensor> w = new ArrayList<>();
                for (int i = 0; i < parameters.size(); i++) {
                    Tensor m = mean.get(i);
                    Tensor sigma = square.get(i).sub(m.square()).relu().sqrt();
                    Tensor x = m.add(sigma.mul(Tensor.normal(m.shape(), 0f, 1f, random)).mul((float) Math.sqrt(0.5)));
                    for (int j = 0; j < k; j++) {
                        x = x.add(deviations.get(j).get(i).mul((float) (z2[j] / Math.sqrt(2.0 * Math.max(1, k - 1)))));
                    }
                    w.add(x);
                }
                out.add(w);
            }
            return out;
        }

        @Override
        public void onBatch(int epoch, long step, float loss) {
            if (step % every == 0) {
                collect();
            }
        }

        @Override
        public void onEpoch(EpochResult result) {
        }
    }

    /**
     * Stochastic-gradient Langevin dynamics, as an optimizer over the mean training loss.
     */
    public static final class Sgld implements Optimizer {

        private float learningRate;
        private final int datasetSize;
        private final float priorPrecision;
        private final float temperature;
        private final SplittableRandom random;
        private final List<List<Tensor>> samples = new ArrayList<>();

        /**
         * Creates the sampler.
         *
         * @param learningRate the step size on the mean loss
         * @param datasetSize the training-set size {@code N}
         * @param priorPrecision the Gaussian prior's precision (weight decay {@code lambda / N})
         * @param temperature 1 for the true posterior, below 1 for a cold one
         * @param seed seeds the noise
         */
        public Sgld(float learningRate, int datasetSize, float priorPrecision, float temperature, long seed) {
            this.learningRate = learningRate;
            this.datasetSize = datasetSize;
            this.priorPrecision = priorPrecision;
            this.temperature = temperature;
            this.random = new SplittableRandom(seed);
        }

        @Override
        public void step(List<Var> parameters) {
            float noise = (float) Math.sqrt(2.0 * learningRate * temperature / datasetSize);
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                Tensor w = p.value();
                Tensor drift = g.add(w.mul(priorPrecision / datasetSize)).mul(learningRate);
                p.assign(w.sub(drift).add(Tensor.normal(w.shape(), 0f, noise, random)));
                p.zeroGrad();
            }
        }

        /**
         * Keeps the current weights as a posterior sample.
         *
         * @param parameters the parameters
         */
        public void snapshot(List<Var> parameters) {
            List<Tensor> s = new ArrayList<>();
            for (Var p : parameters) {
                s.add(p.value());
            }
            samples.add(s);
        }

        /** @return the samples kept by {@link #snapshot} */
        public List<List<Tensor>> samples() {
            return List.copyOf(samples);
        }

        @Override
        public float learningRate() {
            return learningRate;
        }

        @Override
        public void learningRate(float learningRate) {
            this.learningRate = learningRate;
        }

        @Override
        public String toString() {
            return "SGLD(lr=" + learningRate + ", T=" + temperature + ")";
        }
    }
}
