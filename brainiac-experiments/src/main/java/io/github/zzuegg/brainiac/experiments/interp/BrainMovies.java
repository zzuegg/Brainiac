package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.BrainMovie;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Plasticity;
import io.github.zzuegg.brainiac.train.Trainer;
import io.github.zzuegg.brainiac.train.TrainingResult;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * X31, brain activity movies. A small growing {@link Brain}, with sensory and association
 * regions and feedback between them, learns three interleaved spiral arms while
 * {@code Plasticity} rewires it and adds neurons. A {@code BrainMovie} films every epoch,
 * showing neurons light up and get recruited, connections being pruned (red) and regrown
 * (green), and the loss falling. The outcome carries the GIF's bytes, ready to save.
 *
 * <p>The rewiring is that of Evci et al. 2020, "Rigging the Lottery: Making All Tickets
 * Winners", <a href="https://arxiv.org/abs/1911.11134">arXiv:1911.11134</a>.
 */
public final class BrainMovies {

    /**
     * The outcome.
     *
     * @param frames frames filmed
     * @param width frame width in pixels
     * @param height frame height in pixels
     * @param neuronsBefore active neurons before training
     * @param neuronsAfter active neurons after
     * @param finalLoss the last epoch's training loss
     * @param gif the animated GIF
     */
    public record Outcome(int frames, int width, int height, int neuronsBefore, int neuronsAfter, double finalLoss,
            byte[] gif) {

        @Override
        public boolean equals(Object o) {
            return o instanceof Outcome x && frames == x.frames && width == x.width && height == x.height
                    && neuronsBefore == x.neuronsBefore && neuronsAfter == x.neuronsAfter
                    && Double.compare(finalLoss, x.finalLoss) == 0 && Arrays.equals(gif, x.gif);
        }

        @Override
        public int hashCode() {
            return 31 * Arrays.hashCode(gif) + frames;
        }

        @Override
        public String toString() {
            return "Outcome[frames=" + frames + ", " + width + "x" + height + ", neurons " + neuronsBefore + " -> "
                    + neuronsAfter + ", loss " + finalLoss + ", " + gif.length + " bytes]";
        }
    }

    private BrainMovies() {
    }

    private static Dataset spirals(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][];
        float[][] y = new float[n][3];
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            double t = r.nextDouble();
            double a = t * 3.5 + c * 2 * Math.PI / 3;
            x[i] = new float[] {(float) (t * Math.cos(a) + r.nextGaussian() * 0.03),
                (float) (t * Math.sin(a) + r.nextGaussian() * 0.03)};
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    /**
     * Runs the experiment.
     *
     * @param epochs training epochs (one frame each, plus the starting frame)
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int epochs, long seed) {
        Dataset data = spirals(450, seed);
        Model model = Model.sequential().input(2)
                .add(Brain.builder()
                        .region("sensory", 36).initialNeurons(4).activation(Activation.RELU)
                        .region("association", 25).initialNeurons(3).activation(Activation.TANH)
                        .connect("input", "sensory", 1.0)
                        .connect("sensory", "association", 0.5)
                        .connect("association", "sensory", 0.1)
                        .output("association")
                        .ticks(2)
                        .build())
                .dense(3, Activation.SOFTMAX).seed(seed).build();
        Brain brain = (Brain) model.layers().getFirst();
        float[][] probe = new float[60][];
        for (int i = 0; i < probe.length; i++) {
            probe[i] = data.featuresOf(i);
        }
        BrainMovie movie = BrainMovie.of(model, probe);
        movie.capture();
        int before = brain.activeNeurons();
        Plasticity plasticity = Plasticity.of(model).neurogenesis(2, 0.05, 4).rewire(20, 0.2)
                .pruneDeadNeurons(1e-4).seed(seed);
        TrainingResult result = Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f))
                .batchSize(32).epochs(epochs).listener(plasticity).listener(movie).seed(seed).fit(data);
        double loss = result.history().getLast().loss();
        return new Outcome(movie.frames(), movie.width(), movie.height(), before, brain.activeNeurons(), loss,
                movie.toGif(15));
    }
}
