package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Plasticity;
import io.github.zzuegg.brainiac.train.Trainer;

/**
 * The growing brain: regions of neurons joined by sparse connections that are pruned,
 * regrown and extended with new neurons while the network learns.
 *
 * <ol>
 *   <li>Five interleaved spiral arms: a brain that starts with 4 neurons grows new ones
 *       whenever learning stalls, until it can separate the arms.</li>
 *   <li>Images of bars and crosses: a three-region brain (sensory, association, motor)
 *       with feedback connections, settling over 3 ticks, keeps only 10-20% of its
 *       possible connections and rewires them where the gradient says they are needed.
 *       It is compared with a dense network of the same neurons.</li>
 * </ol>
 *
 * <p>{@code ./gradlew :brainiac-examples:run --args=brain}
 */
public final class BrainExample {

    private static final int EPOCHS = 30;

    private BrainExample() {
    }

    /**
     * Runs both parts.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        growing();
        sparse();
    }

    private static Brain brainOf(Model m) {
        return (Brain) m.layers().stream().filter(Brain.class::isInstance).findFirst().orElseThrow();
    }

    private static void growing() {
        System.out.println("== Growing: five spiral arms, starting from 4 neurons ==");
        Dataset train = SpiralExample.data(200, 5, 1);
        Dataset test = SpiralExample.data(100, 5, 2);
        Model model = Model.sequential().input(2)
                .add(Brain.builder()
                        .region("hidden", 128).initialNeurons(4).activation(Activation.RELU)
                        .region("deep", 128).initialNeurons(4).activation(Activation.TANH)
                        .connect("input", "hidden", 1.0)
                        .connect("hidden", "deep", 0.5)
                        .output("deep")
                        .build())
                .dense(5, Activation.SOFTMAX)
                .seed(3)
                .build();
        Brain brain = brainOf(model);
        Plasticity plasticity = Plasticity.of(model)
                .neurogenesis(4, 0.02, 8)
                .rewire(50, 0.2).until(20_000)
                .pruneDeadNeurons(1e-4)
                .seed(1);
        Trainer trainer = Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f))
                .batchSize(32).epochs(300).metric(Metric.ACCURACY).listener(plasticity).seed(1);
        trainer.fit(train);
        System.out.println("epoch  loss    neurons  connections");
        for (Plasticity.Snapshot s : plasticity.history()) {
            if (s.epoch() % 30 == 0 || s.epoch() == 1) {
                System.out.printf("%5d  %.4f  %7d  %11d%n", s.epoch(), s.loss(), s.neurons(), s.connections());
            }
        }
        System.out.println("First neuron changes:");
        plasticity.events().stream().filter(e -> !e.kind().equals("rewire")).limit(8)
                .forEach(e -> System.out.println("  " + e));
        System.out.printf("... %d growth events and %d removals in all%n",
                plasticity.events().stream().filter(e -> e.kind().equals("grow neurons")).count(),
                plasticity.events().stream().filter(e -> e.kind().equals("remove neurons")).count());
        System.out.printf("Test accuracy %.1f%%%n%s%n", 100 * trainer.evaluate(test).get(Metric.ACCURACY),
                brain.describe());
    }

    private static void sparse() {
        System.out.println("== Sparse: bars and crosses with a three-region brain ==");
        Dataset train = MnistExample.shapes(1500, 16, 1);
        Dataset test = MnistExample.shapes(500, 16, 2);
        Model brainModel = Model.sequential().input(1, 16, 16).flatten()
                .add(Brain.builder()
                        .region("sensory", 64).activation(Activation.RELU)
                        .region("association", 32).activation(Activation.RELU)
                        .region("motor", 16)
                        .connect("input", "sensory", 0.1)
                        .connect("sensory", "association", 0.2)
                        .connect("association", "motor", 0.2)
                        .connect("association", "sensory", 0.1)      // feedback
                        .connect("motor", "association", 0.2)        // feedback
                        .output("motor")
                        .ticks(3)
                        .build())
                .dense(3, Activation.SOFTMAX)
                .seed(4)
                .build();
        Brain brain = brainOf(brainModel);
        Plasticity plasticity = Plasticity.of(brainModel).rewire(25, 0.3).until(1_000)
                .growth(Plasticity.Growth.GRADIENT).seed(2);
        Trainer t1 = Trainer.of(brainModel).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.005f))
                .batchSize(32).epochs(EPOCHS).metric(Metric.ACCURACY).listener(plasticity).seed(1);
        long start = System.nanoTime();
        t1.fit(train);
        double brainSeconds = (System.nanoTime() - start) / 1e9;
        float brainAccuracy = t1.evaluate(test).get(Metric.ACCURACY);
        long rewired = plasticity.events().stream().filter(e -> e.kind().equals("rewire")).count();

        Model dense = Model.sequential().input(1, 16, 16).flatten()
                .dense(64, Activation.RELU).dense(32, Activation.RELU).dense(16, Activation.TANH)
                .dense(3, Activation.SOFTMAX).seed(4).build();
        Trainer t2 = Trainer.of(dense).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.005f))
                .batchSize(32).epochs(EPOCHS).metric(Metric.ACCURACY).seed(1);
        start = System.nanoTime();
        t2.fit(train);
        double denseSeconds = (System.nanoTime() - start) / 1e9;
        float denseAccuracy = t2.evaluate(test).get(Metric.ACCURACY);
        long denseConnections = 256 * 64 + 64 * 32 + 32 * 16;

        Model small = Model.sequential().input(1, 16, 16).flatten()
                .dense(9, Activation.RELU).dense(16, Activation.TANH)
                .dense(3, Activation.SOFTMAX).seed(4).build();
        Trainer t3 = Trainer.of(small).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.005f))
                .batchSize(32).epochs(EPOCHS).metric(Metric.ACCURACY).seed(1);
        t3.fit(train);
        float smallAccuracy = t3.evaluate(test).get(Metric.ACCURACY);
        long smallConnections = 256 * 9 + 9 * 16;

        System.out.print(brain.describe());
        System.out.printf("rewired %d times during training%n%n", rewired);
        System.out.printf("%-26s %12s %10s %8s%n", "model", "connections", "accuracy", "train s");
        System.out.printf("%-26s %12d %9.1f%% %8.1f%n", "brain (sparse, feedback)", brain.connections(),
                100 * brainAccuracy, brainSeconds);
        System.out.printf("%-26s %12d %9.1f%% %8.1f%n", "dense, same neurons", denseConnections,
                100 * denseAccuracy, denseSeconds);
        System.out.printf("%-26s %12d %9.1f%% %8s%n", "dense, same connections", smallConnections,
                100 * smallAccuracy, "");
    }
}
