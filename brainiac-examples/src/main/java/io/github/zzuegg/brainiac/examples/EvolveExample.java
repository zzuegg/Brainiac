package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.evolve.Evolution;
import io.github.zzuegg.brainiac.evolve.Genome;
import io.github.zzuegg.brainiac.evolve.Mutations;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Task;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Neuroevolution: starts from the hand-built model of a task, with the same or fewer hidden
 * neurons, and lets a genetic algorithm rewire it, resize it and pick an activation for
 * every neuron. Each candidate is trained briefly and scored on held-out training data;
 * at the end the winner and the hand-built model are both trained with the task's own
 * settings and tested on the test data.
 *
 * <pre>
 * ./gradlew :brainiac-examples:run --args="evolve spiral"
 * ./gradlew :brainiac-examples:run --args="evolve spiral --population 20 --generations 10 --scale 0.75
 *     --mutate connections,neurons,activations,learning-rate --epochs 40 --threads 4"
 * ./gradlew :brainiac-examples:run --args="evolve shapes --device gpu"
 * </pre>
 *
 * <p>Options: {@code --population N}, {@code --generations G}, {@code --mutate a,b,c} (any of
 * {@link Mutations#names()}), {@code --epochs E} (training budget per candidate),
 * {@code --scale S} (start from this fraction of the template's neurons),
 * {@code --size-penalty P} (fitness lost per thousand parameters), {@code --threads T},
 * {@code --seed S}, {@code --device cpu|gpu|best|vulkan}, {@code --batched} or {@code --one-by-one}
 * (train each generation stacked into one model or not, see {@link Evolution#batched}; by
 * default stacked on a GPU), {@code --inherit} (children start
 * from their parents' trained weights, see {@link Evolution#inheritWeights}), {@code --search-only}
 * (skip the final comparison).
 */
public final class EvolveExample {

    private EvolveExample() {
    }

    /**
     * A design that is known to work for a task, matching its hand-built model.
     *
     * @param task the task
     * @param frontEnd fixed layers before the evolved part, or null
     * @param template the evolved part's starting design
     * @param handBuilt makes the hand-built model
     */
    record Start(Task task, Consumer<Model.Builder> frontEnd, Genome template, Supplier<Model> handBuilt) {
    }

    /** Two conv blocks, as in the shapes and MNIST CNN, kept fixed while the rest evolves. */
    static void convFrontEnd(Model.Builder b) {
        b.add(Conv.of(16, 3, 3).padding(Padding.SAME).bias(false)).batchNorm().activation(Activation.RELU)
                .maxPool(2, 2)
                .add(Conv.of(32, 3, 3).padding(Padding.SAME).bias(false)).batchNorm().activation(Activation.RELU)
                .maxPool(2, 2);
    }

    /**
     * The starting point for a task.
     *
     * @param name the task's name
     * @return the task, its template and its hand-built model
     */
    static Start start(String name) {
        Task task = Tasks.get(name);
        return switch (name) {
            case "xor" -> new Start(task, null, Genome.dense(2, Activation.TANH, 8), XorExample::model);
            case "iris" -> new Start(task, null, Genome.dense(4, Activation.RELU, 16),
                    () -> Model.sequential().input(4).dense(16, Activation.RELU).dense(3, Activation.SOFTMAX)
                            .seed(42).build());
            case "spiral" -> new Start(task, null, Genome.dense(2, Activation.RELU, 64, 64),
                    () -> SpiralExample.model(3));
            case "shapes" -> new Start(task, EvolveExample::convFrontEnd,
                    Genome.dense(32 * 4 * 4, Activation.RELU, 64), () -> MnistExample.model(16, 3));
            default -> {
                int inputs = Math.toIntExact(task.inputShape().size());
                yield new Start(task, null, Genome.dense(inputs, Activation.RELU, 64),
                        () -> BenchmarkExample.model("mlp", task));
            }
        };
    }

    /**
     * Runs a search and the comparison.
     *
     * @param args the task name, then options
     */
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: evolve <task> [--population N] [--generations G] [--mutate a,b,c]"
                    + " [--epochs E] [--scale S] [--size-penalty P] [--threads T] [--seed S] [--device cpu|gpu|vulkan]"
                    + " [--batched|--one-by-one] [--inherit] [--search-only]\n"
                    + "  tasks: xor, iris, spiral, shapes, ...   mutations: " + String.join(",", Mutations.names()));
            return;
        }
        Start start = start(args[0]);
        int population = 16;
        int generations = 8;
        String mutate = "connections,neurons,activations,learning-rate";
        int epochs = -1;
        double scale = 0.75;
        double sizePenalty = 0;
        int threads = 1;
        long seed = 1;
        Device device = null;
        Boolean batched = null;
        boolean inherit = false;
        boolean compare = true;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--population" -> population = Integer.parseInt(args[++i]);
                case "--generations" -> generations = Integer.parseInt(args[++i]);
                case "--mutate" -> mutate = args[++i];
                case "--epochs" -> epochs = Integer.parseInt(args[++i]);
                case "--scale" -> scale = Double.parseDouble(args[++i]);
                case "--size-penalty" -> sizePenalty = Double.parseDouble(args[++i]);
                case "--threads" -> threads = Integer.parseInt(args[++i]);
                case "--seed" -> seed = Long.parseLong(args[++i]);
                case "--batched" -> batched = true;
                case "--one-by-one" -> batched = false;
                case "--inherit" -> inherit = true;
                case "--search-only" -> compare = false;
                case "--device" -> device = BenchmarkExample.device(args[++i]);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        Task task = start.task();
        if (epochs < 0) {
            epochs = budget(task);
        }
        Evolution evolution = Evolution.of(task)
                .template(start.template())
                .frontEnd(start.frontEnd())
                .neuronScale(scale)
                .mutations(mutate)
                .population(population)
                .generations(generations)
                .epochs(epochs)
                .sizePenalty(sizePenalty / 1000)
                .threads(threads)
                .seed(seed)
                .inheritWeights(inherit)
                .onGeneration(g -> System.out.println(g));
        if (device != null) {
            evolution.device(device);
        }
        if (batched != null) {
            evolution.batched(batched);
        }
        System.out.printf("Evolving %s: population %d, %d generations, %d epochs per candidate, start at %.0f%%"
                        + " of the template's neurons, mutating %s%s%s%n", task.name(), population, generations, epochs,
                100 * scale, mutate, evolution.batched() ? ", whole population per model" : "",
                inherit ? ", children inherit weights" : "");
        long t0 = System.nanoTime();
        Evolution.Result result = evolution.run();
        double searchSeconds = (System.nanoTime() - t0) / 1e9;
        Genome best = result.best().genome();
        System.out.printf("%nSearch took %.1f s. Best design:%n%s", searchSeconds, best.describe());

        if (!compare) {
            return;
        }
        System.out.println("\nTrained with the task's own settings and tested on the test data:");
        Benchmark.Report hand = evolution.benchmark(start.handBuilt(), "hand-built");
        Benchmark.Report evolved = evolution.benchmark(best, "evolved");
        System.out.print(Benchmark.table(List.of(hand, evolved)));
        long handParams = hand.parameters();
        long evolvedParams = best.effectiveParameters(Math.toIntExact(task.outputShape().size()));
        if (start.frontEnd() != null) {
            Model m = evolution.build(best, 1);
            long brainAll = Genome.brainOf(m).parameters().stream().mapToLong(v -> v.value().shape().size()).sum();
            long fed = best.outputs().stream().mapToInt(o -> best.region(o).neurons().size()).sum();
            long outputs = task.outputShape().size();
            evolvedParams += m.parameterCount() - brainAll - (fed * outputs + outputs);   // the front end's
        }
        System.out.printf("Parameters in use: hand-built %,d, evolved %,d (%.0f%%)%n", handParams, evolvedParams,
                100.0 * evolvedParams / handParams);
        if (task.kind() == Benchmark.Kind.CLASSIFICATION) {
            System.out.printf("Accuracy per thousand parameters: hand-built %.3f, evolved %.3f%n",
                    1000 * hand.accuracy() / handParams, 1000 * evolved.accuracy() / evolvedParams);
        }
    }

    /** A short training budget per candidate: a fifth of the task's epochs, at least 5. */
    static int budget(Task task) {
        io.github.zzuegg.brainiac.train.Trainer probe = io.github.zzuegg.brainiac.train.Trainer
                .of(Model.sequential().input(1).dense(1).build());
        task.defaults().accept(probe);
        return Math.max(5, probe.epochs() / 5);
    }
}
