package io.github.zzuegg.brainiac.evolve;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiConsumer;

/**
 * One kind of change the genetic algorithm may make to a child genome, such as rewiring
 * connections or changing activations. Each mutation decides on its own how likely and
 * how large a change is; {@link Evolution} applies every enabled mutation to every child.
 *
 * <p>Built-in mutations are in {@link Mutations}. Write your own for any other gene:
 *
 * <pre>{@code
 * Mutation moreTicks = Mutation.named("deeper-thought", (g, random) -> {
 *     if (random.nextDouble() < 0.1) g.ticks++;
 * });
 * }</pre>
 */
public interface Mutation {

    /**
     * The name used to switch the mutation on, for example on the command line.
     *
     * @return the name
     */
    String name();

    /**
     * Changes a child genome in place.
     *
     * @param genome the child
     * @param random the only source of randomness to use, so runs are reproducible
     */
    void apply(Genome.Draft genome, SplittableRandom random);

    /**
     * Creates a mutation from a function.
     *
     * @param name its name
     * @param change the change
     * @return the mutation
     */
    static Mutation named(String name, BiConsumer<Genome.Draft, SplittableRandom> change) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(change, "change");
        return new Mutation() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public void apply(Genome.Draft genome, SplittableRandom random) {
                change.accept(genome, random);
            }

            @Override
            public String toString() {
                return name;
            }
        };
    }
}
