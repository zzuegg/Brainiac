package io.github.zzuegg.brainiac.evolve;

import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Combines two parent genomes into a child.
 */
@FunctionalInterface
public interface Crossover {

    /**
     * Lines the parents up by neuron id, in the style of NEAT. The child has the regions,
     * neurons and projections of the fitter parent. A neuron or connection both parents
     * have takes its activation from either at random; a connection only the other parent
     * has is added with probability one half when both its neurons exist in the child.
     * Hyperparameters come from either parent at random.
     */
    Crossover UNIFORM = (fitter, other, random) -> {
        Genome.Draft child = fitter.draft();
        Map<Integer, Genome.Neuron> theirs = new HashMap<>();
        for (Genome.Region r : other.regions()) {
            for (Genome.Neuron n : r.neurons()) {
                theirs.put(n.id(), n);
            }
        }
        for (Genome.Draft.RegionDraft r : child.regions) {
            for (int i = 0; i < r.neurons.size(); i++) {
                Genome.Neuron mine = r.neurons.get(i);
                Genome.Neuron other2 = theirs.get(mine.id());
                if (other2 != null && random.nextBoolean()) {
                    r.neurons.set(i, other2);
                }
            }
        }
        for (Genome.Projection p : other.projections()) {
            Genome.Draft.ProjectionDraft mine = child.projection(p.from(), p.to());
            if (mine == null) {
                continue;
            }
            for (long k : p.connections()) {
                if (!mine.connections.contains(k) && random.nextBoolean()) {
                    mine.connections.add(k);   // dangling ones are dropped by freeze()
                }
            }
        }
        if (random.nextBoolean()) {
            child.ticks = other.ticks();
        }
        if (random.nextBoolean()) {
            child.learningRate = other.learningRate();
        }
        if (random.nextBoolean()) {
            child.optimizer = other.optimizer();
        }
        return child.freeze();
    };

    /** No crossover: the child is a copy of the fitter parent, changed only by mutation. */
    Crossover NONE = (fitter, other, random) -> fitter;

    /**
     * @param fitter the parent with the higher fitness
     * @param other the other parent
     * @param random the only source of randomness to use
     * @return the child
     */
    Genome cross(Genome fitter, Genome other, SplittableRandom random);
}
