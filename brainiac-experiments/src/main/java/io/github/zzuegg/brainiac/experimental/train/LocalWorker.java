package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;

/**
 * One simulated worker (client, device or data-centre island) of a distributed training
 * run: a replica of the model with its own data shard and inner optimizer. Used by
 * {@link FederatedAveraging} and {@link DiLoCo}, which run all workers in-process.
 */
public interface LocalWorker {

    /** @return the replica's parameters, in the same order as the global model's */
    List<Var> parameters();

    /** @return how many training samples the worker holds (the averaging weight) */
    int samples();

    /**
     * Trains locally.
     *
     * @param steps optimizer steps to take
     * @return the mean training loss of those steps
     */
    float train(int steps);
}
