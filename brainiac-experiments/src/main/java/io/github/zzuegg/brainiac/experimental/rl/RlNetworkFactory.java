package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;

/**
 * Builds the networks of an agent with several of them (actor, critics, mixers): given
 * the input size, output size and a seed, a fresh model.
 */
@FunctionalInterface
public interface RlNetworkFactory {

    /**
     * Builds a network.
     *
     * @param inputs values in
     * @param outputs values out
     * @param seed the initialization seed
     * @return the network
     */
    Model create(int inputs, int outputs, long seed);

    /**
     * Plain multi-layer perceptrons with a linear output.
     *
     * @param activation the hidden activation
     * @param hidden the hidden layer sizes
     * @return the factory
     */
    static RlNetworkFactory mlp(Activation activation, int... hidden) {
        int[] h = hidden.clone();
        return (inputs, outputs, seed) -> Nets.mlp(inputs, outputs, activation, seed, h);
    }
}
