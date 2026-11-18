package io.github.zzuegg.brainiac.experimental.security;

/**
 * A federated server's aggregation rule: it turns the flattened client updates of one round
 * {@code Δ_1 ... Δ_n} into the one update the server applies. FedAvg is the mean; the rules of
 * {@link RobustAggregators}, {@link PreAggregation} and {@link LearnedAggregationWeights}
 * replace it with statistics that a bounded share of malicious clients cannot steer.
 *
 * <p>Updates arrive in a fixed client order every round, so stateful rules (centred clipping,
 * worker momentum, learned weights) may keep per-slot state.
 */
@FunctionalInterface
public interface Aggregator {

    /**
     * Aggregates one round.
     *
     * @param updates one flattened update per client, all of the same length
     * @return the aggregate update
     */
    float[] aggregate(float[][] updates);

    /**
     * Aggregates one round when the clients also report their local losses at the global
     * model; rules that ignore losses use {@link #aggregate(float[][])}.
     *
     * @param updates one flattened update per client
     * @param losses each client's reported local loss
     * @return the aggregate update
     */
    default float[] aggregate(float[][] updates, double[] losses) {
        return aggregate(updates);
    }

    /** @return FedAvg's unweighted mean */
    static Aggregator mean() {
        return RobustAggregators::mean;
    }
}
