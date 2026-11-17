package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

class OfflineRlTest {

    static GridWorld grid() {
        return GridWorld.of("S...", "....", "...G").maxSteps(20);
    }

    /** Right along the top row, then down: optimal on {@link #grid()}. */
    static Agent expert(double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return (obs, explore) -> {
            if (r.nextDouble() < noise) {
                return new float[] {r.nextInt(4)};
            }
            int cell = 0;
            while (obs[cell] == 0) {
                cell++;
            }
            return new float[] {cell % 4 < 3 ? 1 : 2};
        };
    }

    /** Mostly random play with some decent episodes mixed in. */
    static OfflineDataset mixed() {
        return OfflineDataset.collect(grid(), Agent.random(grid().actionSpace(), 1), 120, 1)
                .merge(OfflineDataset.collect(grid(), expert(0.5, 2), 30, 2));
    }

    @Test
    void behaviourCloningImitatesAnExpert() {
        OfflineDataset data = OfflineDataset.collect(grid(), expert(0, 1), 10, 1);
        BehaviorCloning bc = BehaviorCloning.discrete(Nets.mlp(12, 4, Activation.RELU, 1, 32), 4).train(data, 200);
        assertTrue(Rl.evaluate(bc, grid(), 1, 1) > 0.9);
    }

    @Test
    void cqlAndIqlImproveOnMediocreData() {
        OfflineDataset data = mixed();
        double behaviour = data.meanReturn();
        BehaviorCloning bc = BehaviorCloning.discrete(Nets.mlp(12, 4, Activation.RELU, 1, 32), 4).train(data, 300);
        Cql cql = Cql.of(() -> Nets.mlp(12, 4, Activation.RELU, 1, 32), 4).alpha(0.5f).discount(0.9f).seed(1)
                .train(data, 1000);
        Iql iql = Iql.of(12, 4, RlNetworkFactory.mlp(Activation.RELU, 32)).seed(1).train(data, 1500);
        double bcScore = Rl.evaluate(bc, grid(), 1, 1);
        double cqlScore = Rl.evaluate(cql, grid(), 1, 1);
        double iqlScore = Rl.evaluate(iql, grid(), 1, 1);
        System.out.printf("offline: data %.3f  BC %.3f  CQL %.3f  IQL %.3f%n", behaviour, bcScore, cqlScore, iqlScore);
        assertTrue(cqlScore > 0.9, "CQL " + cqlScore);
        assertTrue(iqlScore > 0.9, "IQL " + iqlScore);
        assertTrue(cqlScore > behaviour + 0.3 && cqlScore > bcScore);
    }

    @Test
    void decisionTransformerFollowsTheAskedReturn() {
        OfflineDataset data = mixed();
        DecisionTransformer dt = DecisionTransformer.of(12, 4, 4, 20, 32, 1, 3).seed(1).train(data, 300);
        double high = dt.play(grid(), 0.95f, 1);
        System.out.printf("DT: data %.3f, asked 0.95 got %.3f, loss %.3f%n", data.meanReturn(), high, dt.lastLoss());
        assertTrue(high > 0.9, "DT " + high);
    }
}
