package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A cliff-walk variant for risk-sensitive RL: from the start the agent takes either a short
 * bridge (action 0) or a long muddy road (action 1). On each of the bridge's cells the wind
 * blows the agent off with a small probability ({@code -1}, episode over); crossing pays
 * {@code +1}. The road is safe but every step on it costs {@code mud}. With the defaults the
 * bridge has the higher expected return (about 0.77 against 0.7) but a fat lower tail, so a
 * risk-neutral agent crosses the bridge and a CVaR-averse one ({@link ImplicitQuantileDqn}
 * with {@code alpha = 0.25}) takes the road. After the first choice both actions move
 * forward. Observations are one-hot cells.
 */
public final class RiskyBridge implements Environment {

    private final int bridge;
    private final int road;
    private final double wind;
    private final float mud;
    private SplittableRandom random = new SplittableRandom(0);
    private int route;
    private int position;
    private int falls;
    private int crossings;

    /**
     * @param bridge bridge length
     * @param road road length
     * @param wind the chance per bridge cell of being blown off
     * @param mud the cost of every road step
     */
    public RiskyBridge(int bridge, int road, double wind, float mud) {
        this.bridge = bridge;
        this.road = road;
        this.wind = wind;
        this.mud = mud;
    }

    /** @return a bridge of 3 with 4% wind, a road of 6 at 0.05 per step */
    public static RiskyBridge standard() {
        return new RiskyBridge(3, 6, 0.04, 0.05f);
    }

    @Override
    public Space observationSpace() {
        return Space.box(1 + bridge + road, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(2);
    }

    @Override
    public float[] reset(SplittableRandom r) {
        this.random = r;
        route = -1;
        position = 0;
        return observation();
    }

    private float[] observation() {
        float[] o = new float[1 + bridge + road];
        if (route < 0) {
            o[0] = 1;
        } else {
            o[route == 0 ? position : bridge + position] = 1;
        }
        return o;
    }

    @Override
    public Step step(float[] action) {
        if (route < 0) {
            route = Math.round(action[0]) == 0 ? 0 : 1;
        }
        position++;
        if (route == 0) {
            if (random.nextDouble() < wind) {
                falls++;
                return new Step(observation(), -1, true, false);
            }
            if (position >= bridge) {
                crossings++;
                return new Step(observation(), 1, true, false);
            }
            return new Step(observation(), 0, false, false);
        }
        if (position >= road) {
            return new Step(observation(), 1 - mud, true, false);
        }
        return new Step(observation(), -mud, false, false);
    }

    /** @return how often the agent was blown off the bridge */
    public int falls() {
        return falls;
    }

    /** @return how often the bridge was crossed safely */
    public int crossings() {
        return crossings;
    }

    @Override
    public String toString() {
        return "RiskyBridge(" + bridge + ", " + road + ")";
    }
}
