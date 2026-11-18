package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEnvironment;

import java.util.SplittableRandom;

/**
 * A two-agent task with heterogeneous roles: a scout (agent 0) sees which of {@code sites}
 * sites holds the cargo and can only signal; a carrier (agent 1) sees nothing but the scout's
 * signal and can only drive to a site. On the first step the scout signals (the carrier's action
 * is ignored), on the second the carrier drives (the scout's action is ignored), and the team is
 * paid 1 if the carrier reaches the cargo, minus a small fuel cost for driving to the far sites.
 *
 * <p>Both agents' observations carry a one-hot role and the step; the global state is the cargo
 * site, the signal and the step.
 */
public final class ScoutCarrier implements MultiAgentEnvironment {

    private final int sites;
    private int cargo;
    private int signal = -1;
    private int t;

    /** @param sites sites (and signals) */
    public ScoutCarrier(int sites) {
        this.sites = sites;
    }

    /** A scout and carrier with four sites. */
    public ScoutCarrier() {
        this(4);
    }

    @Override
    public int agents() {
        return 2;
    }

    @Override
    public int observationSize() {
        return sites + 3;
    }

    @Override
    public int stateSize() {
        return 2 * sites + 1;
    }

    @Override
    public int actions() {
        return sites;
    }

    private float[][] observations() {
        float[][] o = new float[2][sites + 3];
        o[0][cargo] = 1;
        if (signal >= 0) {
            o[1][signal] = 1;
        }
        o[0][sites] = 1;
        o[1][sites + 1] = 1;
        o[0][sites + 2] = t;
        o[1][sites + 2] = t;
        return o;
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        cargo = random.nextInt(sites);
        signal = -1;
        t = 0;
        return observations();
    }

    @Override
    public float[] state() {
        float[] s = new float[2 * sites + 1];
        s[cargo] = 1;
        if (signal >= 0) {
            s[sites + signal] = 1;
        }
        s[2 * sites] = t;
        return s;
    }

    @Override
    public Step step(int[] actions) {
        if (t == 0) {
            signal = actions[0];
            t = 1;
            return new Step(observations(), 0, false, false);
        }
        int site = actions[1];
        float reward = (site == cargo ? 1 : 0) - 0.05f * site / sites;
        t = 2;
        return new Step(observations(), reward, true, false);
    }

    @Override
    public String toString() {
        return "ScoutCarrier(" + sites + ")";
    }
}
