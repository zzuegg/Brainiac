package io.github.zzuegg.brainiac.race;

import io.github.zzuegg.brainiac.rl.Agent;

/** Drives a car the learner does not control; see {@link Race#withOpponents(int, Driver)}. */
@FunctionalInterface
public interface Driver {

    /**
     * @param race the race
     * @param car the car to drive
     * @return its action {@code {steer, pedal}}
     */
    float[] drive(Race race, int car);

    /**
     * The built-in hand-written driver, each car keeping to its own lane.
     *
     * @return the driver
     */
    static Driver bot() {
        return (race, car) -> race.botAction(car, Math.sin(car * 2.4) * 0.5);
    }

    /**
     * A driver that asks an agent (for example a trained or frozen policy) for actions.
     *
     * @param agent the agent; it acts greedily
     * @return the driver
     */
    static Driver of(Agent agent) {
        return (race, car) -> agent.act(race.observe(car), false);
    }
}
