package io.github.zzuegg.brainiac.race;

import java.util.List;

/**
 * An immutable snapshot of a race, for drawing it.
 *
 * @param race the race number
 * @param step the step within the race
 * @param time the race time, seconds
 * @param cars every car
 */
public record RaceView(int race, int step, double time, List<CarView> cars) {

    /**
     * One car.
     *
     * @param id the car's number
     * @param x position
     * @param y position
     * @param heading where the nose points, radians
     * @param steer front wheel angle, radians
     * @param speed forward speed
     * @param crashed whether it is waiting to respawn
     * @param ghost whether it just respawned and neither collides nor drafts
     * @param draft how much air drag the slipstream removes, 0..1
     * @param sliding whether an axle slides past its grip
     * @param distance distance raced
     * @param lap the lap it is on, from 1
     * @param position its place in the race, from 1
     * @param collisions car-to-car contacts this race
     * @param overtakes cars passed this race
     * @param crashes crashes this race
     * @param rays wall distances along the rays
     * @param steerInput the steering asked for, -1..1
     * @param pedal the pedal asked for, -1 (brake) .. 1 (throttle)
     * @param probes the wall points the steerable sensors look at, {@code {x, y}} each,
     *     left sensors first
     */
    public record CarView(int id, double x, double y, double heading, double steer, double speed, boolean crashed,
                          boolean ghost, double draft, boolean sliding, double distance, int lap, int position,
                          int collisions, int overtakes, int crashes, double[] rays, double steerInput,
                          double pedal, double[][] probes) {
    }
}
