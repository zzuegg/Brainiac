package io.github.zzuegg.brainiac.race;

import java.util.Arrays;

/**
 * Every setting of a {@link Race}: the number of cars, the track, the car physics, the
 * slipstream, collisions, rewards and what each car senses. Setters return the config,
 * so settings chain:
 *
 * <pre>{@code
 * Race race = new Race(new RaceConfig().cars(12).track("serpent").rearGrip(420).nearestCars(4));
 * }</pre>
 *
 * <p>Distances are world units (a car is 18 long; the built-in tracks fit 1000 by 750),
 * times are seconds.
 */
public final class RaceConfig {
    int cars = 8;
    String track = "kidney";
    double halfWidth = 58;
    double dt = 1.0 / 30;
    int substeps = 6;
    int raceSteps = 1800;
    boolean randomStart = true;
    boolean shuffleGrid = true;
    double carLength = 18;
    double carWidth = 9;
    double engine = 220;
    double brake = 450;
    double maxSteer = 0.5;
    double steerRate = 5;
    double steerFalloff = 200;
    double frontGrip = 600;
    double rearGrip = 650;
    double peakSlip = 0.12;
    double slideGrip = 0.75;
    double drag = 0.0011;
    double rolling = 0.25;
    double draftLength = 170;
    double draftWidth = 16;
    double draftStrength = 0.5;
    boolean collisions = true;
    double restitution = 0.3;
    double ghostTime = 1.5;
    double stallTime = 3;
    double progressReward = 0.01;
    double crashPenalty = 5;
    double collisionPenalty = 0.01;
    double overtakeBonus = 1;
    double positionReward = 0;
    double finishReward = 10;
    int rays = 9;
    double raySpread = 75;
    double rayLength = 220;
    double[] probes = new double[] {80, 180, 320};
    int nearestCars = 5;
    double senseRadius = 260;
    double speedScale = 400;
    int wallProbes = 5;
    double probeMin = 20;
    double probeMax = 500;
    double probeRate = 600;

    /** A config with the default settings. */
    public RaceConfig() {
    }

    /** @return an independent copy */
    public RaceConfig copy() {
        RaceConfig c = new RaceConfig();
        c.cars = cars;
        c.track = track;
        c.halfWidth = halfWidth;
        c.dt = dt;
        c.substeps = substeps;
        c.raceSteps = raceSteps;
        c.randomStart = randomStart;
        c.shuffleGrid = shuffleGrid;
        c.carLength = carLength;
        c.carWidth = carWidth;
        c.engine = engine;
        c.brake = brake;
        c.maxSteer = maxSteer;
        c.steerRate = steerRate;
        c.steerFalloff = steerFalloff;
        c.frontGrip = frontGrip;
        c.rearGrip = rearGrip;
        c.peakSlip = peakSlip;
        c.slideGrip = slideGrip;
        c.drag = drag;
        c.rolling = rolling;
        c.draftLength = draftLength;
        c.draftWidth = draftWidth;
        c.draftStrength = draftStrength;
        c.collisions = collisions;
        c.restitution = restitution;
        c.ghostTime = ghostTime;
        c.stallTime = stallTime;
        c.progressReward = progressReward;
        c.crashPenalty = crashPenalty;
        c.collisionPenalty = collisionPenalty;
        c.overtakeBonus = overtakeBonus;
        c.positionReward = positionReward;
        c.finishReward = finishReward;
        c.rays = rays;
        c.raySpread = raySpread;
        c.rayLength = rayLength;
        c.probes = probes.clone();
        c.nearestCars = nearestCars;
        c.senseRadius = senseRadius;
        c.speedScale = speedScale;
        c.wallProbes = wallProbes;
        c.probeMin = probeMin;
        c.probeMax = probeMax;
        c.probeRate = probeRate;
        return c;
    }

    /**
     * Sets how many cars race; default 8.
     *
     * @param cars the value
     * @return this config
     */
    public RaceConfig cars(int cars) {
        if (!(cars > 0)) {
            throw new IllegalArgumentException("cars must be positive, got " + cars);
        }
        this.cars = cars;
        return this;
    }

    /**
     * Sets the built-in track: {@code kidney}, {@code oval} or {@code serpent}; default kidney.
     *
     * @param track the value
     * @return this config
     */
    public RaceConfig track(String track) {
        this.track = track;
        return this;
    }

    /**
     * Sets half the road width, in world units (the cars are 18 by 9); default 58.
     *
     * @param halfWidth the value
     * @return this config
     */
    public RaceConfig halfWidth(double halfWidth) {
        if (!(halfWidth > 0)) {
            throw new IllegalArgumentException("halfWidth must be positive, got " + halfWidth);
        }
        this.halfWidth = halfWidth;
        return this;
    }

    /**
     * Sets seconds per environment step; default 1.0 / 30.
     *
     * @param dt the value
     * @return this config
     */
    public RaceConfig dt(double dt) {
        if (!(dt > 0)) {
            throw new IllegalArgumentException("dt must be positive, got " + dt);
        }
        this.dt = dt;
        return this;
    }

    /**
     * Sets physics substeps per environment step; more is steadier; default 6.
     *
     * @param substeps the value
     * @return this config
     */
    public RaceConfig substeps(int substeps) {
        if (!(substeps > 0)) {
            throw new IllegalArgumentException("substeps must be positive, got " + substeps);
        }
        this.substeps = substeps;
        return this;
    }

    /**
     * Sets steps per race, after which every car's episode is cut off and a new race starts on the grid; default 1800.
     *
     * @param raceSteps the value
     * @return this config
     */
    public RaceConfig raceSteps(int raceSteps) {
        if (!(raceSteps > 0)) {
            throw new IllegalArgumentException("raceSteps must be positive, got " + raceSteps);
        }
        this.raceSteps = raceSteps;
        return this;
    }

    /**
     * Sets whether each race starts at a random point of the lap instead of the start line; default true.
     *
     * @param randomStart the value
     * @return this config
     */
    public RaceConfig randomStart(boolean randomStart) {
        this.randomStart = randomStart;
        return this;
    }

    /**
     * Sets whether the grid order is shuffled every race; default true.
     *
     * @param shuffleGrid the value
     * @return this config
     */
    public RaceConfig shuffleGrid(boolean shuffleGrid) {
        this.shuffleGrid = shuffleGrid;
        return this;
    }

    /**
     * Sets car length; default 18.
     *
     * @param carLength the value
     * @return this config
     */
    public RaceConfig carLength(double carLength) {
        if (!(carLength > 0)) {
            throw new IllegalArgumentException("carLength must be positive, got " + carLength);
        }
        this.carLength = carLength;
        return this;
    }

    /**
     * Sets car width; default 9.
     *
     * @param carWidth the value
     * @return this config
     */
    public RaceConfig carWidth(double carWidth) {
        if (!(carWidth > 0)) {
            throw new IllegalArgumentException("carWidth must be positive, got " + carWidth);
        }
        this.carWidth = carWidth;
        return this;
    }

    /**
     * Sets engine acceleration at full throttle, units/s^2 (rear-wheel drive); default 220.
     *
     * @param engine the value
     * @return this config
     */
    public RaceConfig engine(double engine) {
        if (!(engine > 0)) {
            throw new IllegalArgumentException("engine must be positive, got " + engine);
        }
        this.engine = engine;
        return this;
    }

    /**
     * Sets braking deceleration at full brake, units/s^2; default 450.
     *
     * @param brake the value
     * @return this config
     */
    public RaceConfig brake(double brake) {
        if (!(brake > 0)) {
            throw new IllegalArgumentException("brake must be positive, got " + brake);
        }
        this.brake = brake;
        return this;
    }

    /**
     * Sets the largest front wheel angle, radians; default 0.5.
     *
     * @param maxSteer the value
     * @return this config
     */
    public RaceConfig maxSteer(double maxSteer) {
        if (!(maxSteer > 0)) {
            throw new IllegalArgumentException("maxSteer must be positive, got " + maxSteer);
        }
        this.maxSteer = maxSteer;
        return this;
    }

    /**
     * Sets how fast the front wheels turn, radians per second; default 5.
     *
     * @param steerRate the value
     * @return this config
     */
    public RaceConfig steerRate(double steerRate) {
        if (!(steerRate > 0)) {
            throw new IllegalArgumentException("steerRate must be positive, got " + steerRate);
        }
        this.steerRate = steerRate;
        return this;
    }

    /**
     * Sets speed at which the usable steering angle halves, so fast cars steer more gently; default 200.
     *
     * @param steerFalloff the value
     * @return this config
     */
    public RaceConfig steerFalloff(double steerFalloff) {
        if (!(steerFalloff > 0)) {
            throw new IllegalArgumentException("steerFalloff must be positive, got " + steerFalloff);
        }
        this.steerFalloff = steerFalloff;
        return this;
    }

    /**
     * Sets front axle grip: the most sideways acceleration it can give, units/s^2 (friction times gravity); default 600.
     *
     * @param frontGrip the value
     * @return this config
     */
    public RaceConfig frontGrip(double frontGrip) {
        if (!(frontGrip > 0)) {
            throw new IllegalArgumentException("frontGrip must be positive, got " + frontGrip);
        }
        this.frontGrip = frontGrip;
        return this;
    }

    /**
     * Sets rear axle grip, shared with the engine's drive force; at or below the front grip the car oversteers and drifts more easily; default 650.
     *
     * @param rearGrip the value
     * @return this config
     */
    public RaceConfig rearGrip(double rearGrip) {
        if (!(rearGrip > 0)) {
            throw new IllegalArgumentException("rearGrip must be positive, got " + rearGrip);
        }
        this.rearGrip = rearGrip;
        return this;
    }

    /**
     * Sets tyre slip angle (radians) at which an axle's grip peaks; default 0.12.
     *
     * @param peakSlip the value
     * @return this config
     */
    public RaceConfig peakSlip(double peakSlip) {
        if (!(peakSlip > 0)) {
            throw new IllegalArgumentException("peakSlip must be positive, got " + peakSlip);
        }
        this.peakSlip = peakSlip;
        return this;
    }

    /**
     * Sets fraction of the peak grip left once an axle is sliding well past its peak; default 0.75.
     *
     * @param slideGrip the value
     * @return this config
     */
    public RaceConfig slideGrip(double slideGrip) {
        if (!(slideGrip >= 0 && slideGrip <= 1)) {
            throw new IllegalArgumentException("slideGrip must be within 0..1, got " + slideGrip);
        }
        this.slideGrip = slideGrip;
        return this;
    }

    /**
     * Sets air drag coefficient: deceleration = drag * speed^2; default 0.0011.
     *
     * @param drag the value
     * @return this config
     */
    public RaceConfig drag(double drag) {
        if (!(drag >= 0)) {
            throw new IllegalArgumentException("drag must not be negative, got " + drag);
        }
        this.drag = drag;
        return this;
    }

    /**
     * Sets rolling resistance: deceleration = rolling * speed; default 0.25.
     *
     * @param rolling the value
     * @return this config
     */
    public RaceConfig rolling(double rolling) {
        if (!(rolling >= 0)) {
            throw new IllegalArgumentException("rolling must not be negative, got " + rolling);
        }
        this.rolling = rolling;
        return this;
    }

    /**
     * Sets how far behind another car the slipstream reaches; default 170.
     *
     * @param draftLength the value
     * @return this config
     */
    public RaceConfig draftLength(double draftLength) {
        if (!(draftLength >= 0)) {
            throw new IllegalArgumentException("draftLength must not be negative, got " + draftLength);
        }
        this.draftLength = draftLength;
        return this;
    }

    /**
     * Sets how far to either side of the car ahead's path the slipstream reaches; default 16.
     *
     * @param draftWidth the value
     * @return this config
     */
    public RaceConfig draftWidth(double draftWidth) {
        if (!(draftWidth >= 0)) {
            throw new IllegalArgumentException("draftWidth must not be negative, got " + draftWidth);
        }
        this.draftWidth = draftWidth;
        return this;
    }

    /**
     * Sets the most air drag the slipstream removes, right behind a car (0 turns drafting off); default 0.5.
     *
     * @param draftStrength the value
     * @return this config
     */
    public RaceConfig draftStrength(double draftStrength) {
        if (!(draftStrength >= 0 && draftStrength <= 1)) {
            throw new IllegalArgumentException("draftStrength must be within 0..1, got " + draftStrength);
        }
        this.draftStrength = draftStrength;
        return this;
    }

    /**
     * Sets whether cars collide with each other; default true.
     *
     * @param collisions the value
     * @return this config
     */
    public RaceConfig collisions(boolean collisions) {
        this.collisions = collisions;
        return this;
    }

    /**
     * Sets bounciness of car-to-car hits, 0 to 1; default 0.3.
     *
     * @param restitution the value
     * @return this config
     */
    public RaceConfig restitution(double restitution) {
        if (!(restitution >= 0 && restitution <= 1)) {
            throw new IllegalArgumentException("restitution must be within 0..1, got " + restitution);
        }
        this.restitution = restitution;
        return this;
    }

    /**
     * Sets seconds a respawned car neither collides nor drafts; default 1.5.
     *
     * @param ghostTime the value
     * @return this config
     */
    public RaceConfig ghostTime(double ghostTime) {
        if (!(ghostTime >= 0)) {
            throw new IllegalArgumentException("ghostTime must not be negative, got " + ghostTime);
        }
        this.ghostTime = ghostTime;
        return this;
    }

    /**
     * Sets seconds without progress after which a car's episode ends; default 3.
     *
     * @param stallTime the value
     * @return this config
     */
    public RaceConfig stallTime(double stallTime) {
        if (!(stallTime > 0)) {
            throw new IllegalArgumentException("stallTime must be positive, got " + stallTime);
        }
        this.stallTime = stallTime;
        return this;
    }

    /**
     * Sets reward per world unit of progress along the track; default 0.01.
     *
     * @param progressReward the value
     * @return this config
     */
    public RaceConfig progressReward(double progressReward) {
        if (!(progressReward >= 0)) {
            throw new IllegalArgumentException("progressReward must not be negative, got " + progressReward);
        }
        this.progressReward = progressReward;
        return this;
    }

    /**
     * Sets penalty for leaving the track or stalling; default 5.
     *
     * @param crashPenalty the value
     * @return this config
     */
    public RaceConfig crashPenalty(double crashPenalty) {
        if (!(crashPenalty >= 0)) {
            throw new IllegalArgumentException("crashPenalty must not be negative, got " + crashPenalty);
        }
        this.crashPenalty = crashPenalty;
        return this;
    }

    /**
     * Sets penalty per unit/s of impact speed in a car-to-car hit, for both cars; default 0.01.
     *
     * @param collisionPenalty the value
     * @return this config
     */
    public RaceConfig collisionPenalty(double collisionPenalty) {
        if (!(collisionPenalty >= 0)) {
            throw new IllegalArgumentException("collisionPenalty must not be negative, got " + collisionPenalty);
        }
        this.collisionPenalty = collisionPenalty;
        return this;
    }

    /**
     * Sets reward for passing a car, and penalty for being passed; default 1.
     *
     * @param overtakeBonus the value
     * @return this config
     */
    public RaceConfig overtakeBonus(double overtakeBonus) {
        if (!(overtakeBonus >= 0)) {
            throw new IllegalArgumentException("overtakeBonus must not be negative, got " + overtakeBonus);
        }
        this.overtakeBonus = overtakeBonus;
        return this;
    }

    /**
     * Sets wall distance rays, spread evenly over the ray fan; default 9.
     *
     * @param rays the value
     * @return this config
     */
    public RaceConfig rays(int rays) {
        if (!(rays >= 0)) {
            throw new IllegalArgumentException("rays must not be negative, got " + rays);
        }
        this.rays = rays;
        return this;
    }

    /**
     * Sets the angle of the outermost ray from straight ahead, degrees; default 75.
     *
     * @param raySpread the value
     * @return this config
     */
    public RaceConfig raySpread(double raySpread) {
        if (!(raySpread >= 0)) {
            throw new IllegalArgumentException("raySpread must not be negative, got " + raySpread);
        }
        this.raySpread = raySpread;
        return this;
    }

    /**
     * Sets how far the rays see; default 220.
     *
     * @param rayLength the value
     * @return this config
     */
    public RaceConfig rayLength(double rayLength) {
        if (!(rayLength > 0)) {
            throw new IllegalArgumentException("rayLength must be positive, got " + rayLength);
        }
        this.rayLength = rayLength;
        return this;
    }

    /**
     * Sets distances ahead along the track at which the road's direction is sensed; default 80, 180, 320.
     *
     * @param probes the value
     * @return this config
     */
    public RaceConfig probes(double... probes) {
        this.probes = probes.clone();
        return this;
    }

    /**
     * Sets how many of the nearest cars each car sees; default 5.
     *
     * @param nearestCars the value
     * @return this config
     */
    public RaceConfig nearestCars(int nearestCars) {
        if (!(nearestCars >= 0)) {
            throw new IllegalArgumentException("nearestCars must not be negative, got " + nearestCars);
        }
        this.nearestCars = nearestCars;
        return this;
    }

    /**
     * Sets how far away other cars are seen; default 260.
     *
     * @param senseRadius the value
     * @return this config
     */
    public RaceConfig senseRadius(double senseRadius) {
        if (!(senseRadius > 0)) {
            throw new IllegalArgumentException("senseRadius must be positive, got " + senseRadius);
        }
        this.senseRadius = senseRadius;
        return this;
    }

    /**
     * Sets speed that maps to 1 in observations; default 400.
     *
     * @param speedScale the value
     * @return this config
     */
    public RaceConfig speedScale(double speedScale) {
        if (!(speedScale > 0)) {
            throw new IllegalArgumentException("speedScale must be positive, got " + speedScale);
        }
        this.speedScale = speedScale;
        return this;
    }

    /** @return how many cars race */
    public int cars() {
        return cars;
    }

    /** @return the built-in track: {@code kidney}, {@code oval} or {@code serpent} */
    public String track() {
        return track;
    }

    /** @return half the road width, in world units (the cars are 18 by 9) */
    public double halfWidth() {
        return halfWidth;
    }

    /** @return seconds per environment step */
    public double dt() {
        return dt;
    }

    /** @return physics substeps per environment step; more is steadier */
    public int substeps() {
        return substeps;
    }

    /** @return steps per race, after which every car's episode is cut off and a new race starts on the grid */
    public int raceSteps() {
        return raceSteps;
    }

    /** @return whether each race starts at a random point of the lap instead of the start line */
    public boolean randomStart() {
        return randomStart;
    }

    /** @return whether the grid order is shuffled every race */
    public boolean shuffleGrid() {
        return shuffleGrid;
    }

    /** @return car length */
    public double carLength() {
        return carLength;
    }

    /** @return car width */
    public double carWidth() {
        return carWidth;
    }

    /** @return engine acceleration at full throttle, units/s^2 (rear-wheel drive) */
    public double engine() {
        return engine;
    }

    /** @return braking deceleration at full brake, units/s^2 */
    public double brake() {
        return brake;
    }

    /** @return the largest front wheel angle, radians */
    public double maxSteer() {
        return maxSteer;
    }

    /** @return how fast the front wheels turn, radians per second */
    public double steerRate() {
        return steerRate;
    }

    /** @return speed at which the usable steering angle halves, so fast cars steer more gently */
    public double steerFalloff() {
        return steerFalloff;
    }

    /** @return front axle grip: the most sideways acceleration it can give, units/s^2 (friction times gravity) */
    public double frontGrip() {
        return frontGrip;
    }

    /** @return rear axle grip, shared with the engine's drive force; at or below the front grip the car oversteers and drifts more easily */
    public double rearGrip() {
        return rearGrip;
    }

    /** @return tyre slip angle (radians) at which an axle's grip peaks */
    public double peakSlip() {
        return peakSlip;
    }

    /** @return fraction of the peak grip left once an axle is sliding well past its peak */
    public double slideGrip() {
        return slideGrip;
    }

    /** @return air drag coefficient: deceleration = drag * speed^2 */
    public double drag() {
        return drag;
    }

    /** @return rolling resistance: deceleration = rolling * speed */
    public double rolling() {
        return rolling;
    }

    /** @return how far behind another car the slipstream reaches */
    public double draftLength() {
        return draftLength;
    }

    /** @return how far to either side of the car ahead's path the slipstream reaches */
    public double draftWidth() {
        return draftWidth;
    }

    /** @return the most air drag the slipstream removes, right behind a car (0 turns drafting off) */
    public double draftStrength() {
        return draftStrength;
    }

    /** @return whether cars collide with each other */
    public boolean collisions() {
        return collisions;
    }

    /** @return bounciness of car-to-car hits, 0 to 1 */
    public double restitution() {
        return restitution;
    }

    /** @return seconds a respawned car neither collides nor drafts */
    public double ghostTime() {
        return ghostTime;
    }

    /** @return seconds without progress after which a car's episode ends */
    public double stallTime() {
        return stallTime;
    }

    /** @return reward per world unit of progress along the track */
    public double progressReward() {
        return progressReward;
    }

    /** @return penalty for leaving the track or stalling */
    public double crashPenalty() {
        return crashPenalty;
    }

    /** @return penalty per unit/s of impact speed in a car-to-car hit, for both cars */
    public double collisionPenalty() {
        return collisionPenalty;
    }

    /** @return reward for passing a car, and penalty for being passed */
    public double overtakeBonus() {
        return overtakeBonus;
    }

    /** @return wall distance rays, spread evenly over the ray fan */
    public int rays() {
        return rays;
    }

    /** @return the angle of the outermost ray from straight ahead, degrees */
    public double raySpread() {
        return raySpread;
    }

    /** @return how far the rays see */
    public double rayLength() {
        return rayLength;
    }

    /** @return distances ahead along the track at which the road's direction is sensed */
    public double[] probes() {
        return probes.clone();
    }

    /** @return how many of the nearest cars each car sees */
    public int nearestCars() {
        return nearestCars;
    }

    /** @return how far away other cars are seen */
    public double senseRadius() {
        return senseRadius;
    }

    /** @return speed that maps to 1 in observations */
    public double speedScale() {
        return speedScale;
    }

    /**
     * Sets the reward per step for the place in the race: the leader gets this much, last
     * place the same amount taken away, the places between in proportion; default 0 (off).
     *
     * @param positionReward the value
     * @return this config
     */
    public RaceConfig positionReward(double positionReward) {
        if (!(positionReward >= 0)) {
            throw new IllegalArgumentException("positionReward must not be negative, got " + positionReward);
        }
        this.positionReward = positionReward;
        return this;
    }

    /**
     * Sets the reward for the final standing when a race ends: the winner gets this much,
     * last place the same amount taken away, the places between in proportion; default 10.
     *
     * @param finishReward the value
     * @return this config
     */
    public RaceConfig finishReward(double finishReward) {
        if (!(finishReward >= 0)) {
            throw new IllegalArgumentException("finishReward must not be negative, got " + finishReward);
        }
        this.finishReward = finishReward;
        return this;
    }

    /** @return the reward per step for the place in the race (leader +, last -) */
    public double positionReward() {
        return positionReward;
    }

    /** @return the reward for the final standing (winner +, last -) */
    public double finishReward() {
        return finishReward;
    }

    /**
     * Sets how many steerable sensors each car has on each wall; default 5 (5 on the left,
     * 5 on the right). Each looks at the wall at a distance ahead along the track that the
     * car itself chooses with an extra action, so a car can look far ahead on a straight
     * and close in a bend. 0 turns them off.
     *
     * @param wallProbes sensors per side
     * @return this config
     */
    public RaceConfig wallProbes(int wallProbes) {
        if (wallProbes < 0) {
            throw new IllegalArgumentException("wallProbes must not be negative, got " + wallProbes);
        }
        this.wallProbes = wallProbes;
        return this;
    }

    /**
     * Sets the range of distances ahead the steerable wall sensors can look; default 20 to
     * 500.
     *
     * @param min the nearest
     * @param max the farthest
     * @return this config
     */
    public RaceConfig probeRange(double min, double max) {
        if (!(min >= 0 && max > min)) {
            throw new IllegalArgumentException("need 0 <= min < max, got " + min + ".." + max);
        }
        this.probeMin = min;
        this.probeMax = max;
        return this;
    }

    /**
     * Sets how fast a steerable sensor moves towards the distance the car asks for, units
     * per second; default 600.
     *
     * @param probeRate the speed
     * @return this config
     */
    public RaceConfig probeRate(double probeRate) {
        if (!(probeRate > 0)) {
            throw new IllegalArgumentException("probeRate must be positive, got " + probeRate);
        }
        this.probeRate = probeRate;
        return this;
    }

    /** @return steerable wall sensors per side */
    public int wallProbes() {
        return wallProbes;
    }

    /** @return the nearest distance a steerable sensor can look ahead */
    public double probeMin() {
        return probeMin;
    }

    /** @return the farthest distance a steerable sensor can look ahead */
    public double probeMax() {
        return probeMax;
    }

    /** @return how fast a steerable sensor moves, units per second */
    public double probeRate() {
        return probeRate;
    }

    /** @return the number of values in each car's action */
    public int actionSize() {
        return 2 + 2 * wallProbes;
    }

    /** @return the number of values each car observes */
    public int observationSize() {
        return Race.OWN + rays + probes.length + 2 * wallProbes * Race.PER_PROBE + nearestCars * Race.PER_CAR;
    }

    @Override
    public String toString() {
        return cars + " cars on " + track + ", grip " + frontGrip + "/" + rearGrip + ", draft " + draftStrength
                + ", collisions " + collisions + ", observations " + observationSize()
                + " (probes " + Arrays.toString(probes) + ")";
    }
}
