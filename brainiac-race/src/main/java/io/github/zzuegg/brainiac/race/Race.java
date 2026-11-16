package io.github.zzuegg.brainiac.race;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.VectorEnvironment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Several cars racing on one track. Each car is one entry of this
 * {@link VectorEnvironment}: it observes its own state, the walls and the cars around it,
 * steers and works the pedal, and is rewarded for progress along the track.
 *
 * <p><b>Physics.</b> Each car is a bicycle model with a front and a rear axle. An axle's
 * sideways grip grows with its slip angle up to the axle's grip limit and falls off once
 * it slides, and the driven rear axle shares its grip between accelerating and cornering,
 * so cars understeer when they brake late and oversteer or drift when they power out of
 * a bend ({@link RaceConfig#frontGrip(double)}, {@link RaceConfig#rearGrip(double)}).
 *
 * <p><b>Slipstream.</b> Air drag limits top speed. A car close behind another car,
 * in its path, loses up to {@link RaceConfig#draftStrength(double)} of its drag, so it
 * gains on the car ahead and can pull out and pass it.
 *
 * <p><b>Collisions.</b> Cars are two circles each and bounce off each other with
 * {@link RaceConfig#restitution(double)}, which also spins them; both cars are penalised
 * by the impact speed. A car that leaves the road, stalls or turns back crashes: its
 * episode ends and it respawns on the centre line, as a ghost for a moment.
 *
 * <p><b>Actions:</b> {@code {steer, pedal, sensor...}}, each in -1..1; steer +1 turns
 * right, pedal above 0 is throttle and below 0 is brake. Then one value per steerable
 * wall sensor ({@link RaceConfig#wallProbes(int)} per side, left first) sets how far ahead
 * it looks, from {@link RaceConfig#probeMin()} (-1) to {@link RaceConfig#probeMax()} (+1);
 * the sensor glides there at {@link RaceConfig#probeRate()}.
 *
 * <p><b>Observations</b> ({@link RaceConfig#observationSize()} values, all in the car's own
 * frame and scaled to about -1..1):
 * <ol>
 *   <li>own state ({@value #OWN}): forward speed, sideways speed, yaw rate, wheel angle,
 *       position across the road, heading relative to the road (sin, cos), slipstream;</li>
 *   <li>one wall distance per ray, fanned out ahead;</li>
 *   <li>the road's direction at each probe distance ahead, relative to the car's heading;</li>
 *   <li>for each steerable wall sensor ({@value #PER_PROBE} values): where the wall is at
 *       the distance it looks ahead, ahead of and to the right of the car, and that
 *       distance;</li>
 *   <li>for each of the nearest cars ({@value #PER_CAR} values each, nearest first; zeros
 *       when fewer are in range): seen flag, position ahead and to the right, velocity
 *       relative to this car ahead and to the right, its heading relative to this car
 *       (sin, cos) and its yaw rate.</li>
 * </ol>
 *
 * <p><b>Episodes.</b> A race lasts {@link RaceConfig#raceSteps(int)} steps; then every
 * car's episode is cut off (truncated) and the next {@link #reset(int, SplittableRandom)}
 * lines all cars up on a fresh grid. A crash ends only that car's episode.
 */
public final class Race implements VectorEnvironment {

    /** Observed values about the car itself. */
    public static final int OWN = 8;
    /** Observed values per nearby car. */
    public static final int PER_CAR = 8;
    /** Observed values per steerable wall sensor. */
    public static final int PER_PROBE = 3;

    private final RaceConfig config;
    private final Track track;
    private final Car[] cars;
    private final int obsSize;
    private final double[] rayAngles;
    private final double[][] lastGap;
    private final double[] reward;
    private final boolean[] crashedNow;
    private SplittableRandom random = new SplittableRandom(0);
    private int steps;
    private int races;
    private boolean raceOver = true;
    private volatile RaceView view;
    private volatile java.util.function.Consumer<RaceView> listener;

    /**
     * @param config the settings (copied)
     */
    public Race(RaceConfig config) {
        this(config, Track.named(config.track, config.halfWidth));
    }

    /**
     * @param config the settings (copied); its track name and width are ignored
     * @param track the track to race on
     */
    public Race(RaceConfig config, Track track) {
        this.config = config.copy();
        this.track = track;
        int n = this.config.cars;
        cars = new Car[n];
        for (int i = 0; i < n; i++) {
            cars[i] = new Car(i, this.config.rays, 2 * this.config.wallProbes);
            resetProbes(cars[i]);
        }
        obsSize = this.config.observationSize();
        rayAngles = new double[this.config.rays];
        for (int k = 0; k < rayAngles.length; k++) {
            double f = rayAngles.length == 1 ? 0 : -1 + 2.0 * k / (rayAngles.length - 1);
            rayAngles[k] = Math.toRadians(this.config.raySpread) * f;
        }
        lastGap = new double[n][n];
        reward = new double[n];
        crashedNow = new boolean[n];
        newRace();
        races = 0;
    }

    /** @return the settings (a copy) */
    public RaceConfig config() {
        return config.copy();
    }

    /** @return the track */
    public Track track() {
        return track;
    }

    /** @return how many races have started */
    public int races() {
        return races;
    }

    /** @return the steps taken in the current race */
    public int raceStep() {
        return steps;
    }

    @Override
    public int count() {
        return cars.length;
    }

    @Override
    public Space observationSpace() {
        return Space.box(obsSize, -2, 2);
    }

    @Override
    public Space actionSpace() {
        return Space.box(config.actionSize(), -1, 1);
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        this.random = random;
        newRace();
        float[][] obs = new float[cars.length][];
        for (int i = 0; i < cars.length; i++) {
            cars[i].fresh = false;
            obs[i] = observe(i);
        }
        publish();
        return obs;
    }

    @Override
    public float[] reset(int index, SplittableRandom random) {
        this.random = random;
        if (raceOver) {
            newRace();
        }
        Car c = cars[index];
        if (c.status == Car.Status.CRASHED) {
            respawn(c);
        }
        c.fresh = false;
        publish();
        return observe(index);
    }

    @Override
    public Environment.Step[] step(float[][] actions) {
        if (actions.length != cars.length) {
            throw new IllegalArgumentException("expected " + cars.length + " actions, got " + actions.length);
        }
        double probeStep = config.probeRate * config.dt;
        if (raceOver) {
            newRace();
        }
        RaceConfig c = config;
        for (int i = 0; i < cars.length; i++) {
            Car car = cars[i];
            if (car.status == Car.Status.CRASHED) {
                respawn(car);
            }
            car.steerInput = Math.clamp(actions[i][0], -1, 1);
            car.pedal = Math.clamp(actions[i][1], -1, 1);
            // the rest of the action points the steerable wall sensors; a shorter action
            // (such as the bot's) leaves them where they are
            for (int k = 0; k < car.probe.length; k++) {
                if (actions[i].length > 2 + k) {
                    double f = (Math.clamp(actions[i][2 + k], -1, 1) + 1) / 2;
                    car.probeTarget[k] = config.probeMin + f * (config.probeMax - config.probeMin);
                }
                car.probe[k] += Math.clamp(car.probeTarget[k] - car.probe[k], -probeStep, probeStep);
            }
            car.impact = 0;
            reward[i] = 0;
            crashedNow[i] = false;
        }
        updateDraft();
        double h = c.dt / c.substeps;
        double[] before = new double[cars.length];
        for (int i = 0; i < cars.length; i++) {
            before[i] = cars[i].distance;
        }
        for (int s = 0; s < c.substeps; s++) {
            for (Car car : cars) {
                if (car.status == Car.Status.ACTIVE) {
                    car.integrate(c, h);
                    locate(car);
                }
            }
            if (c.collisions) {
                collide();
            }
            for (Car car : cars) {
                if (car.status == Car.Status.ACTIVE && Math.abs(car.lateral) > track.halfWidth() - c.carWidth / 2) {
                    crash(car);
                }
            }
        }
        steps++;
        for (int i = 0; i < cars.length; i++) {
            Car car = cars[i];
            car.ghost = Math.max(0, car.ghost - c.dt);
            if (car.status != Car.Status.ACTIVE) {
                continue;
            }
            double progress = car.distance - before[i];
            reward[i] += c.progressReward * progress;
            reward[i] -= c.collisionPenalty * car.impact;
            if (car.distance > car.best + 0.5) {
                car.best = car.distance;
                car.stall = 0;
            } else {
                car.stall += c.dt;
            }
            if (car.stall > c.stallTime || car.distance < car.best - 4 * c.carLength) {
                crash(car);
            }
        }
        overtakes();
        for (Car car : cars) {
            if (car.status == Car.Status.ACTIVE) {
                sense(car);
            }
        }
        boolean over = steps >= c.raceSteps;
        if (cars.length > 1 && (c.positionReward > 0 || over && c.finishReward > 0)) {
            // the place in the race: +1 for the leader down to -1 for last
            int[] rank = ranks();
            for (int i = 0; i < cars.length; i++) {
                double place = (cars.length + 1 - 2.0 * rank[i]) / (cars.length - 1);
                if (cars[i].status == Car.Status.ACTIVE) {
                    reward[i] += c.positionReward * place;
                }
                if (over) {
                    reward[i] += c.finishReward * place;
                }
            }
        }
        Environment.Step[] out = new Environment.Step[cars.length];
        for (int i = 0; i < cars.length; i++) {
            if (crashedNow[i]) {
                reward[i] -= c.crashPenalty;
            }
            out[i] = new Environment.Step(observe(i), (float) reward[i], crashedNow[i], over && !crashedNow[i]);
        }
        if (over) {
            raceOver = true;
        }
        publish();
        return out;
    }

    /**
     * Sets up a new race: all cars on a two-wide staggered grid, standing still.
     */
    private void newRace() {
        RaceConfig c = config;
        steps = 0;
        races++;
        raceOver = false;
        int start = c.randomStart ? random.nextInt(track.samples()) : 0;
        int[] order = new int[cars.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        if (c.shuffleGrid) {
            for (int i = order.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
        double startAlong = start * track.segment();
        for (int slot = 0; slot < cars.length; slot++) {
            Car car = cars[order[slot]];
            int row = slot / 2;
            int col = slot % 2;
            double back = c.carLength + row * 2.2 * c.carLength + col * 1.1 * c.carLength;
            double side = (col == 0 ? -0.4 : 0.4) * Math.min(track.halfWidth(), 4 * c.carWidth);
            double along = startAlong - back;
            placeAt(car, along, side, 0);
            car.distance = -back;
            car.best = car.distance;
            car.ghost = 0;
            car.fresh = true;
            resetProbes(car);
            car.collisions = 0;
            car.overtakes = 0;
            car.crashes = 0;
        }
        for (Car car : cars) {
            locate(car);
            car.best = car.distance;
            car.draft = 0;
            sense(car);
        }
        for (int i = 0; i < cars.length; i++) {
            for (int j = 0; j < cars.length; j++) {
                lastGap[i][j] = cars[i].distance - cars[j].distance;
            }
        }
    }

    /** Puts a car somewhere on the track, for tests: arc length from the start line, offset to the right. */
    void place(int index, double along, double side, double speed) {
        Car car = cars[index];
        placeAt(car, along, side, speed);
        car.distance = along;
        locate(car);
        car.distance = along;
        car.best = along;
        car.ghost = 0;
        sense(car);
        for (int j = 0; j < cars.length; j++) {
            lastGap[index][j] = car.distance - cars[j].distance;
            lastGap[j][index] = -lastGap[index][j];
        }
        publish();
    }

    /** @return the car's current slipstream drag reduction, for tests */
    double draft(int index) {
        return cars[index].draft;
    }

    /** The wall point a steerable sensor looks at: left sensors first, then right. */
    private double[] probePoint(Car car, int p) {
        int i = car.index + (int) Math.round(car.probe[p] / track.segment());
        return p < car.probe.length / 2 ? track.left(i) : track.right(i);
    }

    /** Spreads a car's steerable sensors evenly over their range. */
    private void resetProbes(Car car) {
        int side = car.probe.length / 2;
        for (int p = 0; p < car.probe.length; p++) {
            int k = p % Math.max(1, side);
            double f = side <= 1 ? 0.5 : k / (double) (side - 1);
            car.probe[p] = config.probeMin + f * (config.probeMax - config.probeMin);
            car.probeTarget[p] = car.probe[p];
        }
    }

    /**
     * Where one car's steerable wall sensors look.
     *
     * @param index the car
     * @return one {@code {x, y}} wall point per sensor, left sensors first
     */
    public double[][] probePoints(int index) {
        Car car = cars[index];
        double[][] out = new double[car.probe.length][];
        for (int p = 0; p < out.length; p++) {
            out[p] = probePoint(car, p);
        }
        return out;
    }

    private void placeAt(Car car, double along, double side, double speed) {
        double l = track.length();
        double s = ((along % l) + l) % l;
        double f = s / track.segment();
        int i = (int) Math.floor(f);
        double t = f - i;
        double[] p0 = track.center(i);
        double[] p1 = track.center(i + 1);
        double[] tan = track.tangent(i);
        double x = p0[0] + (p1[0] - p0[0]) * t - tan[1] * side;
        double y = p0[1] + (p1[1] - p0[1]) * t + tan[0] * side;
        car.index = i;
        car.along = s;
        car.place(x, y, track.heading(i), speed);
    }

    private void respawn(Car car) {
        double back = 3 * track.segment();
        placeAt(car, car.along - back, 0, 60);
        car.distance -= back;
        car.best = car.distance;
        car.ghost = config.ghostTime;
        car.draft = 0;
        locate(car);
        car.distance = car.best;
        sense(car);
        for (int j = 0; j < cars.length; j++) {
            lastGap[car.id][j] = car.distance - cars[j].distance;
            lastGap[j][car.id] = -lastGap[car.id][j];
        }
    }

    private void crash(Car car) {
        if (car.status == Car.Status.CRASHED) {
            return;
        }
        car.status = Car.Status.CRASHED;
        car.crashes++;
        crashedNow[car.id] = true;
    }

    /** Updates where the car is along and across the track, and its race distance. */
    private void locate(Car car) {
        Track.Projection p = track.project(car.x, car.y, car.index, 4, 8);
        double delta = p.along() - car.along;
        double l = track.length();
        if (delta > l / 2) {
            delta -= l;
        } else if (delta < -l / 2) {
            delta += l;
        }
        car.distance += delta;
        car.along = p.along();
        car.index = p.index();
        car.lateral = p.lateral();
    }

    private void updateDraft() {
        RaceConfig c = config;
        for (Car car : cars) {
            car.draft = 0;
            if (!car.solid() || c.draftStrength == 0) {
                continue;
            }
            double hx = Math.cos(car.heading);
            double hy = Math.sin(car.heading);
            for (Car other : cars) {
                if (other == car || !other.solid()) {
                    continue;
                }
                double dx = other.x - car.x;
                double dy = other.y - car.y;
                double ahead = dx * hx + dy * hy;
                double side = Math.abs(-dx * hy + dy * hx);
                if (ahead <= c.carLength * 0.5 || ahead > c.draftLength || side > c.draftWidth) {
                    continue;
                }
                double align = Math.cos(other.heading - car.heading);
                if (align < 0.7) {
                    continue;
                }
                double f = c.draftStrength * (1 - ahead / c.draftLength) * (1 - 0.5 * side / c.draftWidth);
                car.draft = Math.max(car.draft, f);
            }
        }
    }

    private void collide() {
        RaceConfig c = config;
        double r = c.carWidth / 2;
        double off = Math.max(0, c.carLength / 2 - r);
        double inertia = (c.carLength * c.carLength + c.carWidth * c.carWidth) / 12;
        for (int i = 0; i < cars.length; i++) {
            Car a = cars[i];
            if (!a.solid()) {
                continue;
            }
            for (int j = i + 1; j < cars.length; j++) {
                Car b = cars[j];
                if (!b.solid()) {
                    continue;
                }
                double dx = b.x - a.x;
                double dy = b.y - a.y;
                if (dx * dx + dy * dy > (c.carLength + c.carWidth) * (c.carLength + c.carWidth)) {
                    continue;
                }
                // deepest overlap between a's and b's circles
                double best = 0;
                double nx = 0;
                double ny = 0;
                double px = 0;
                double py = 0;
                for (int sa = -1; sa <= 1; sa += 2) {
                    double ax = a.x + Math.cos(a.heading) * off * sa;
                    double ay = a.y + Math.sin(a.heading) * off * sa;
                    for (int sb = -1; sb <= 1; sb += 2) {
                        double bx = b.x + Math.cos(b.heading) * off * sb;
                        double by = b.y + Math.sin(b.heading) * off * sb;
                        double ex = bx - ax;
                        double ey = by - ay;
                        double d = Math.hypot(ex, ey);
                        double pen = 2 * r - d;
                        if (pen > best) {
                            best = pen;
                            if (d < 1e-9) {
                                ex = dx;
                                ey = dy;
                                d = Math.max(1e-9, Math.hypot(ex, ey));
                            }
                            nx = ex / d;
                            ny = ey / d;
                            px = (ax + bx) / 2;
                            py = (ay + by) / 2;
                        }
                    }
                }
                if (best <= 0) {
                    continue;
                }
                // push apart, then exchange an impulse at the contact point
                a.x -= nx * best / 2;
                a.y -= ny * best / 2;
                b.x += nx * best / 2;
                b.y += ny * best / 2;
                double rax = px - a.x;
                double ray = py - a.y;
                double rbx = px - b.x;
                double rby = py - b.y;
                double vax = a.vx - a.omega * ray;
                double vay = a.vy + a.omega * rax;
                double vbx = b.vx - b.omega * rby;
                double vby = b.vy + b.omega * rbx;
                double rvx = vbx - vax;
                double rvy = vby - vay;
                double vn = rvx * nx + rvy * ny;
                if (vn >= 0) {
                    continue;
                }
                double raN = rax * ny - ray * nx;
                double rbN = rbx * ny - rby * nx;
                double jn = -(1 + c.restitution) * vn / (2 + raN * raN / inertia + rbN * rbN / inertia);
                // a little sideways friction between the bodies
                double tx = -ny;
                double ty = nx;
                double vt = rvx * tx + rvy * ty;
                double raT = rax * ty - ray * tx;
                double rbT = rbx * ty - rby * tx;
                double jt = Math.clamp(-vt / (2 + raT * raT / inertia + rbT * rbT / inertia), -0.3 * jn, 0.3 * jn);
                double ix = jn * nx + jt * tx;
                double iy = jn * ny + jt * ty;
                a.vx -= ix;
                a.vy -= iy;
                a.omega -= (rax * iy - ray * ix) / inertia;
                b.vx += ix;
                b.vy += iy;
                b.omega += (rbx * iy - rby * ix) / inertia;
                a.impact = Math.max(a.impact, -vn);
                b.impact = Math.max(b.impact, -vn);
                if (-vn > 10) {
                    a.collisions++;
                    b.collisions++;
                }
            }
        }
    }

    private void overtakes() {
        RaceConfig c = config;
        double near = 4 * c.carLength;
        for (int i = 0; i < cars.length; i++) {
            for (int j = i + 1; j < cars.length; j++) {
                Car a = cars[i];
                Car b = cars[j];
                double gap = a.distance - b.distance;
                double was = lastGap[i][j];
                lastGap[i][j] = gap;
                lastGap[j][i] = -gap;
                if (!a.solid() || !b.solid() || Math.signum(gap) == Math.signum(was) || gap == 0 || was == 0) {
                    continue;
                }
                if (Math.hypot(a.x - b.x, a.y - b.y) > near) {
                    continue;
                }
                Car passer = gap > 0 ? a : b;
                Car passed = gap > 0 ? b : a;
                passer.overtakes++;
                reward[passer.id] += c.overtakeBonus;
                reward[passed.id] -= c.overtakeBonus;
            }
        }
    }

    private void sense(Car car) {
        for (int k = 0; k < rayAngles.length; k++) {
            car.rays[k] = track.raycast(car.x, car.y, car.heading + rayAngles[k], config.rayLength, car.index);
        }
    }

    /**
     * What one car sees right now.
     *
     * @param index the car
     * @return its observation
     */
    public float[] observe(int index) {
        RaceConfig c = config;
        Car car = cars[index];
        float[] o = new float[obsSize];
        double hx = Math.cos(car.heading);
        double hy = Math.sin(car.heading);
        double vs = c.speedScale;
        int k = 0;
        o[k++] = (float) (car.forwardSpeed() / vs);
        o[k++] = (float) (car.sideSpeed() / vs);
        o[k++] = (float) Math.clamp(car.omega / 3, -2, 2);
        o[k++] = (float) (car.steer / c.maxSteer);
        o[k++] = (float) Math.clamp(car.lateral / track.halfWidth(), -2, 2);
        double rel = Track.wrap(car.heading - track.heading(car.index));
        o[k++] = (float) Math.sin(rel);
        o[k++] = (float) Math.cos(rel);
        o[k++] = (float) (c.draftStrength > 0 ? car.draft / c.draftStrength : 0);
        for (double r : car.rays) {
            o[k++] = (float) (r / c.rayLength);
        }
        for (double p : c.probes) {
            int i = car.index + (int) Math.round(p / track.segment());
            o[k++] = (float) Math.clamp(Track.wrap(track.heading(i) - car.heading) / (Math.PI / 2), -1, 1);
        }
        // steerable wall sensors: where the wall is at the chosen distance ahead, in the
        // car's frame, and the distance itself
        for (int p = 0; p < car.probe.length; p++) {
            double[] w = probePoint(car, p);
            double dx = w[0] - car.x;
            double dy = w[1] - car.y;
            o[k++] = (float) Math.clamp((dx * hx + dy * hy) / c.probeMax, -2, 2);
            o[k++] = (float) Math.clamp((-dx * hy + dy * hx) / c.probeMax, -2, 2);
            o[k++] = (float) ((car.probe[p] - c.probeMin) / (c.probeMax - c.probeMin) * 2 - 1);
        }
        // the nearest cars, nearest first
        if (c.nearestCars > 0) {
            Integer[] others = new Integer[cars.length];
            double[] dist = new double[cars.length];
            int m = 0;
            for (Car other : cars) {
                if (other == car || other.status != Car.Status.ACTIVE) {
                    continue;
                }
                double d = Math.hypot(other.x - car.x, other.y - car.y);
                if (d <= c.senseRadius) {
                    dist[other.id] = d;
                    others[m++] = other.id;
                }
            }
            Integer[] seen = Arrays.copyOf(others, m);
            Arrays.sort(seen, (p, q) -> Double.compare(dist[p], dist[q]));
            for (int s = 0; s < c.nearestCars; s++) {
                if (s >= seen.length) {
                    k += PER_CAR;
                    continue;
                }
                Car other = cars[seen[s]];
                double dx = other.x - car.x;
                double dy = other.y - car.y;
                double dvx = other.vx - car.vx;
                double dvy = other.vy - car.vy;
                double dh = other.heading - car.heading;
                o[k++] = 1;
                o[k++] = (float) ((dx * hx + dy * hy) / c.senseRadius);
                o[k++] = (float) ((-dx * hy + dy * hx) / c.senseRadius);
                o[k++] = (float) ((dvx * hx + dvy * hy) / vs);
                o[k++] = (float) ((-dvx * hy + dvy * hx) / vs);
                o[k++] = (float) Math.sin(dh);
                o[k++] = (float) Math.cos(dh);
                o[k++] = (float) Math.clamp(other.omega / 3, -2, 2);
            }
        }
        return o;
    }

    /**
     * What a simple hand-written driver with full track knowledge would do: aim at a point
     * on the racing line ahead and brake for the tightest bend coming up. Useful as an
     * opponent and as a yardstick.
     *
     * @param index the car
     * @param lane where across the road to drive, -1 (left edge) to 1 (right edge)
     * @return its action {@code {steer, pedal}}
     */
    public float[] botAction(int index, double lane) {
        RaceConfig c = config;
        Car car = cars[index];
        double v = car.forwardSpeed();
        int ahead = car.index + (int) Math.round((40 + v * 0.3) / track.segment());
        double[] p = track.center(ahead);
        double[] t = track.tangent(ahead);
        double off = lane * 0.6 * track.halfWidth();
        double px = p[0] - t[1] * off;
        double py = p[1] + t[0] * off;
        double steer = Math.clamp(Track.wrap(Math.atan2(py - car.y, px - car.x) - car.heading) * 3, -1, 1);
        double grip = 0.75 * Math.min(c.frontGrip, c.rearGrip);
        double target = 1e9;
        int look = (int) Math.round(600 / track.segment());
        for (int k = 0; k <= look; k++) {
            double kap = Math.abs(track.curvature(car.index + k));
            if (kap < 1e-4) {
                continue;
            }
            double vmax = Math.sqrt(grip / kap);
            double ok = Math.sqrt(vmax * vmax + 2 * 0.8 * c.brake * k * track.segment());
            target = Math.min(target, ok);
        }
        double pedal = v < target - 8 ? 1 : v < target ? 0.3 : v > target + 8 ? -1 : 0;
        return new float[] {(float) steer, (float) pedal};
    }

    /**
     * Lets some cars be driven by something other than the learner: the learner sees and
     * drives the first {@code learners} cars, and {@code driver} drives the rest.
     *
     * @param learners how many cars the learner drives
     * @param driver drives every other car
     * @return a vector environment of the learner's cars
     */
    public VectorEnvironment withOpponents(int learners, Driver driver) {
        if (learners < 1 || learners > cars.length) {
            throw new IllegalArgumentException("learners must be within 1.." + cars.length + ", got " + learners);
        }
        Race race = this;
        return new VectorEnvironment() {
            @Override
            public int count() {
                return learners;
            }

            @Override
            public Space observationSpace() {
                return race.observationSpace();
            }

            @Override
            public Space actionSpace() {
                return race.actionSpace();
            }

            @Override
            public float[][] reset(SplittableRandom random) {
                return Arrays.copyOf(race.reset(random), learners);
            }

            @Override
            public float[] reset(int index, SplittableRandom random) {
                return race.reset(index, random);
            }

            @Override
            public Environment.Step[] step(float[][] actions) {
                float[][] all = new float[cars.length][];
                System.arraycopy(actions, 0, all, 0, learners);
                for (int i = learners; i < cars.length; i++) {
                    all[i] = driver.drive(race, i);
                }
                Environment.Step[] steps = race.step(all);
                for (int i = learners; i < cars.length; i++) {
                    if (steps[i].terminated()) {
                        race.reset(i, random);
                    }
                }
                return Arrays.copyOf(steps, learners);
            }
        };
    }

    private void publish() {
        List<RaceView.CarView> list = new ArrayList<>(cars.length);
        int[] rank = ranks();
        for (Car car : cars) {
            list.add(new RaceView.CarView(car.id, car.x, car.y, car.heading, car.steer, car.forwardSpeed(),
                    car.status == Car.Status.CRASHED, car.ghost > 0, car.draft, car.sliding, car.distance,
                    (int) Math.floor(car.distance / track.length()) + 1, rank[car.id], car.collisions,
                    car.overtakes, car.crashes, car.rays.clone(), car.steerInput, car.pedal, probePoints(car.id)));
        }
        RaceView v = new RaceView(races, steps, steps * config.dt, List.copyOf(list));
        view = v;
        java.util.function.Consumer<RaceView> l = listener;
        if (l != null) {
            l.accept(v);
        }
    }

    /**
     * Calls a listener with a snapshot after every step and reset, on the thread that
     * steps the race. A window uses this to pace or pause the race.
     *
     * @param listener the listener, or {@code null} for none
     */
    public void onStep(java.util.function.Consumer<RaceView> listener) {
        this.listener = listener;
    }

    private int[] ranks() {
        Integer[] order = new Integer[cars.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (p, q) -> Double.compare(cars[q].distance, cars[p].distance));
        int[] rank = new int[cars.length];
        for (int r = 0; r < order.length; r++) {
            rank[order[r]] = r + 1;
        }
        return rank;
    }

    /**
     * The latest state of every car, safe to read from another thread (for example a
     * window drawing the race while another thread trains).
     *
     * @return the snapshot after the last step or reset
     */
    public RaceView view() {
        return view;
    }

    /** @return the ray angles relative to the car's heading, radians */
    public double[] rayAngles() {
        return rayAngles.clone();
    }

    @Override
    public String toString() {
        return "Race(" + config + ", track " + track + ")";
    }
}
