package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/**
 * Land a small craft on a pad in low gravity, with a main engine and two side thrusters:
 * a harder control task with delayed consequences, in the spirit of the classic
 * lunar-lander benchmark (with its own, simpler physics).
 *
 * <p>The craft starts high above the pad, drifting and slightly tilted. Actions: 0 do
 * nothing, 1 fire the left thruster (pushes right and turns clockwise), 2 fire the main
 * engine, 3 fire the right thruster. Its two legs touch the ground through springs; a
 * landing faster than 3 m/s breaks them, and a tilted one tips the craft over.
 * Observation (8 values): position and velocity relative to the pad (scaled to about
 * [-1, 1]), angle, angular velocity, and whether each leg touches the ground.
 *
 * <p>Reward: getting closer to the pad, slower and more upright earns reward and moving
 * away loses it; each leg touching the ground earns 10; the main engine costs 0.3 a step
 * and a thruster 0.03. Coming to rest on the legs ends the episode with +100, touching the
 * ground with the body, breaking a leg or flying off the sides with -100. Episodes are cut off after
 * {@code maxSteps} (1000 by default). Like the classic task, an average of 200 counts as
 * solved; doing nothing scores far below 0.
 */
public final class Lander implements Environment {

    private static final double DT = 0.02;
    private static final int SUBSTEPS = 10;
    private static final double GRAVITY = 4;
    private static final double MAIN_THRUST = 9;
    private static final double SIDE_THRUST = 1.5;
    private static final double SIDE_TURN = 2.5;
    private static final double INERTIA = 0.4;
    private static final double LEG_SPREAD = 0.6;
    private static final double LEG_DROP = 0.7;
    private static final double BODY_HALF_WIDTH = 0.5;
    private static final double BODY_HALF_HEIGHT = 0.35;
    private static final double SPRING = 300;
    private static final double DAMPING = 25;
    private static final double FRICTION = 0.8;
    private static final double LEG_BREAK = 3;
    private static final double START_HEIGHT = 14;
    private static final double WIDTH = 10;
    private static final double PAD_HALF_WIDTH = 1;

    private final int maxSteps;
    private double x;
    private double y;
    private double vx;
    private double vy;
    private double angle;
    private double spin;
    private boolean leftDown;
    private boolean rightDown;
    private double shaping;
    private int steps;
    private boolean over = true;

    /** A lander cut off after 1000 steps. */
    public Lander() {
        this(1000);
    }

    /**
     * @param maxSteps the most steps per episode
     */
    public Lander(int maxSteps) {
        if (maxSteps < 1) {
            throw new IllegalArgumentException("maxSteps must be at least 1");
        }
        this.maxSteps = maxSteps;
    }

    @Override
    public Space observationSpace() {
        float inf = Float.POSITIVE_INFINITY;
        return Space.box(new float[] {-1, -0.1f, -inf, -inf, -(float) Math.PI, -inf, 0, 0},
                new float[] {1, 1.5f, inf, inf, (float) Math.PI, inf, 1, 1});
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(4);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        x = random.nextDouble(-3, 3);
        y = START_HEIGHT;
        vx = random.nextDouble(-1.5, 1.5);
        vy = random.nextDouble(-1.5, 0.5);
        angle = random.nextDouble(-0.15, 0.15);
        spin = random.nextDouble(-0.2, 0.2);
        leftDown = false;
        rightDown = false;
        steps = 0;
        over = false;
        shaping = shaping();
        return observation();
    }

    private float[] observation() {
        return new float[] {(float) (x / WIDTH), (float) (y / WIDTH), (float) (vx / 5), (float) (vy / 5),
                (float) angle, (float) (spin / 2), leftDown ? 1 : 0, rightDown ? 1 : 0};
    }

    private double shaping() {
        double nx = x / WIDTH;
        double ny = y / WIDTH;
        double nvx = vx / 5;
        double nvy = vy / 5;
        return -100 * Math.sqrt(nx * nx + ny * ny) - 100 * Math.sqrt(nvx * nvx + nvy * nvy)
                - 100 * Math.abs(angle) + 10 * (leftDown ? 1 : 0) + 10 * (rightDown ? 1 : 0);
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        int a = Math.round(action[0]);
        if (a < 0 || a > 3) {
            throw new IllegalArgumentException("lander actions are 0 to 3, got " + action[0]);
        }
        boolean crashed = false;
        double h = DT / SUBSTEPS;
        for (int s = 0; s < SUBSTEPS && !crashed; s++) {
            double sin = Math.sin(angle);
            double cos = Math.cos(angle);
            double ax = 0;
            double ay = -GRAVITY;
            double turn = 0;
            if (a == 2) {                      // main engine along the craft's up direction
                ax += -sin * MAIN_THRUST;
                ay += cos * MAIN_THRUST;
            } else if (a == 1 || a == 3) {     // thruster: push sideways and turn
                double dir = a == 1 ? 1 : -1;
                ax += dir * cos * SIDE_THRUST;
                ay += dir * sin * SIDE_THRUST;
                turn -= dir * SIDE_TURN;
            }
            leftDown = false;
            rightDown = false;
            for (int leg = -1; leg <= 1; leg += 2) {
                // the leg tip in world coordinates, and its velocity
                double ox = leg * LEG_SPREAD;
                double oy = -LEG_DROP;
                double tx = x + ox * cos - oy * sin;
                double ty = y + ox * sin + oy * cos;
                if (ty >= 0) {
                    continue;
                }
                double rx = tx - x;
                double ry = ty - y;
                double tvx = vx - spin * ry;
                double tvy = vy + spin * rx;
                if (tvy < -LEG_BREAK) {
                    crashed = true;             // a leg hit the ground too hard and broke
                }
                double up = Math.max(0, SPRING * -ty - DAMPING * tvy);
                double side = -Math.signum(tvx) * Math.min(FRICTION * up, Math.abs(tvx) / h);
                ax += side;
                ay += up;
                turn += (rx * up - ry * side) / INERTIA;
                if (leg < 0) {
                    leftDown = true;
                } else {
                    rightDown = true;
                }
            }
            vx += ax * h;
            vy += ay * h;
            spin += turn * h;
            spin *= 1 - 0.1 * h;
            x += vx * h;
            y += vy * h;
            angle += spin * h;
            crashed |= bodyTouches();
        }
        steps++;
        double previous = shaping;
        shaping = shaping();
        double reward = shaping - previous - (a == 2 ? 0.3 : a == 0 ? 0 : 0.03);
        boolean offSide = Math.abs(x) > WIDTH;
        boolean rest = leftDown && rightDown && Math.abs(vx) < 0.05 && Math.abs(vy) < 0.05 && Math.abs(spin) < 0.05;
        boolean terminated = crashed || offSide || rest;
        if (crashed || offSide) {
            reward = -100;
        } else if (rest) {
            reward += 100;
        }
        boolean cut = !terminated && steps >= maxSteps;
        over = terminated || cut;
        return new Step(observation(), (float) reward, terminated, cut);
    }

    private boolean bodyTouches() {
        double sin = Math.sin(angle);
        double cos = Math.cos(angle);
        for (int cx = -1; cx <= 1; cx += 2) {
            for (int cy = -1; cy <= 1; cy += 2) {
                double ox = cx * BODY_HALF_WIDTH;
                double oy = cy * BODY_HALF_HEIGHT;
                if (y + ox * sin + oy * cos < 0) {
                    return true;
                }
            }
        }
        return false;
    }

    /** @return whether the craft is over the pad */
    public boolean overPad() {
        return Math.abs(x) <= PAD_HALF_WIDTH;
    }

    @Override
    public String render() {
        int width = 41;
        int col = (int) Math.round((x + WIDTH) / (2 * WIDTH) * (width - 1));
        col = Math.max(0, Math.min(width - 1, col));
        StringBuilder ground = new StringBuilder("_".repeat(width));
        int pad = (int) Math.round(PAD_HALF_WIDTH / (2 * WIDTH) * (width - 1));
        for (int i = width / 2 - pad; i <= width / 2 + pad; i++) {
            ground.setCharAt(i, '=');
        }
        ground.setCharAt(col, angle < -0.1 ? '\\' : angle > 0.1 ? '/' : 'A');
        return ground + String.format("  height %5.1f  speed %4.1f  tilt %+5.1f deg%s", y, Math.hypot(vx, vy),
                Math.toDegrees(angle), leftDown || rightDown ? "  touching" : "");
    }

    @Override
    public String toString() {
        return "Lander";
    }
}
