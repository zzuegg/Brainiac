package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/**
 * Balance a pole on a cart by pushing the cart left (action 0) or right (action 1). The
 * reward is 1 for every step the pole stays up; the episode ends when the pole tilts more
 * than 12 degrees, the cart leaves the track, or after {@code maxSteps} steps (500 by
 * default). Observation: cart position, cart velocity, pole angle, pole angular velocity.
 * The physics are the classic Barto, Sutton and Anderson (1983) cart-pole equations.
 */
public final class CartPole implements Environment {

    private static final double GRAVITY = 9.8;
    private static final double CART_MASS = 1.0;
    private static final double POLE_MASS = 0.1;
    private static final double TOTAL_MASS = CART_MASS + POLE_MASS;
    private static final double HALF_LENGTH = 0.5;
    private static final double POLE_MOMENT = POLE_MASS * HALF_LENGTH;
    private static final double FORCE = 10;
    private static final double DT = 0.02;
    private static final double ANGLE_LIMIT = 12 * 2 * Math.PI / 360;
    private static final double POSITION_LIMIT = 2.4;

    private final int maxSteps;
    private double x;
    private double xDot;
    private double theta;
    private double thetaDot;
    private int steps;
    private boolean over = true;

    /** A cart-pole cut off after 500 steps. */
    public CartPole() {
        this(500);
    }

    /**
     * @param maxSteps the most steps per episode
     */
    public CartPole(int maxSteps) {
        if (maxSteps < 1) {
            throw new IllegalArgumentException("maxSteps must be at least 1");
        }
        this.maxSteps = maxSteps;
    }

    @Override
    public Space observationSpace() {
        float inf = Float.POSITIVE_INFINITY;
        return Space.box(new float[] {-4.8f, -inf, -0.42f, -inf}, new float[] {4.8f, inf, 0.42f, inf});
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(2);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        x = random.nextDouble(-0.05, 0.05);
        xDot = random.nextDouble(-0.05, 0.05);
        theta = random.nextDouble(-0.05, 0.05);
        thetaDot = random.nextDouble(-0.05, 0.05);
        steps = 0;
        over = false;
        return observation();
    }

    private float[] observation() {
        return new float[] {(float) x, (float) xDot, (float) theta, (float) thetaDot};
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        int a = Math.round(action[0]);
        if (a != 0 && a != 1) {
            throw new IllegalArgumentException("cart-pole actions are 0 (left) and 1 (right), got " + action[0]);
        }
        double force = a == 1 ? FORCE : -FORCE;
        double cos = Math.cos(theta);
        double sin = Math.sin(theta);
        double temp = (force + POLE_MOMENT * thetaDot * thetaDot * sin) / TOTAL_MASS;
        double thetaAcc = (GRAVITY * sin - cos * temp)
                / (HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos * cos / TOTAL_MASS));
        double xAcc = temp - POLE_MOMENT * thetaAcc * cos / TOTAL_MASS;
        x += DT * xDot;
        xDot += DT * xAcc;
        theta += DT * thetaDot;
        thetaDot += DT * thetaAcc;
        steps++;
        boolean fell = Math.abs(x) > POSITION_LIMIT || Math.abs(theta) > ANGLE_LIMIT;
        boolean cut = !fell && steps >= maxSteps;
        over = fell || cut;
        return new Step(observation(), 1, fell, cut);
    }

    @Override
    public String render() {
        int width = 41;
        int cart = (int) Math.round((x + POSITION_LIMIT) / (2 * POSITION_LIMIT) * (width - 1));
        cart = Math.max(0, Math.min(width - 1, cart));
        StringBuilder track = new StringBuilder("-".repeat(width));
        track.setCharAt(cart, theta < -0.05 ? '\\' : theta > 0.05 ? '/' : '|');
        return track + String.format("  angle %+5.1f deg", Math.toDegrees(theta));
    }

    @Override
    public String toString() {
        return "CartPole";
    }
}
