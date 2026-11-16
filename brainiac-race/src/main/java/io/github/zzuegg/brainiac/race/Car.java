package io.github.zzuegg.brainiac.race;

/**
 * One car's state and its tyre physics: a bicycle model with a front and a rear axle.
 * Each axle's sideways force grows with its slip angle up to the axle's grip and drops
 * once it slides, and the rear axle shares its grip between driving and cornering
 * (a friction circle), so too much throttle in a bend breaks the rear loose.
 */
final class Car {

    enum Status { ACTIVE, CRASHED }

    final int id;
    // world state
    double x;
    double y;
    double heading;
    double vx;
    double vy;
    double omega;
    // controls
    double steerInput;
    double pedal;
    double steer;
    // track state
    int index;
    double along;
    double lateral;
    double distance;
    double best;
    double stall;
    double ghost;
    Status status = Status.ACTIVE;
    boolean fresh;
    // per step
    double draft;
    double impact;
    double slipFront;
    double slipRear;
    boolean sliding;
    int collisions;
    int overtakes;
    int crashes;
    final double[] rays;

    Car(int id, int rays) {
        this.id = id;
        this.rays = new double[rays];
    }

    double forwardSpeed() {
        return vx * Math.cos(heading) + vy * Math.sin(heading);
    }

    double sideSpeed() {
        return -vx * Math.sin(heading) + vy * Math.cos(heading);
    }

    double speed() {
        return Math.hypot(vx, vy);
    }

    boolean solid() {
        return status == Status.ACTIVE && ghost <= 0;
    }

    void place(double x, double y, double heading, double speed) {
        this.x = x;
        this.y = y;
        this.heading = heading;
        this.vx = Math.cos(heading) * speed;
        this.vy = Math.sin(heading) * speed;
        this.omega = 0;
        this.steer = 0;
        this.steerInput = 0;
        this.pedal = 0;
        this.stall = 0;
        this.status = Status.ACTIVE;
    }

    /** Advances the car by {@code h} seconds. */
    void integrate(RaceConfig c, double h) {
        double cos = Math.cos(heading);
        double sin = Math.sin(heading);
        double u = vx * cos + vy * sin;
        double v = -vx * sin + vy * cos;
        double wheelbase = 0.7 * c.carLength;
        double a = wheelbase / 2;
        double b = wheelbase / 2;
        double inertia = (c.carLength * c.carLength + c.carWidth * c.carWidth) / 12;

        double target = steerInput * c.maxSteer / (1 + Math.abs(u) / c.steerFalloff);
        steer += Math.clamp(target - steer, -c.steerRate * h, c.steerRate * h);

        // longitudinal: rear-wheel drive; brakes split 60 / 40 and never push the car backwards
        double throttle = Math.max(0, pedal);
        double brakeForce = Math.min(Math.max(0, -pedal) * c.brake, Math.abs(u) / h);
        double brakeSigned = Math.signum(u) * brakeForce;
        double frontMax = c.frontGrip * b / wheelbase;
        double rearMax = c.rearGrip * a / wheelbase;
        double fxRear = Math.clamp(throttle * c.engine - 0.4 * brakeSigned, -rearMax, rearMax);
        double fxFront = Math.clamp(-0.6 * brakeSigned, -frontMax, frontMax);

        // lateral: slip angle of each axle, and what grip is left after the longitudinal force
        double ue = Math.max(Math.abs(u), 1);
        double dir = u >= 0 ? 1 : -1;
        slipFront = Math.atan2(v + a * omega, ue) - steer * dir;
        slipRear = Math.atan2(v - b * omega, ue);
        double capFront = Math.sqrt(Math.max(0, frontMax * frontMax - fxFront * fxFront));
        double capRear = Math.sqrt(Math.max(0, rearMax * rearMax - fxRear * fxRear));
        // tyres only build sideways force once the car rolls
        double rolling = Math.min(1, Math.abs(u) / 10);
        double fyFront = -tyre(slipFront, capFront, c) * rolling;
        double fyRear = -tyre(slipRear, capRear, c) * rolling;
        sliding = Math.abs(u) > 30 && (Math.abs(slipRear) > c.peakSlip || Math.abs(slipFront) > c.peakSlip);

        // air drag (less in a slipstream) and rolling resistance, against the direction of travel
        double speed = Math.hypot(u, v);
        double resist = c.drag * (1 - draft) * speed * speed + c.rolling * speed;
        double rx = speed > 1e-9 ? -resist * u / speed : 0;
        double ry = speed > 1e-9 ? -resist * v / speed : 0;

        double cs = Math.cos(steer);
        double ss = Math.sin(steer);
        double du = fxRear + fxFront * cs - fyFront * ss + rx + v * omega;
        double dv = fyRear + fyFront * cs + fxFront * ss + ry - u * omega;
        double dw = (a * (fyFront * cs + fxFront * ss) - b * fyRear) / inertia;
        double u2 = u + du * h;
        double v2 = v + dv * h;
        double w2 = omega + dw * h;
        if (throttle == 0 && u != 0 && Math.signum(u2) != Math.signum(u)) {
            u2 = 0;
        }
        // below walking pace the tyre model is stiff and noisy, so blend into rolling without slip
        double k = Math.clamp((Math.abs(u2) - 15) / 45, 0, 1);
        double kinematic = u2 * Math.tan(steer) / wheelbase;
        omega = k * w2 + (1 - k) * kinematic;
        v2 *= k;
        heading = Track.wrap(heading + omega * h);
        cos = Math.cos(heading);
        sin = Math.sin(heading);
        vx = u2 * cos - v2 * sin;
        vy = u2 * sin + v2 * cos;
        x += vx * h;
        y += vy * h;
    }

    private static double tyre(double slip, double cap, RaceConfig c) {
        double s = Math.abs(slip) / c.peakSlip;
        double f = s <= 1 ? s : Math.max(c.slideGrip, 1 - (1 - c.slideGrip) * (s - 1) / 2);
        return Math.signum(slip) * cap * f;
    }
}
