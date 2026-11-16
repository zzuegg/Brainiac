package io.github.zzuegg.brainiac.race;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.VectorEnvironment;

import java.awt.image.BufferedImage;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Track geometry, car physics, slipstream, collisions, observations and episodes. */
class RaceTest {

    private static Environment.Step[] drive(Race race, float steer, float pedal) {
        float[][] a = new float[race.count()][];
        for (int i = 0; i < a.length; i++) {
            a[i] = new float[] {steer, pedal};
        }
        return race.step(a);
    }

    /** A wide oval with no randomness, so a car can drive about freely. */
    private static RaceConfig open(int cars) {
        return new RaceConfig().cars(cars).track("oval").halfWidth(2000).randomStart(false).shuffleGrid(false)
                .stallTime(100);
    }

    @Test
    void trackWallsAreHalfAWidthFromTheCentreLine() {
        Track t = Track.kidney(50);
        assertTrue(t.length() > 2000, "lap length " + t.length());
        for (int i = 0; i < t.samples(); i += 37) {
            double[] c = t.center(i);
            Track.Projection p = t.project(c[0], c[1], i, 2, 2);
            assertEquals(0, p.lateral(), 1e-6);
            double[] l = t.left(i);
            double[] r = t.right(i);
            assertEquals(50, Math.hypot(l[0] - c[0], l[1] - c[1]), 1e-6);
            assertTrue(t.project(r[0], r[1], i, 2, 2).lateral() > 0, "the right wall is to the right");
            double side = t.heading(i) + Math.PI / 2;
            assertEquals(50, t.raycast(c[0], c[1], side, 500, i), 2.5);
        }
        assertThrows(IllegalArgumentException.class, () -> Track.named("moon", 50));
    }

    @Test
    void aCarAcceleratesToTheSpeedWhereDragMatchesTheEngine() {
        Race race = new Race(open(1).track("serpent").halfWidth(58));
        race.reset(new SplittableRandom(1));
        RaceConfig c = race.config();
        race.place(0, 0, 0, 0);
        double last = 0;
        for (int t = 0; t < 30; t++) {
            drive(race, 0, 1);
            double v = race.view().cars().getFirst().speed();
            assertTrue(v > last, "speeds up");
            last = v;
        }
        // engine = drag * v^2 + rolling * v
        double top = (-c.rolling() + Math.sqrt(c.rolling() * c.rolling() + 4 * c.drag() * c.engine())) / (2 * c.drag());
        Race wide = new Race(open(1));
        wide.reset(new SplittableRandom(1));
        for (int t = 0; t < 900; t++) {
            drive(wide, 0, 1);
        }
        assertEquals(top, wide.view().cars().getFirst().speed(), 3);
    }

    @Test
    void gentleSteeringGripsAndFullLockWithThrottleSlides() {
        Race race = new Race(open(1));
        race.reset(new SplittableRandom(1));
        race.place(0, 0, 0, 250);
        int sliding = 0;
        for (int t = 0; t < 30; t++) {
            drive(race, 0.15f, 0.3f);
            sliding += race.view().cars().getFirst().sliding() ? 1 : 0;
        }
        assertEquals(0, sliding, "a gentle bend at speed stays within the grip");
        race.place(0, 0, 0, 250);
        double turned = 0;
        double heading = race.view().cars().getFirst().heading();
        for (int t = 0; t < 30; t++) {
            drive(race, 1, 1);
            sliding += race.view().cars().getFirst().sliding() ? 1 : 0;
        }
        turned = Math.abs(Track.wrap(race.view().cars().getFirst().heading() - heading));
        assertTrue(sliding > 10, "full lock and full throttle at speed slides, " + sliding + " steps");
        assertTrue(turned > 0.5, "and the car rotates, " + turned);
    }

    @Test
    void lessRearGripSpinsTheCarMore() {
        double[] yaw = new double[2];
        double[] grips = {700, 380};
        for (int k = 0; k < 2; k++) {
            Race race = new Race(open(1).rearGrip(grips[k]));
            race.reset(new SplittableRandom(1));
            race.place(0, 0, 0, 220);
            double max = 0;
            for (int t = 0; t < 20; t++) {
                Environment.Step s = race.step(new float[][] {{0.6f, 1}})[0];
                max = Math.max(max, Math.abs(s.observation()[1]));
            }
            yaw[k] = max;
        }
        assertTrue(yaw[1] > yaw[0] * 1.2, "a loose rear axle slides sideways more: " + yaw[0] + " vs " + yaw[1]);
    }

    /** A long, thin loop: straight from about 120 to 700 units after the start line. */
    private static Track dragStrip() {
        double[][] p = new double[16][];
        for (int i = 0; i < 8; i++) {
            p[i] = new double[] {0.1 + 0.8 * i / 7.0, 0.4};
            p[15 - i] = new double[] {0.1 + 0.8 * i / 7.0, 0.6};
        }
        return Track.of("drag strip", p, 40);
    }

    @Test
    void aCarRightBehindAnotherGetsASlipstreamAndCatchesUp() {
        Race race = new Race(open(2), dragStrip());
        race.reset(new SplittableRandom(1));
        race.place(0, 100, 0, 200);
        race.place(1, 40, 0, 200);
        drive(race, 0, 1);
        assertTrue(race.draft(1) > 0.2, "the car behind drafts: " + race.draft(1));
        assertEquals(0, race.draft(0), "the leader does not");
        double gap0 = race.view().cars().get(0).distance() - race.view().cars().get(1).distance();
        for (int t = 0; t < 25; t++) {
            drive(race, 0, 1);
        }
        double gap1 = race.view().cars().get(0).distance() - race.view().cars().get(1).distance();
        assertTrue(gap1 < gap0 - 3, "the gap closes: " + gap0 + " -> " + gap1);

        Race side = new Race(open(2), dragStrip());
        side.reset(new SplittableRandom(1));
        side.place(0, 100, -25, 200);
        side.place(1, 40, 0, 200);
        drive(side, 0, 1);
        assertEquals(0, side.draft(1), "no slipstream beside the other car's path");

        Race noDraft = new Race(open(2).draftStrength(0), dragStrip());
        noDraft.reset(new SplittableRandom(1));
        noDraft.place(0, 100, 0, 200);
        noDraft.place(1, 40, 0, 200);
        drive(noDraft, 0, 1);
        assertEquals(0, noDraft.draft(1));
    }

    @Test
    void carsBounceOffEachOtherAndArePenalised() {
        RaceConfig c = open(2).collisionPenalty(0.05).overtakeBonus(0);
        Race race = new Race(c);
        race.reset(new SplittableRandom(1));
        race.place(0, 40, 0, 0);
        race.place(1, 0, 0, 200);
        boolean hit = false;
        for (int t = 0; t < 15 && !hit; t++) {
            Environment.Step[] s = race.step(new float[][] {{0, 0}, {0, 1}});
            hit = race.view().cars().get(1).collisions() > 0;
            if (hit) {
                assertTrue(s[0].reward() < 0 && s[1].reward() < s[1].reward() + 1, "both are penalised");
                assertTrue(s[0].reward() < -0.5, "the car that was hit is penalised: " + s[0].reward());
            }
        }
        assertTrue(hit, "the cars collide");
        assertTrue(race.view().cars().get(0).speed() > 50, "the hit car is pushed forwards");
        double gap = race.view().cars().get(0).distance() - race.view().cars().get(1).distance();
        assertTrue(gap > c.carLength() - 1, "the cars do not overlap, gap " + gap);

        Race ghosts = new Race(open(2).collisions(false));
        ghosts.reset(new SplittableRandom(1));
        ghosts.place(0, 40, 0, 0);
        ghosts.place(1, 0, 0, 200);
        for (int t = 0; t < 15; t++) {
            ghosts.step(new float[][] {{0, 0}, {0, 1}});
        }
        assertEquals(0, ghosts.view().cars().get(1).collisions());
        assertTrue(ghosts.view().cars().get(1).distance() > ghosts.view().cars().get(0).distance(), "drove through");
    }

    @Test
    void passingACarIsRewarded() {
        RaceConfig c = new RaceConfig().cars(2).track("serpent").randomStart(false).shuffleGrid(false)
                .overtakeBonus(1).collisionPenalty(0).progressReward(0);
        Race race = new Race(c);
        race.reset(new SplittableRandom(1));
        race.place(0, 30, -20, 100);
        race.place(1, 0, 20, 300);
        double bonus = 0;
        double penalty = 0;
        for (int t = 0; t < 8; t++) {
            Environment.Step[] s = race.step(new float[][] {{0, 0}, {0, 1}});
            bonus += s[1].reward();
            penalty += s[0].reward();
        }
        assertEquals(1, bonus, 1e-6);
        assertEquals(-1, penalty, 1e-6);
        assertEquals(1, race.view().cars().get(1).overtakes());
    }

    @Test
    void observationsShowTheNearestCarsInTheCarsOwnFrame() {
        RaceConfig c = open(3).nearestCars(2);
        Race race = new Race(c, dragStrip());
        race.reset(new SplittableRandom(1));
        race.place(0, 150, 0, 100);
        race.place(1, 210, 0, 150);
        race.place(2, 350, 0, 100);
        float[] o = race.observe(0);
        assertEquals(c.observationSize(), o.length);
        assertEquals(Race.OWN + c.rays() + c.probes().length + 2 * Race.PER_CAR, o.length);
        int k = Race.OWN + c.rays() + c.probes().length;
        assertEquals(1, o[k], "the nearest car is seen");
        assertEquals(60 / c.senseRadius(), o[k + 1], 0.02, "60 ahead");
        assertEquals(0, o[k + 2], 0.02, "straight ahead");
        assertEquals(50 / c.speedScale(), o[k + 3], 0.02, "pulling away at 50");
        assertEquals(1, o[k + 6], 0.01, "same heading");
        assertEquals(1, o[k + 8], "the second nearest is seen too");
        assertEquals(200 / c.senseRadius(), o[k + 9], 0.05);
        assertEquals(100 / c.speedScale(), o[Race.OWN - 8], 0.01, "own forward speed");

        Race blind = new Race(open(3).nearestCars(0).rays(3).probes(100));
        assertEquals(Race.OWN + 3 + 1, blind.observationSpace().size());
        assertEquals(Race.OWN + 3 + 1, blind.reset(new SplittableRandom(1))[0].length);
    }

    @Test
    void leavingTheRoadCrashesAndTheCarRespawnsAsAGhost() {
        Race race = new Race(new RaceConfig().cars(2).track("oval").randomStart(false));
        race.reset(new SplittableRandom(1));
        boolean crashed = false;
        for (int t = 0; t < 200 && !crashed; t++) {
            Environment.Step[] s = race.step(new float[][] {{-1, 1}, {0, 0}});
            if (s[0].terminated()) {
                crashed = true;
                assertTrue(s[0].reward() <= -race.config().crashPenalty() + 0.5, "penalised");
                assertFalse(s[1].done(), "the other car carries on");
            }
        }
        assertTrue(crashed, "steering off the road crashes");
        assertTrue(race.view().cars().getFirst().crashed());
        race.reset(0, new SplittableRandom(2));
        RaceView.CarView car = race.view().cars().getFirst();
        assertFalse(car.crashed());
        assertTrue(car.ghost(), "respawned as a ghost");
        assertEquals(1, car.crashes());
    }

    @Test
    void aRaceEndsForEveryCarAndTheNextOneStartsOnTheGrid() {
        Race race = new Race(new RaceConfig().cars(4).raceSteps(10));
        race.reset(new SplittableRandom(1));
        assertEquals(1, race.races());
        Environment.Step[] s = null;
        for (int t = 0; t < 10; t++) {
            s = drive(race, 0, 0.5f);
        }
        for (Environment.Step step : s) {
            assertTrue(step.truncated() && !step.terminated());
        }
        for (int i = 0; i < 4; i++) {
            race.reset(i, new SplittableRandom(i));
        }
        assertEquals(2, race.races(), "one new race, not one per car");
        assertEquals(0, race.raceStep());
        for (RaceView.CarView car : race.view().cars()) {
            assertEquals(0, car.speed(), 1e-9);
        }
    }

    @Test
    void opponentsCanBeDrivenByTheBot() {
        Race race = new Race(new RaceConfig().cars(5));
        VectorEnvironment env = race.withOpponents(2, Driver.bot());
        assertEquals(2, env.count());
        assertEquals(2, env.reset(new SplittableRandom(1)).length);
        for (int t = 0; t < 60; t++) {
            assertEquals(2, env.step(new float[][] {{0, 0}, {0, 0}}).length);
        }
        assertTrue(race.view().cars().get(4).distance() > race.view().cars().get(0).distance() + 100,
                "the bots drive off");
    }

    @Test
    void theBotLapsWithoutCrashing() {
        for (String track : Track.names()) {
            Race race = new Race(new RaceConfig().cars(1).track(track).randomStart(false));
            race.reset(new SplittableRandom(1));
            int crashes = 0;
            for (int t = 0; t < 900; t++) {
                Environment.Step[] s = race.step(new float[][] {race.botAction(0, 0)});
                if (s[0].terminated()) {
                    crashes++;
                    race.reset(0, new SplittableRandom(t));
                }
            }
            assertTrue(crashes <= 1, track + ": " + crashes + " crashes");
            assertTrue(race.view().cars().getFirst().distance() > 150 * 30, track + ": the bot averages over 150");
        }
    }

    @Test
    void theRaceCanBeDrawnOffScreen() {
        Race race = new Race(new RaceConfig().cars(6));
        race.reset(new SplittableRandom(1));
        for (int t = 0; t < 30; t++) {
            drive(race, 0, 1);
        }
        BufferedImage img = new RaceRenderer(race).image(640, 400);
        assertEquals(640, img.getWidth());
    }
}
