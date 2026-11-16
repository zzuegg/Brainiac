package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.race.Driver;
import io.github.zzuegg.brainiac.race.Race;
import io.github.zzuegg.brainiac.race.RaceConfig;
import io.github.zzuegg.brainiac.race.RaceRenderer;
import io.github.zzuegg.brainiac.race.RaceView;
import io.github.zzuegg.brainiac.race.RaceWindow;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.VectorEnvironment;

import java.awt.GraphicsEnvironment;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;

/**
 * Cars learn to race each other with PPO: every car on the track is driven by the same
 * policy, and all their experience trains it (self-play). The physics has front and rear
 * axle grip (so cars slide and drift), a slipstream that makes passing pay off, and
 * car-to-car collisions. Each car sees its own state, the walls, the road ahead and its
 * nearest rivals with their relative positions and velocities.
 *
 * <pre>
 * race train [--cars 8] [--track kidney] [--steps 3000000] [--save race.model]
 *            [--opponents self|bots] [--no-window] [--set name=value ...]
 * race watch [--model race.model | --bots] [--cars 8] [--track kidney] [--set name=value ...]
 * race picture [--model race.model | --bots] [--frames 300] [--out race.png]
 * </pre>
 *
 * <p>{@code --set} changes any {@link RaceConfig} setting, for example
 * {@code --set rearGrip=500 --set draftStrength=0.6 --set nearestCars=4}. Use the same
 * settings that change the observation ({@code rays}, {@code probes},
 * {@code nearestCars}) for training and watching. {@code watch} without a model shows the
 * hand-written bots.
 */
public final class RaceExample {

    private RaceExample() {
    }

    /**
     * @param args the command and options
     */
    public static void main(String[] args) {
        String command = args.length > 0 ? args[0] : "train";
        Options o = Options.parse(args.length > 0 ? java.util.Arrays.copyOfRange(args, 1, args.length) : args);
        try {
            switch (command) {
                case "train" -> train(o);
                case "watch" -> watch(o);
                case "picture" -> picture(o);
                default -> System.out.println("Usage: race train|watch|picture [options]; see RaceExample");
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void train(Options o) throws IOException {
        RaceConfig config = o.config();
        Race race = new Race(config);
        int obs = config.observationSize();
        System.out.println("Training on " + race);
        Ppo ppo = Ppo.of(Nets.mlp(obs, 2, 1, 64, 64), Nets.mlp(obs, 1, 2, 64, 64), race.actionSpace())
                .learningRate(3e-4f).rolloutSteps(256).minibatchSize(256).epochs(5).initialLogStd(-0.5f)
                .seed(o.seed);
        VectorEnvironment env = o.opponents.equals("bots") ? race.withOpponents(1, Driver.bot()) : race;
        RaceWindow window = o.window && !GraphicsEnvironment.isHeadless()
                ? RaceWindow.open(race, "Brainiac racing: training", false) : null;
        if (window != null) {
            System.out.println("Window open: r switches between real time and full speed, space pauses.");
        }
        Path save = Path.of(o.model);
        long start = System.nanoTime();
        long chunk = Math.max(20_000, o.steps / 20);
        long done = 0;
        System.out.printf("%9s %7s %13s %13s %8s %10s %11s%n", "steps", "time s", "train return", "greedy dist*",
                "crashes", "overtakes", "collisions");
        while (done < o.steps && (window == null || !window.closed())) {
            long n = Math.min(chunk, o.steps - done);
            RlResult r = ppo.fit(env, n, RlListener.NONE);
            done += n;
            ppo.policy().save(save);
            Score s = score(policyAgent(ppo.policy(), race.actionSpace()), o, 7);
            String line = String.format(Locale.ROOT, "%9d %7.0f %13.1f %13.0f %8d %10d %11d", done,
                    (System.nanoTime() - start) / 1e9, r.meanReturn(100), s.distance, s.crashes, s.overtakes,
                    s.collisions);
            System.out.println(line);
            if (window != null) {
                window.status(String.format(Locale.ROOT, "steps %,d   greedy %.0f per car", done, s.distance));
            }
        }
        Score bots = score((ob, explore) -> null, o, 7);
        System.out.printf("hand-written bots for comparison: %.0f per car, %d crashes, %d overtakes, %d collisions%n",
                bots.distance, bots.crashes, bots.overtakes, bots.collisions);
        System.out.println("* greedy dist: average distance per car in one race with every car driven by the policy");
        System.out.println("Saved the policy to " + save.toAbsolutePath());
        if (window != null && !window.closed()) {
            System.out.println("Now showing the trained policy in real time; close the window to stop.");
            window.realTime(true);
            window.status("trained policy, " + String.format(Locale.ROOT, "%,d", done) + " steps");
            window.watch(Driver.of(policyAgent(ppo.policy(), race.actionSpace())), o.seed);
        }
    }

    private static void watch(Options o) throws IOException {
        RaceConfig config = o.config();
        Race race = new Race(config);
        Driver driver = driverFor(o, race);
        RaceWindow.open(race, "Brainiac racing", true).watch(driver, o.seed);
    }

    private static void picture(Options o) throws IOException {
        RaceConfig config = o.config();
        Race race = new Race(config);
        Driver driver = driverFor(o, race);
        SplittableRandom random = new SplittableRandom(o.seed);
        race.reset(random.split());
        for (int t = 0; t < o.frames; t++) {
            float[][] a = new float[race.count()][];
            for (int i = 0; i < a.length; i++) {
                a[i] = driver.drive(race, i);
            }
            var steps = race.step(a);
            for (int i = 0; i < steps.length; i++) {
                if (steps[i].done()) {
                    race.reset(i, random.split());
                }
            }
        }
        RaceRenderer r = new RaceRenderer(race);
        javax.imageio.ImageIO.write(r.image(1280, 760), "png", Path.of(o.out).toFile());
        System.out.println("Wrote " + Path.of(o.out).toAbsolutePath());
    }

    private static Driver driverFor(Options o, Race race) throws IOException {
        Path model = Path.of(o.model);
        if (o.bots) {
            return Driver.bot();
        }
        if (o.modelGiven || Files.exists(model)) {
            System.out.println("Driving every car with the policy in " + model.toAbsolutePath());
            return Driver.of(policyAgent(Model.load(model), race.actionSpace()));
        }
        System.out.println("No policy file " + model + " yet, so the hand-written bots drive. Train with: race train");
        return Driver.bot();
    }

    /** The policy's mean action, clipped into the action box: how a trained PPO policy drives greedily. */
    private static Agent policyAgent(Model policy, Space actions) {
        Space.Box box = (Space.Box) actions;
        return (observation, explore) -> box.clip(policy.predict(observation));
    }

    private record Score(double distance, int crashes, int overtakes, int collisions) {
    }

    /** One race with every car driven by the agent (or the bots when it returns null). */
    private static Score score(Agent agent, Options o, long seed) {
        Race race = new Race(o.config().randomStart(false));
        SplittableRandom random = new SplittableRandom(seed);
        float[][] obs = race.reset(random.split());
        Driver bot = Driver.bot();
        int crashes = 0;
        for (int t = 0; t < race.config().raceSteps(); t++) {
            float[][] a = new float[race.count()][];
            for (int i = 0; i < a.length; i++) {
                float[] act = agent.act(obs[i], false);
                a[i] = act == null ? bot.drive(race, i) : act;
            }
            var steps = race.step(a);
            for (int i = 0; i < steps.length; i++) {
                obs[i] = steps[i].observation();
                if (steps[i].terminated()) {
                    crashes++;
                    obs[i] = race.reset(i, random.split());
                }
            }
        }
        RaceView view = race.view();
        double dist = view.cars().stream().mapToDouble(RaceView.CarView::distance).average().orElse(0);
        int overtakes = view.cars().stream().mapToInt(RaceView.CarView::overtakes).sum();
        int collisions = view.cars().stream().mapToInt(RaceView.CarView::collisions).sum() / 2;
        return new Score(dist, crashes, overtakes, collisions);
    }

    private static final class Options {
        int cars = 8;
        String track = "kidney";
        long steps = 3_000_000;
        String model = "race.model";
        boolean modelGiven;
        boolean bots;
        String opponents = "self";
        boolean window = true;
        long seed = 1;
        int frames = 300;
        String out = "race.png";
        final List<String> sets = new ArrayList<>();

        static Options parse(String[] args) {
            Options o = new Options();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                switch (a) {
                    case "--cars" -> o.cars = Integer.parseInt(args[++i]);
                    case "--track" -> o.track = args[++i];
                    case "--steps" -> o.steps = Long.parseLong(args[++i].replace("_", ""));
                    case "--save", "--model" -> {
                        o.model = args[++i];
                        o.modelGiven = true;
                    }
                    case "--opponents" -> o.opponents = args[++i];
                    case "--no-window" -> o.window = false;
                    case "--bots" -> o.bots = true;
                    case "--seed" -> o.seed = Long.parseLong(args[++i]);
                    case "--frames" -> o.frames = Integer.parseInt(args[++i]);
                    case "--out" -> o.out = args[++i];
                    case "--set" -> o.sets.add(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + a);
                }
            }
            return o;
        }

        RaceConfig config() {
            RaceConfig c = new RaceConfig().cars(cars).track(track);
            for (String s : sets) {
                String[] kv = s.split("=", 2);
                if (kv.length != 2) {
                    throw new IllegalArgumentException("--set needs name=value, got " + s);
                }
                set(c, kv[0].trim(), kv[1].trim());
            }
            return c;
        }

        private static void set(RaceConfig c, String name, String value) {
            for (Method m : RaceConfig.class.getMethods()) {
                if (!m.getName().equals(name) || m.getParameterCount() != 1 || m.getReturnType() != RaceConfig.class) {
                    continue;
                }
                Class<?> t = m.getParameterTypes()[0];
                try {
                    Object v;
                    if (t == int.class) {
                        v = Integer.parseInt(value);
                    } else if (t == double.class) {
                        v = Double.parseDouble(value);
                    } else if (t == boolean.class) {
                        v = Boolean.parseBoolean(value);
                    } else if (t == String.class) {
                        v = value;
                    } else if (t == double[].class) {
                        v = java.util.Arrays.stream(value.split(",")).mapToDouble(Double::parseDouble).toArray();
                    } else {
                        continue;
                    }
                    m.invoke(c, v);
                    return;
                } catch (ReflectiveOperationException e) {
                    Throwable cause = e.getCause() != null ? e.getCause() : e;
                    throw new IllegalArgumentException("cannot set " + name + "=" + value + ": " + cause.getMessage(),
                            cause);
                }
            }
            throw new IllegalArgumentException("no race setting called " + name);
        }
    }
}
