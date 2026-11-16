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
 *            [--opponents self|bots|past] [--discount 0.99] [--hidden 64,64] [--round steps]
 *            [--lr 3e-4] [--anneal] [--target-kl K] [--adaptive-lr K] [--plateau rounds] [--no-window] [--set name=value ...]
 * race watch [--model race.model | --bots] [--cars 8] [--track kidney] [--set name=value ...]
 * race picture [--model race.model | --bots] [--frames 300] [--out race.png]
 * </pre>
 *
 * <p>{@code --opponents}: {@code self} (default) drives every car with the learning
 * policy; {@code bots} and {@code past} let it drive half the cars against the
 * hand-written bots or against the best policy saved so far.
 *
 * <p>Training keeps the best policy so far in {@code race.model}: after every round
 * the new policy races the best one (the bots at first), each driving half the cars, and
 * replaces it if it finishes ahead on average. The latest policy is in
 * {@code race.model.last}.
 *
 * <p>Learning rate: {@code --lr} sets it, {@code --anneal} lowers it linearly to 0 over
 * the run, {@code --adaptive-lr K} lowers or raises it when an update moves the policy
 * much more or less than KL divergence K, {@code --target-kl K} cuts an update short
 * when it moves too far, and {@code --plateau N} halves it after N rounds without a new
 * best policy.
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
        int act = config.actionSize();
        System.out.println("Networks: " + obs + " inputs, hidden " + java.util.Arrays.toString(o.hidden) + ", "
                + act + " outputs (steer, pedal, " + (act - 2) + " sensor distances)");
        io.github.zzuegg.brainiac.train.Optimizer optimizer = io.github.zzuegg.brainiac.train.Optimizer.adam(o.lr);
        Ppo ppo = Ppo.of(Nets.mlp(obs, act, 1, o.hidden), Nets.mlp(obs, 1, 2, o.hidden), race.actionSpace())
                .optimizer(optimizer).targetKl(o.targetKl).adaptiveLearningRate(o.adaptiveKl).rolloutSteps(256).minibatchSize(256).epochs(5).initialLogStd(-0.5f)
                .discount(o.discount).seed(o.seed);
        // opponents: copies of the policy itself (self), the hand-written bots, or the best
        // policy saved so far (past), which then drives every other car
        java.util.concurrent.atomic.AtomicReference<Driver> past =
                new java.util.concurrent.atomic.AtomicReference<>(Driver.bot());
        VectorEnvironment env = switch (o.opponents) {
            case "self" -> race;
            case "bots" -> race.withOpponents(Math.max(1, config.cars() / 2), Driver.bot());
            case "past" -> race.withOpponents(Math.max(1, config.cars() / 2), (r, i) -> past.get().drive(r, i));
            default -> throw new IllegalArgumentException("--opponents must be self, bots or past");
        };
        RaceWindow window = o.window && !GraphicsEnvironment.isHeadless()
                ? RaceWindow.open(race, "Brainiac racing: training", false) : null;
        if (window != null) {
            System.out.println("Window open: r switches between real time and full speed, space pauses.");
        }
        Path save = Path.of(o.model);
        Path last = Path.of(o.model + ".last");
        Driver champion = Driver.bot();
        int champions = 0;
        long start = System.nanoTime();
        long chunk = o.round > 0 ? o.round : Math.max(20_000, o.steps / 20);
        long done = 0;
        double plateau = 1;
        int sinceBest = 0;
        System.out.printf(Locale.ROOT, "learning rate %.1e%s%s%s%s%n", o.lr, o.anneal ? ", decaying linearly to 0" : "",
                o.adaptiveKl > 0 ? ", adapting to KL " + o.adaptiveKl : "",
                o.targetKl > 0 ? ", epochs stop above KL " + o.targetKl : "",
                o.plateau > 0 ? ", halved after " + o.plateau + " rounds without a new best" : "");
        System.out.printf("%9s %7s %13s %13s %13s %8s %10s %11s%n", "steps", "time s", "train return",
                "place vs best", "greedy dist*", "crashes", "overtakes", "collisions");
        while (done < o.steps && (window == null || !window.closed())) {
            long n = Math.min(chunk, o.steps - done);
            // the base rate for this round: linear decay over the whole run, times the
            // plateau factor; PPO's KL adaptation scales it further within the round
            double decay = o.anneal ? 1 - (double) done / o.steps : 1;
            optimizer.learningRate((float) Math.max(1e-8, o.lr * decay * plateau));
            RlResult r = ppo.fit(env, n, RlListener.NONE);
            done += n;
            ppo.policy().save(last);
            Agent current = policyAgent(ppo.policy(), race.actionSpace());
            Score s = score(current, o, 7);
            // the best policy is the one that wins: race the new one against the best so far
            // (the bots at first), half the cars each, and keep it if it finishes ahead
            double place = headToHead(Driver.of(current), champion, o, 3);
            boolean better = place < (config.cars() + 1) / 2.0;
            if (better) {
                ppo.policy().save(save);
                champion = Driver.of(policyAgent(Model.load(save), race.actionSpace()));
                past.set(champion);
                champions++;
                sinceBest = 0;
            } else if (o.plateau > 0 && ++sinceBest >= o.plateau) {
                plateau /= 2;
                sinceBest = 0;
            }
            double rate = ppo.diagnostics().getOrDefault("learningRate", (double) o.lr);
            double kl = ppo.diagnostics().getOrDefault("approxKl", 0.0);
            String line = String.format(Locale.ROOT, "%9d %7.0f %13.1f %13.2f %13.0f %8d %10d %11d%s", done,
                    (System.nanoTime() - start) / 1e9, r.meanReturn(100), place, s.distance, s.crashes,
                    s.overtakes, s.collisions, better ? "  beats the best, saved (#" + champions + ")" : "")
                    + String.format(Locale.ROOT, "  lr %.1e kl %.4f", rate, kl);
            System.out.println(line);
            if (window != null) {
                window.status(String.format(Locale.ROOT, "steps %,d   greedy %.0f per car", done, s.distance));
            }
        }
        Score bots = score((ob, explore) -> null, o, 7);
        System.out.printf("hand-written bots for comparison: %.0f per car, %d crashes, %d overtakes, %d collisions%n",
                bots.distance, bots.crashes, bots.overtakes, bots.collisions);
        System.out.println("place vs best: average finishing place of the new policy in 3 races against the best");
        System.out.println("so far (the bots at first), each driving half the cars; below "
                + String.format(Locale.ROOT, "%.1f", (config.cars() + 1) / 2.0) + " means it wins and becomes the best");
        System.out.println("* greedy dist: average distance per car in one race with every car driven by the policy");
        System.out.println("Saved the best policy to " + save.toAbsolutePath() + " and the last one to " + last);
        if (window != null && !window.closed()) {
            System.out.println("Now showing the trained policy in real time; close the window to stop.");
            window.realTime(true);
            window.status("best trained policy, " + String.format(Locale.ROOT, "%,d", done) + " steps");
            window.watch(Driver.of(policyAgent(Model.load(save), race.actionSpace())), o.seed);
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

    /**
     * Races two drivers against each other, each driving every other car, and returns the
     * first driver's average finishing place (1 is first).
     */
    private static double headToHead(Driver a, Driver b, Options o, int races) {
        Race race = new Race(o.config());
        SplittableRandom random = new SplittableRandom(99);
        double sum = 0;
        int n = 0;
        for (int k = 0; k < races; k++) {
            race.reset(random.split());
            int shift = k % 2;
            for (int t = 0; t < race.config().raceSteps(); t++) {
                float[][] act = new float[race.count()][];
                for (int i = 0; i < act.length; i++) {
                    act[i] = (i + shift) % 2 == 0 ? a.drive(race, i) : b.drive(race, i);
                }
                var steps = race.step(act);
                for (int i = 0; i < steps.length; i++) {
                    if (steps[i].terminated()) {
                        race.reset(i, random.split());
                    }
                }
            }
            for (RaceView.CarView car : race.view().cars()) {
                if ((car.id() + shift) % 2 == 0) {
                    sum += car.position();
                    n++;
                }
            }
        }
        return sum / n;
    }

    private record Score(double reward, double distance, int crashes, int overtakes, int collisions) {
    }

    /** One race with every car driven by the agent (or the bots when it returns null). */
    private static Score score(Agent agent, Options o, long seed) {
        Race race = new Race(o.config().randomStart(false));
        SplittableRandom random = new SplittableRandom(seed);
        float[][] obs = race.reset(random.split());
        Driver bot = Driver.bot();
        int crashes = 0;
        double reward = 0;
        for (int t = 0; t < race.config().raceSteps(); t++) {
            float[][] a = new float[race.count()][];
            for (int i = 0; i < a.length; i++) {
                float[] act = agent.act(obs[i], false);
                a[i] = act == null ? bot.drive(race, i) : act;
            }
            var steps = race.step(a);
            for (int i = 0; i < steps.length; i++) {
                obs[i] = steps[i].observation();
                reward += steps[i].reward();
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
        return new Score(reward / race.count(), dist, crashes, overtakes, collisions);
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
        float discount = 0.99f;
        int[] hidden = {64, 64};
        float lr = 3e-4f;
        boolean anneal;
        float targetKl;
        float adaptiveKl;
        int plateau;
        long round;
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
                    case "--discount" -> o.discount = Float.parseFloat(args[++i]);
                    case "--round" -> o.round = Long.parseLong(args[++i].replace("_", ""));
                    case "--lr" -> o.lr = Float.parseFloat(args[++i]);
                    case "--anneal" -> o.anneal = true;
                    case "--target-kl" -> o.targetKl = Float.parseFloat(args[++i]);
                    case "--adaptive-lr" -> o.adaptiveKl = Float.parseFloat(args[++i]);
                    case "--plateau" -> o.plateau = Integer.parseInt(args[++i]);
                    case "--hidden" -> o.hidden = java.util.Arrays.stream(args[++i].split(","))
                            .mapToInt(Integer::parseInt).toArray();
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
