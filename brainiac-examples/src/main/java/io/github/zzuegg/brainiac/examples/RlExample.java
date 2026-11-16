package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Lander;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Reinforce;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

/**
 * Reinforcement learning, from a table to deep networks.
 *
 * <ol>
 *   <li>The cliff walk: tabular Q-learning learns the shortest path right along the
 *       cliff edge, SARSA a safer one, because it accounts for its own random moves.</li>
 *   <li>Cart-pole: REINFORCE, A2C, PPO and DQN learn to balance a pole, compared with
 *       acting at random (500 is the most an episode can score).</li>
 *   <li>Pendulum: PPO with a Gaussian policy learns continuous torques to swing a
 *       pendulum up and hold it.</li>
 *   <li>Lander: PPO learns to fire a main engine and two thrusters to land on a pad, a
 *       harder task where an average of 200 counts as solved.</li>
 * </ol>
 *
 * <p>{@code ./gradlew :brainiac-examples:run --args=rl} (several minutes on a laptop),
 * {@code --args="rl quick"} for shorter training, or {@code --args="rl lander"} for the
 * lander alone.
 */
public final class RlExample {

    private RlExample() {
    }

    /**
     * Runs the three parts.
     *
     * @param args {@code quick} for a shorter run
     */
    public static void main(String[] args) {
        if (args.length > 0 && args[0].equals("lander")) {
            lander(args.length > 1 ? Double.parseDouble(args[1]) : 1);
            return;
        }
        double scale = args.length > 0 && args[0].equals("quick") ? 0.4 : 1;
        cliff();
        cartPole(scale);
        pendulum(scale);
        lander(scale);
    }

    private static void cliff() {
        System.out.println("== Cliff walk (S start, G goal, C cliff: -100) ==");
        for (Tabular t : List.of(Tabular.qLearning(4).learningRate(0.5).seed(1),
                Tabular.sarsa(4).learningRate(0.5).seed(1))) {
            t.fit(GridWorld.cliff(), 20_000);
            System.out.printf("%s, greedy return %.0f:%n%s%n", t, Rl.evaluate(t, GridWorld.cliff(), 1, 1),
                    GridWorld.cliff().renderPolicy(t));
        }
    }

    private record Contender(String name, Supplier<Learner> make, long steps, int envs) {
    }

    private static void cartPole(double scale) {
        System.out.println("== Cart-pole (balance for up to 500 steps) ==");
        Space actions = new CartPole().actionSpace();
        System.out.printf("%-10s %9s %8s %10s%n", "agent", "steps", "train s", "greedy");
        System.out.printf("%-10s %9s %8s %10.1f%n", "random", "-", "-",
                Rl.evaluate(Agent.random(actions, 1), new CartPole(), 20, 7));
        List<Contender> contenders = List.of(
                new Contender("REINFORCE", () -> Reinforce.of(Nets.mlp(4, 2, 1, 32), actions)
                        .baseline(Nets.mlp(4, 1, 2, 32)).entropyBonus(0.001f).clipGradientNorm(5).seed(1), 60_000, 1),
                new Contender("A2C", () -> Ppo.a2c(Nets.mlp(4, 2, 1, 64), Nets.mlp(4, 1, 2, 64), actions).seed(1),
                        200_000, 8),
                new Contender("PPO", () -> Ppo.of(Nets.mlp(4, 2, 1, 64, 64), Nets.mlp(4, 1, 2, 64, 64), actions)
                        .learningRate(1e-3f).targetKl(0.05f).seed(1), 60_000, 4),
                new Contender("DQN", () -> Dqn.of(() -> Nets.mlp(4, 2, Activation.RELU, 1, 64, 64), 2)
                        .nSteps(3).targetUpdateEvery(1_000).seed(1), 50_000, 1));
        Learner best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Contender c : contenders) {
            Learner learner = c.make().get();
            long steps = Math.round(c.steps() * scale);
            long start = System.nanoTime();
            RlResult result = learner instanceof Ppo ppo && c.envs() > 1
                    ? ppo.fit(CartPole::new, c.envs(), steps, RlListener.NONE)
                    : learner.fit(new CartPole(), steps);
            double seconds = (System.nanoTime() - start) / 1e9;
            double score = Rl.evaluate(learner, new CartPole(), 10, 7);
            System.out.printf("%-10s %9d %8.1f %10.1f   (%d training episodes)%n", c.name(), result.totalSteps(),
                    seconds, score, result.episodes().size());
            if (score > bestScore) {
                bestScore = score;
                best = learner;
            }
        }
        System.out.println("\nThe best agent, every 25th step of one episode:");
        List<String> frames = Rl.watch(best, new CartPole(), 3, 500);
        for (int i = 0; i < frames.size(); i += 25) {
            System.out.println(frames.get(i));
        }
        System.out.println();
    }

    private static void pendulum(double scale) {
        System.out.println("== Pendulum (continuous torque; 0 is perfect, random is about -1200) ==");
        Pendulum env = new Pendulum();
        Ppo ppo = Ppo.of(Nets.mlp(3, 1, 1, 64, 64), Nets.mlp(3, 1, 2, 64, 64), env.actionSpace())
                .learningRate(1e-3f).discount(0.9f).rolloutSteps(512).seed(1);
        System.out.printf("before training: %.0f%n", Rl.evaluate(ppo, env, 5, 3));
        long start = System.nanoTime();
        ppo.fit(Pendulum::new, 4, Math.round(150_000 * scale), RlListener.NONE);
        System.out.printf("after %.0f s of PPO: %.0f%n", (System.nanoTime() - start) / 1e9, Rl.evaluate(ppo, env, 5, 3));
        List<String> frames = Rl.watch(ppo, env, 3, 200);
        System.out.println(frames.get(0));
        System.out.println(frames.get(100));
        System.out.println(frames.get(frames.size() - 1));
    }

    /**
     * Trains PPO on the lander and shows a landing.
     *
     * @param scale the share of the full 1,000,000 training steps
     */
    public static void lander(double scale) {
        System.out.println("== Lander (200 is solved; the classic lunar lander's PPO baselines reach 200-250) ==");
        Lander env = new Lander();
        System.out.printf(Locale.ROOT, "random: %.0f%n", Rl.evaluate(Agent.random(env.actionSpace(), 1), env, 20, 7));
        Ppo ppo = Ppo.of(Nets.mlp(8, 4, 1, 64, 64), Nets.mlp(8, 1, 2, 64, 64), env.actionSpace())
                .rolloutSteps(1024).minibatchSize(256).epochs(4).discount(0.999f).gaeLambda(0.98f)
                .entropyBonus(0.01f).learningRate(3e-4f).seed(1);
        long start = System.nanoTime();
        long steps = Math.round(1_000_000 * scale);
        long[] next = {steps / 10};
        ppo.fit(Lander::new, 8, steps, (episode, result) -> {
            if (episode.totalSteps() >= next[0]) {
                next[0] += steps / 10;
                System.out.printf(Locale.ROOT, "  %,9d steps  %5.0f s  last 50 episodes %6.0f%n",
                        episode.totalSteps(), (System.nanoTime() - start) / 1e9, result.meanReturn(50));
            }
        });
        System.out.printf(Locale.ROOT, "after %.0f s of PPO: %.0f over 100 episodes%n",
                (System.nanoTime() - start) / 1e9, Rl.evaluate(ppo, new Lander(), 100, 7));
        List<String> frames = Rl.watch(ppo, new Lander(), 3, 1000);
        for (int i = 0; i < frames.size(); i += Math.max(1, frames.size() / 12)) {
            System.out.println(frames.get(i));
        }
        System.out.println(frames.get(frames.size() - 1));
    }
}
