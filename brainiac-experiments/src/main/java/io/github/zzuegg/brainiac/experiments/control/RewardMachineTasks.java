package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.LabelledGrid;
import io.github.zzuegg.brainiac.experimental.rl.RewardMachine;
import io.github.zzuegg.brainiac.experimental.rl.RewardMachineLearner;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.15: reward machines in an office world: "get coffee ({@code c}), then the mail
 * ({@code m}), then go to the office ({@code o})". Learners at several step budgets: Q-learning
 * with counterfactual experience over the machine states (CRM), the same without
 * counterfactuals (Q-learning on the product states), and the core tabular Q-learner given the
 * machine's reward but not its state (history-free). Metric: share of seeds whose greedy policy
 * completes the task.
 *
 * <p>Paper: Toro Icarte et al., <a href="https://arxiv.org/abs/2010.03950">arXiv:2010.03950</a>.
 */
public final class RewardMachineTasks {

    /** The office world: coffee, mail room and office. */
    public static final String[] OFFICE = {
        "c.....#..",
        "...#..#..",
        "...#.....",
        "S..#..m..",
        "......#..",
        "####.##..",
        "o........",
    };

    private RewardMachineTasks() {
    }

    /**
     * @param budgets the step budgets
     * @param seeds seeds per budget
     * @return "method, budget" to greedy success rate
     */
    public static Map<String, Double> run(int[] budgets, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        RewardMachine task = RewardMachine.sequence(1, "cmo");
        String[] names = {"CRM", "product Q-learning", "history-free Q-learning (core)"};
        for (int m = 0; m < names.length; m++) {
            for (int budget : budgets) {
                double ok = 0;
                for (int s = 0; s < seeds; s++) {
                    long seed = 1 + s;
                    RewardMachine.Product exposed = task.environment(new LabelledGrid(OFFICE).maxSteps(100), true);
                    RewardMachine.Product hidden = task.environment(new LabelledGrid(OFFICE).maxSteps(100), false);
                    Learner l = switch (m) {
                        case 0 -> RewardMachineLearner.of(task, 4).seed(seed);
                        case 1 -> RewardMachineLearner.of(task, 4).counterfactual(false).seed(seed);
                        default -> Tabular.qLearning(4).learningRate(0.5).discount(0.95)
                                .epsilon(Schedule.constant(0.1)).seed(seed);
                    };
                    RewardMachine.Product env = m == 2 ? hidden : exposed;
                    l.fit(env, budget);
                    ok += completes(l, task.environment(new LabelledGrid(OFFICE).maxSteps(100), m != 2), seed) ? 1 : 0;
                }
                out.put(names[m] + ", " + budget, ok / seeds);
            }
        }
        return out;
    }

    private static boolean completes(Agent agent, RewardMachine.Product env, long seed) {
        float[] o = env.reset(new SplittableRandom(seed));
        Environment.Step s;
        do {
            s = env.step(agent.act(o, false));
            o = s.observation();
        } while (!s.done());
        return env.completions() > 0;
    }
}
