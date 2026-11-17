package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.LatentActionWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.MlpWorldModel;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Genie at toy scale: a {@link LatentActionWorldModel} watches random play in a grid world
 * (frames are one-hot images of the agent's cell) <em>without the actions</em>, and must
 * discover the four moves as latent codes. It is compared with the action-conditioned
 * {@link MlpWorldModel} trained on the same transitions <em>with</em> their actions.
 *
 * <p>Scores: <b>action recovery</b>, how well the latent codes inferred from frame pairs
 * identify the true moves (each code is mapped to its most frequent move); and
 * <b>controllability</b>, the share of next frames predicted in the right cell when the
 * model is driven by the code of the intended move, against the baseline driven by the true
 * action.
 */
@SuppressWarnings("exports")
public final class LatentActionPlay {
    private static final String[] MAP = {
        "S.....",
        "......",
        "..#...",
        "......",
        "....#.",
        ".....G"};

    /**
     * The outcome.
     *
     * @param actionRecovery the share of moving transitions whose latent code maps to the true move
     * @param latentCellAccuracy the share of next frames in the right cell, driven by latent codes
     * @param baselineCellAccuracy the same for the action-conditioned model, driven by the true actions
     */
    public record Result(double actionRecovery, double latentCellAccuracy, double baselineCellAccuracy) {
    }

    private LatentActionPlay() {
    }

    private static List<Transition> collect(int n, SplittableRandom r) {
        Environment env = GridWorld.of(MAP);
        List<Transition> out = new ArrayList<>(n);
        float[] obs = env.reset(r.split());
        while (out.size() < n) {
            float[] a = {r.nextInt(4)};
            Environment.Step s = env.step(a);
            out.add(new Transition(obs, a, s.reward(), s.observation(), s.terminated(), 0.99f));
            obs = s.done() ? env.reset(r.split()) : s.observation();
        }
        return out;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static boolean moved(Transition t) {
        return argmax(t.observation()) != argmax(t.nextObservation());
    }

    /**
     * Runs the comparison.
     *
     * @param trainSteps training steps of 64 transitions for both models
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int trainSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Transition> train = collect(3000, r);
        List<Transition> test = collect(500, r);
        int cells = train.get(0).observation().length;
        LatentActionWorldModel genie = LatentActionWorldModel.spatial(6, 6, 8, 64, seed);
        MlpWorldModel baseline = MlpWorldModel.of(cells, 4, 64, seed);
        for (int s = 0; s < trainSteps; s++) {
            List<Transition> batch = new ArrayList<>(64);
            for (int k = 0; k < 64; k++) {
                batch.add(train.get(r.nextInt(train.size())));
            }
            genie.temperature((float) Math.max(0.3, 1.0 - 0.7 * s / (0.7 * trainSteps))).train(batch);
            baseline.train(batch);
        }
        // map codes to moves by majority vote on moving training transitions (the only use of labels)
        int[][] votes = new int[genie.latentActions()][4];
        for (Transition t : train.subList(0, 1000)) {
            if (moved(t)) {
                votes[genie.inferAction(t.observation(), t.nextObservation())][Math.round(t.action()[0])]++;
            }
        }
        int[] codeToMove = new int[genie.latentActions()];
        int[] moveToCode = new int[4];
        int[] bestVotes = new int[4];
        for (int c = 0; c < votes.length; c++) {
            for (int m = 0; m < 4; m++) {
                if (votes[c][m] > votes[c][codeToMove[c]]) {
                    codeToMove[c] = m;
                }
                if (votes[c][m] > bestVotes[m]) {
                    bestVotes[m] = votes[c][m];
                    moveToCode[m] = c;
                }
            }
        }
        int moving = 0;
        int recovered = 0;
        int latentHit = 0;
        int baselineHit = 0;
        for (Transition t : test) {
            if (!moved(t)) {
                continue;
            }
            moving++;
            int move = Math.round(t.action()[0]);
            recovered += codeToMove[genie.inferAction(t.observation(), t.nextObservation())] == move ? 1 : 0;
            int target = argmax(t.nextObservation());
            latentHit += argmax(genie.predict(t.observation(), moveToCode[move])) == target ? 1 : 0;
            Tensor next = baseline.predict(Tensor.of(Shape.of(1, cells), t.observation()), new int[] {move}).next();
            baselineHit += argmax(next.toArray()) == target ? 1 : 0;
        }
        return new Result((double) recovered / moving, (double) latentHit / moving, (double) baselineHit / moving);
    }
}
