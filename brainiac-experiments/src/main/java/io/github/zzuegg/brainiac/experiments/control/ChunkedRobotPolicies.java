package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ActionTokenizer;
import io.github.zzuegg.brainiac.experimental.data.BinningActionTokenizer;
import io.github.zzuegg.brainiac.experimental.data.FastActionTokenizer;
import io.github.zzuegg.brainiac.experimental.rl.ActionChunking;
import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.DiffusionPolicy;
import io.github.zzuegg.brainiac.experimental.rl.FlowActionPolicy;
import io.github.zzuegg.brainiac.experimental.rl.MultimodalReach;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RealTimeChunking;
import io.github.zzuegg.brainiac.experimental.rl.TokenizedChunkPolicy;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Imitation on {@link MultimodalReach}, whose demonstrators pass an obstacle on the left or
 * the right with equal odds: the chunked generative robot policies against behaviour
 * cloning with an MSE head ({@link BehaviorCloning}) and with a Gaussian head.
 *
 * <ul>
 *   <li>{@link #imitation}: success rate of MSE BC, Gaussian BC, {@link DiffusionPolicy} and
 *       {@link FlowActionPolicy}, all from the same demonstrations.</li>
 *   <li>{@link #realTime}: the flow policy run with a simulated inference delay, switching
 *       chunks naively, by temporal ensembling or by real-time chunking
 *       ({@link RealTimeChunking}): success and the action jump at chunk boundaries.</li>
 *   <li>{@link #tokenized}: an autoregressive transformer policy over FAST action tokens
 *       ({@link FastActionTokenizer}) against the same policy over per-step uniform bins:
 *       success and tokens per chunk.</li>
 * </ul>
 */
@SuppressWarnings("exports")
public final class ChunkedRobotPolicies {
    /** The chunk length used throughout. */
    public static final int HORIZON = 8;

    private ChunkedRobotPolicies() {
    }

    /**
     * Collects demonstrations.
     *
     * @param episodes how many
     * @param seed the seed
     * @return the demonstrations
     */
    public static OfflineDataset demonstrations(int episodes, long seed) {
        return OfflineDataset.collect(new MultimodalReach(40), MultimodalReach.expert(0.05, seed), episodes, seed);
    }

    /**
     * Behaviour cloning with a Gaussian head: the network outputs a mean and a log standard
     * deviation per action and is trained by negative log-likelihood; acting samples.
     *
     * @param data the demonstrations
     * @param steps optimizer steps
     * @param seed the seed
     * @return the policy
     */
    public static Agent gaussianBc(OfflineDataset data, int steps, long seed) {
        Model net = Nets.mlp(2, 4, Activation.RELU, seed, 64, 64);
        Optimizer opt = Optimizer.adam(2e-3f);
        ActionChunking.Pairs pairs = ActionChunking.pairs(data, 1);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            float[] obs = new float[64 * 2];
            float[] act = new float[64 * 2];
            for (int i = 0; i < 64; i++) {
                int k = r.nextInt(pairs.size());
                System.arraycopy(pairs.observations()[k], 0, obs, 2 * i, 2);
                System.arraycopy(pairs.chunks()[k], 0, act, 2 * i, 2);
            }
            Var out = net.forward(Var.of(Tensor.of(Shape.of(64, 2), obs)), true);
            Var mean = out.slice(1, 0, 2);
            Var logStd = out.slice(1, 2, 4).clamp(-4, 1);
            Var z = Var.of(Tensor.of(Shape.of(64, 2), act)).sub(mean).div(logStd.exp());
            Var nll = z.square().mul(0.5f).add(logStd).mean();
            nll.backward();
            opt.step(net.parameters());
        }
        SplittableRandom noise = new SplittableRandom(seed + 1);
        return (obs, explore) -> {
            float[] o = net.predict(obs);
            float[] a = new float[2];
            for (int k = 0; k < 2; k++) {
                double std = Math.exp(Math.max(-4, Math.min(1, o[2 + k])));
                a[k] = (float) Math.max(-1, Math.min(1, o[k] + std * noise.nextGaussian()));
            }
            return a;
        };
    }

    /**
     * Trains every policy on the same demonstrations and scores them.
     *
     * @param demos the number of demonstrations
     * @param steps optimizer steps for each policy
     * @param episodes evaluation episodes
     * @param seed the seed
     * @return the success rate per method, plus the demonstrator's
     */
    public static Map<String, Double> imitation(int demos, int steps, int episodes, long seed) {
        OfflineDataset data = demonstrations(demos, seed);
        MultimodalReach env = new MultimodalReach(40);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("demonstrator", env.successRate(MultimodalReach.expert(0.05, seed + 5), episodes, seed));
        out.put("MSE BC", env.successRate(BehaviorCloning.continuous(Nets.mlp(2, 2, Activation.RELU, seed, 64, 64))
                .seed(seed).train(data, steps), episodes, seed));
        out.put("Gaussian BC", env.successRate(gaussianBc(data, steps, seed), episodes, seed));
        DiffusionPolicy dp = DiffusionPolicy.of(2, 2, HORIZON, 128, 2, seed).execute(4);
        dp.train(ActionChunking.pairs(data, HORIZON), steps, 64);
        out.put("Diffusion Policy", env.successRate(dp, episodes, seed));
        FlowActionPolicy fp = flowPolicy(data, steps, seed);
        out.put("flow action expert", env.successRate(fp, episodes, seed));
        return out;
    }

    /**
     * Trains the flow action expert.
     *
     * @param data the demonstrations
     * @param steps optimizer steps
     * @param seed the seed
     * @return the trained policy
     */
    public static FlowActionPolicy flowPolicy(OfflineDataset data, int steps, long seed) {
        FlowActionPolicy fp = FlowActionPolicy.of(2, 2, HORIZON, 128, 2, seed).execute(4);
        fp.train(ActionChunking.pairs(data, HORIZON), steps, 64);
        return fp;
    }

    /**
     * Autoregressive token policies ({@link TokenizedChunkPolicy}) with FAST tokens and with
     * per-step uniform binning, at the same budget.
     *
     * @param demos the number of demonstrations
     * @param steps optimizer steps for each policy
     * @param episodes evaluation episodes
     * @param seed the seed
     * @return per tokenizer, {@code {success rate, mean tokens per chunk, vocabulary size}}
     */
    public static Map<String, double[]> tokenized(int demos, int steps, int episodes, long seed) {
        OfflineDataset data = demonstrations(demos, seed);
        ActionChunking.Pairs pairs = ActionChunking.pairs(data, HORIZON);
        List<float[][]> chunks = new ArrayList<>();
        for (float[] c : pairs.chunks()) {
            chunks.add(ActionChunking.unflatten(c, 2));
        }
        Map<String, ActionTokenizer> tokenizers = new LinkedHashMap<>();
        tokenizers.put("FAST", FastActionTokenizer.train(chunks, 6f, 96));
        tokenizers.put("binning", new BinningActionTokenizer(HORIZON, 2, 64));
        MultimodalReach env = new MultimodalReach(40);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, ActionTokenizer> e : tokenizers.entrySet()) {
            ActionTokenizer tok = e.getValue();
            double tokens = 0;
            int max = 0;
            for (float[][] c : chunks) {
                int n = tok.encodeChunk(c).length;
                tokens += n;
                max = Math.max(max, n);
            }
            TokenizedChunkPolicy policy = TokenizedChunkPolicy.of(tok, 2, 64, 1.2f, max, 32, 2, seed).execute(4);
            policy.train(pairs, steps, 16);
            out.put(e.getKey(), new double[] {env.successRate(policy, episodes, seed), tokens / chunks.size(),
                tok.size()});
        }
        return out;
    }

    /**
     * Runs one trained flow policy with an inference delay under each switching mode.
     *
     * @param policy the trained flow policy
     * @param delay the inference delay in steps
     * @param period steps between switches
     * @param episodes evaluation episodes
     * @param seed the seed
     * @return per mode, {@code {success rate, mean action jump at switches, mean action change}}
     */
    public static Map<String, double[]> realTime(FlowActionPolicy policy, int delay, int period, int episodes,
            long seed) {
        MultimodalReach env = new MultimodalReach(40);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (RealTimeChunking.Mode mode : RealTimeChunking.Mode.values()) {
            RealTimeChunking agent = RealTimeChunking.of(policy, mode, delay, period, seed);
            double success = env.successRate(agent, episodes, seed);
            out.put(mode.toString(), new double[] {success, agent.meanBoundaryChange(), agent.meanChange()});
        }
        return out;
    }
}
