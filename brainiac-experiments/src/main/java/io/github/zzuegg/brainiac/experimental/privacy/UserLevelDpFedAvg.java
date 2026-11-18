package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.RdpAccountant;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * User-level differentially private federated averaging (DP-FedAvg; McMahan, Ramage, Talwar
 * and Zhang, ICLR 2018, <a href="https://arxiv.org/abs/1710.06963">arXiv:1710.06963</a>).
 * Every round each client is sampled independently with probability {@code q}; a sampled
 * client trains locally and its whole update {@code Δ_i = w_i - w} is clipped to L2 norm
 * {@code S}; the server adds Gaussian noise {@code N(0, σ² S²)} to the sum and divides by the
 * expected number of clients {@code q N}. Because the unit of clipping is the client, the
 * guarantee covers all of a user's data, not one example; the {@link RdpAccountant} analysis
 * of the sampled Gaussian mechanism applies unchanged with clients in place of examples.
 *
 * <p>Optionally the server sees only the sum, computed by {@link SecureAggregation} over the
 * clipped updates.
 */
public final class UserLevelDpFedAvg {

    private final List<Var> global;
    private final List<FederatedClient> clients;
    private final double clipNorm;
    private final double noiseMultiplier;
    private final double samplingRate;
    private final RdpAccountant accountant;
    private final SplittableRandom random;
    private boolean secure;
    private double dropout;

    private UserLevelDpFedAvg(List<Var> global, List<FederatedClient> clients, double clipNorm,
            double noiseMultiplier, double samplingRate, long seed) {
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.clipNorm = clipNorm;
        this.noiseMultiplier = noiseMultiplier;
        this.samplingRate = samplingRate;
        this.accountant = RdpAccountant.of(samplingRate, Math.max(noiseMultiplier, 1e-9));
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates the algorithm.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param clipNorm the per-client update norm bound {@code S}
     * @param noiseMultiplier the noise standard deviation over {@code S}; 0 for no privacy
     * @param samplingRate the per-client sampling probability {@code q}
     * @param seed the seed of sampling and noise
     * @return the algorithm
     */
    public static UserLevelDpFedAvg of(List<Var> global, List<FederatedClient> clients, double clipNorm,
            double noiseMultiplier, double samplingRate, long seed) {
        return new UserLevelDpFedAvg(global, clients, clipNorm, noiseMultiplier, samplingRate, seed);
    }

    /**
     * Sums the clipped updates with secure aggregation, with this share of the sampled
     * clients dropping out after the setup (their updates are lost).
     *
     * @param dropoutRate the dropout probability per client
     * @return this
     */
    public UserLevelDpFedAvg secureAggregation(double dropoutRate) {
        this.secure = true;
        this.dropout = dropoutRate;
        return this;
    }

    /**
     * One round.
     *
     * @param localSteps local SGD steps per client
     * @return how many clients contributed
     */
    public int round(int localSteps) {
        Tensor[] start = FederatedAlgorithms.values(global);
        int size = 0;
        for (Tensor t : start) {
            size += Math.toIntExact(t.size());
        }
        List<Integer> sampled = new java.util.ArrayList<>();
        for (int k = 0; k < clients.size(); k++) {
            if (random.nextDouble() < samplingRate) {
                sampled.add(k);
            }
        }
        float[][] updates = new float[sampled.size()][];
        for (int s = 0; s < sampled.size(); s++) {
            FederatedClient c = clients.get(sampled.get(s));
            c.load(List.of(start));
            c.train(localSteps);
            float[] u = new float[size];
            int o = 0;
            double norm = 0;
            for (int p = 0; p < start.length; p++) {
                float[] w = c.parameters().get(p).value().to(DType.FLOAT32).toArray();
                float[] w0 = start[p].to(DType.FLOAT32).toArray();
                for (int j = 0; j < w.length; j++) {
                    u[o + j] = w[j] - w0[j];
                    norm += u[o + j] * (double) u[o + j];
                }
                o += w.length;
            }
            double scale = Math.min(1, clipNorm / Math.max(Math.sqrt(norm), 1e-12));
            for (int j = 0; j < size; j++) {
                u[j] *= (float) scale;
            }
            updates[s] = u;
        }
        float[] sum = new float[size];
        int contributed = updates.length;
        if (secure && updates.length >= 2) {
            boolean[] dropped = new boolean[updates.length];
            int alive = 0;
            for (int s = 0; s < dropped.length; s++) {
                dropped[s] = random.nextDouble() < dropout;
                alive += dropped[s] ? 0 : 1;
            }
            if (alive < 2) {
                java.util.Arrays.fill(dropped, false);
                alive = dropped.length;
            }
            int threshold = Math.max(1, alive / 2);
            sum = SecureAggregation.of(updates.length, threshold, 1 << 20, random.nextLong())
                    .aggregate(updates, dropped).sum();
            contributed = alive;
        } else {
            for (float[] u : updates) {
                for (int j = 0; j < size; j++) {
                    sum[j] += u[j];
                }
            }
        }
        double expected = samplingRate * clients.size();
        int o = 0;
        for (int p = 0; p < start.length; p++) {
            float[] w0 = start[p].to(DType.FLOAT32).toArray();
            for (int j = 0; j < w0.length; j++) {
                double noisy = sum[o + j] + random.nextGaussian() * noiseMultiplier * clipNorm;
                w0[j] += (float) (noisy / expected);
            }
            o += w0.length;
            global.get(p).assign(Tensor.of(start[p].shape(), w0));
        }
        accountant.step(1);
        return contributed;
    }

    /**
     * @param delta the failure probability
     * @return the user-level {@code ε} spent so far
     */
    public double epsilon(double delta) {
        return noiseMultiplier <= 0 ? Double.POSITIVE_INFINITY : accountant.epsilon(delta);
    }
}
