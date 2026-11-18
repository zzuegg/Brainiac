package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.experimental.compress.NetworkSlimming;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Channel pruning of the {@link ShapesCnn} (wave 8 item 8.5.6): network slimming (L1 on the
 * batch-norm scales, then removing the smallest-scale channels globally and fine-tuning) against
 * filter-magnitude pruning of a network trained without the penalty, a uniformly thinner network
 * of the same multiply-adds trained from scratch, and the slimmed shape trained from scratch.
 * Used by {@link PruningShootout#cnn}.
 */
final class CnnSlimming {

    private CnnSlimming() {
    }

    static List<CompactRow> run(double[] shares, int steps, int fineTune, float lambda, long seed) {
        Tensor[] train = ShapesCnn.data(192, seed);
        Tensor[] test = ShapesCnn.data(192, seed + 1);
        int c1 = 16;
        int c2 = 32;
        ShapesCnn plain = ShapesCnn.create(c1, c2, seed);
        plain.train(train, Optimizer.adam(0.01f), steps, 0);
        ShapesCnn sparse = ShapesCnn.create(c1, c2, seed);
        sparse.train(train, Optimizer.adam(0.01f), steps, lambda);
        List<CompactRow> rows = new ArrayList<>();
        rows.add(row("dense", plain, test));
        rows.add(row("dense+L1", sparse, test));
        for (double share : shares) {
            List<boolean[]> keep = NetworkSlimming.channelMasks(sparse.gammas(), share, 0.25);
            ShapesCnn slim = surgery(sparse, keep.get(0), keep.get(1));
            slim.train(train, Optimizer.adam(0.005f), fineTune, 0);
            rows.add(row("slimming " + share, slim, test));
            List<boolean[]> byNorm = NetworkSlimming.perLayer(List.of(NetworkSlimming.filterNorms(plain.w1().value()),
                    NetworkSlimming.filterNorms(plain.w2().value())), share);
            ShapesCnn filters = surgery(plain, byNorm.get(0), byNorm.get(1));
            filters.train(train, Optimizer.adam(0.005f), fineTune, 0);
            rows.add(row("filter-L1 " + share, filters, test));
            int[] ch = slim.channels();
            ShapesCnn scratchSlim = ShapesCnn.create(ch[0], ch[1], seed + 7);
            scratchSlim.train(train, Optimizer.adam(0.01f), steps + fineTune, 0);
            rows.add(row("slimmed shape from scratch", scratchSlim, test));
            int[] uni = uniformWidth(c1, c2, slim.macs());
            ShapesCnn uniform = ShapesCnn.create(uni[0], uni[1], seed + 7);
            uniform.train(train, Optimizer.adam(0.01f), steps + fineTune, 0);
            rows.add(row("uniform width from scratch", uniform, test));
        }
        return rows;
    }

    static ShapesCnn surgery(ShapesCnn net, boolean[] keep1, boolean[] keep2) {
        List<io.github.zzuegg.brainiac.autograd.Var> p = net.parameters();
        return new ShapesCnn(NetworkSlimming.selectConv(p.get(0).value(), keep1, null),
                NetworkSlimming.select(p.get(1).value(), keep1), NetworkSlimming.select(p.get(2).value(), keep1),
                NetworkSlimming.selectConv(p.get(3).value(), keep2, keep1),
                NetworkSlimming.select(p.get(4).value(), keep2), NetworkSlimming.select(p.get(5).value(), keep2),
                NetworkSlimming.selectRows(p.get(6).value(), keep2), p.get(7).value());
    }

    private static int[] uniformWidth(int c1, int c2, long macs) {
        int[] best = {1, 1};
        long gap = Long.MAX_VALUE;
        for (int a = 1; a <= c1; a++) {
            int b = Math.max(1, Math.round(a * c2 / (float) c1));
            long m = ShapesCnn.create(a, b, 0).macs();
            if (Math.abs(m - macs) < gap) {
                gap = Math.abs(m - macs);
                best = new int[] {a, b};
            }
        }
        return best;
    }

    private static CompactRow row(String method, ShapesCnn net, Tensor[] test) {
        int[] ch = net.channels();
        return new CompactRow(method + " (" + ch[0] + "/" + ch[1] + ")", net.accuracy(test), net.macs(),
                net.parameterCount(), net.latencyMillis(test[0]));
    }
}
