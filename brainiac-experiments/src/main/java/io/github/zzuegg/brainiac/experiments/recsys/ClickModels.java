package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.nn.CrossNetwork;
import io.github.zzuegg.brainiac.experimental.nn.FactorizationMachine;
import io.github.zzuegg.brainiac.experimental.recsys.WideAndDeep;
import io.github.zzuegg.brainiac.nn.Model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Deep click-through-rate models (W7 2.4: Wide &amp; Deep, DeepFM) against logistic regression,
 * the factorization machine and DCN-V2 on {@link SyntheticRecsys#clicks}, where a hidden
 * interaction between two categorical fields drives clicks. Every model gets the same Adam
 * budget; the metrics are test accuracy, ROC-AUC and log-loss. NeuMF, the third model of the
 * item, ranks items and is measured in {@link CollaborativeFiltering#neuMf}.
 */
public final class ClickModels {

    private ClickModels() {
    }

    /**
     * @param steps gradient steps per model
     * @param seed the seed
     * @return model name to test metrics
     */
    public static Map<String, BinaryBench.Result> run(int steps, long seed) {
        int fields = 3;
        int cats = 8;
        SyntheticRecsys.Table all = SyntheticRecsys.clicks(6000, fields, cats, seed);
        SyntheticRecsys.Table train = all.rows(0, 4000);
        SyntheticRecsys.Table test = all.rows(4000, 6000);
        int cols = fields * cats;
        Map<String, BinaryBench.Result> out = new LinkedHashMap<>();
        Model logistic = Model.sequential().input(cols).dense(1).seed(seed).build();
        out.put("logistic", BinaryBench.train(logistic::forward, logistic.parameters(), train, test, steps, 0.03f,
                seed));
        FactorizationMachine fm = new FactorizationMachine(cols, 8, new SplittableRandom(seed));
        out.put("FM", BinaryBench.train(fm::forward, fm.parameters(), train, test, steps, 0.03f, seed));
        CrossNetwork cross = new CrossNetwork(cols, 2, new SplittableRandom(seed));
        Model head = Model.sequential().input(cols).dense(1).seed(seed).build();
        List<Var> dcn = new ArrayList<>(cross.parameters());
        dcn.addAll(head.parameters());
        out.put("DCN-V2", BinaryBench.train((x, t) -> head.forward(cross.forward(x, t), t), dcn, train, test, steps,
                0.01f, seed));
        WideAndDeep wd = WideAndDeep.wideDeep(fields, cats, 8, 32, seed);
        out.put("Wide&Deep", BinaryBench.train(wd::forward, wd.parameters(), train, test, steps, 0.01f, seed));
        WideAndDeep dfm = WideAndDeep.deepFm(fields, cats, 8, 32, seed);
        out.put("DeepFM", BinaryBench.train(dfm::forward, dfm.parameters(), train, test, steps, 0.01f, seed));
        return out;
    }
}
