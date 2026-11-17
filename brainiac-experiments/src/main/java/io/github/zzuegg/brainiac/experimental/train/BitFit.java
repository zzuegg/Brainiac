package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.ArrayList;
import java.util.List;

/**
 * BitFit (Ben Zaken et al. 2021, <a href="https://arxiv.org/abs/2106.10199">arXiv:2106.10199</a>):
 * fine-tune only the bias terms of a pretrained model, well under 0.1% of its parameters,
 * which is competitive with full fine-tuning on small and medium data. The selection is by
 * shape: every one-dimensional parameter is a bias (norm gains and shifts are
 * one-dimensional too and are included, as in most BitFit implementations).
 */
public final class BitFit {

    private BitFit() {
    }

    /**
     * @param parameters a model's parameters
     * @return the one-dimensional ones (biases, norm gains and shifts)
     */
    public static List<Var> biases(List<Var> parameters) {
        List<Var> out = new ArrayList<>();
        for (Var p : parameters) {
            if (p.shape().rank() == 1) {
                out.add(p);
            }
        }
        return out;
    }

    /**
     * @param selected the trained parameters
     * @param all every parameter
     * @return the share of values trained
     */
    public static double share(List<Var> selected, List<Var> all) {
        long s = 0;
        long a = 0;
        for (Var v : selected) {
            s += v.shape().size();
        }
        for (Var v : all) {
            a += v.shape().size();
        }
        return a == 0 ? 0 : (double) s / a;
    }
}
