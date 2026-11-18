/**
 * Experimental security tools for learning systems, written as if they were part of a core
 * package {@code io.github.zzuegg.brainiac.security}: Byzantine-robust federated aggregation
 * (Krum, median, trimmed mean, geometric median, bucketing, centred clipping, nearest-neighbour
 * mixing, learned aggregation weights) and the model-poisoning attacks that stress it;
 * backdoor triggers, poisoned-sample detectors and model repair; score-based, decision-based
 * and transfer attacks with a query-counting oracle; AutoAttack evaluation; and model
 * extraction with its defences. Once one proves itself it moves to core unchanged apart from
 * dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.security;
