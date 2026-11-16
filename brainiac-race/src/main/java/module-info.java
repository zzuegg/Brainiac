/**
 * Multi-car racing: a track, cars with simple tyre physics, slipstream and collisions,
 * exposed as a {@link io.github.zzuegg.brainiac.rl.VectorEnvironment} so one policy can
 * learn to race against copies of itself, plus a window to watch the race.
 */
module io.github.zzuegg.brainiac.race {
    requires transitive io.github.zzuegg.brainiac.rl;
    requires transitive java.desktop;
    exports io.github.zzuegg.brainiac.race;
}
