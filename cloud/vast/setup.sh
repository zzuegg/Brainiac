#!/usr/bin/env bash
# Prepares a rented NVIDIA machine (Vast.ai or any Ubuntu container with an NVIDIA GPU) for
# Brainiac and runs one job on it. Run from the top of a Brainiac checkout:
#
#   bash cloud/vast/setup.sh check            # Vulkan only: is NVIDIA's Vulkan driver usable here?
#   bash cloud/vast/setup.sh smoke            # check, JDKs, build, Vulkan backend tests, benchmarks
#   bash cloud/vast/setup.sh tests            # check, JDKs, ./gradlew build (every test)
#   bash cloud/vast/setup.sh bench            # check, JDKs, benchmarks only
#   bash cloud/vast/setup.sh worker [--models DIR]...   # a studio worker on 127.0.0.1:7870
#   bash cloud/vast/setup.sh run CMD...       # check, JDKs, then any command (e.g. ./gradlew ...)
#
# The VastCli launcher (brainiac-cloud) runs "check" first with only this script sent over (a
# host that fails it, exit 2, is destroyed and remembered), then ships the checkout and runs the job.
# Brainiac's GPU backend is Vulkan, not CUDA: NVIDIA's container runtime only mounts the
# driver's Vulkan libraries when the container was started with NVIDIA_DRIVER_CAPABILITIES
# including "graphics" (VastLaunch sets "all"). Hosts that hand out CUDA only fail the check.
set -euo pipefail

JOB="${1:-smoke}"
shift || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPT=/opt/brainiac
DATA="${BRAINIAC_DATA:-/root/brainiac-data}"
export DEBIAN_FRONTEND=noninteractive

step() { echo; echo "=== $* ($(date -u +%H:%M:%S) UTC)"; }
fail() { echo "BRAINIAC_SETUP failed: $*" >&2; exit 2; }

# ------------------------------------------------------------------ packages and Vulkan
packages() {
    step "packages"
    local need=()
    command -v curl >/dev/null || need+=(curl ca-certificates)
    command -v vulkaninfo >/dev/null || need+=(vulkan-tools)
    ldconfig -p | grep -q 'libvulkan.so.1' || need+=(libvulkan1)
    command -v tar >/dev/null || need+=(tar)
    # NVIDIA's Vulkan driver (libGLX_nvidia) links against X11 and GLVND client libraries,
    # which slim CUDA images leave out; without them the loader drops the driver
    for pkg in libx11-6 libxext6 libxcb1 libglvnd0 libegl1 libgl1; do
        dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg")
    done
    if ((${#need[@]})); then
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends "${need[@]}" >/dev/null
    fi
    echo "ok: ${need[*]:-nothing to install}"
}

vulkan() {
    step "GPU and Vulkan driver"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        || fail "nvidia-smi does not work: no NVIDIA GPU in this container"
    local icd
    icd=$(grep -lis nvidia /etc/vulkan/icd.d/*.json /usr/share/vulkan/icd.d/*.json 2>/dev/null | head -n1 || true)
    if [[ -z "$icd" ]]; then
        # the libraries can be mounted without the ICD file that points the loader at them
        local lib=""
        for l in libGLX_nvidia.so.0 libEGL_nvidia.so.0; do
            if ldconfig -p | grep -q "$l"; then lib=$l; break; fi
        done
        [[ -n "$lib" ]] || fail "the host gives this container NVIDIA's compute (CUDA) driver only, no graphics/Vulkan
libraries (libGLX_nvidia). Brainiac needs Vulkan: destroy this instance and pick another offer or host."
        mkdir -p /etc/vulkan/icd.d
        cat > /etc/vulkan/icd.d/nvidia_icd.json <<EOF
{ "file_format_version": "1.0.1", "ICD": { "library_path": "$lib", "api_version": "1.3" } }
EOF
        icd=/etc/vulkan/icd.d/nvidia_icd.json
        echo "wrote $icd for $lib"
    fi
    echo "ICD: $icd"
    # a driver library that still misses a dependency: say which one
    local drv
    drv=$(ldconfig -p | grep -m1 -oE '/[^ ]*libGLX_nvidia.so.0' || true)
    if [[ -n "$drv" ]] && ldd "$drv" | grep -q 'not found'; then
        echo "missing libraries of $drv:"; ldd "$drv" | grep 'not found'
    fi
    local summary
    summary=$(vulkaninfo --summary 2>&1) || {
        echo "$summary" | tail -n 30
        if echo "$summary" | grep -qE 'Found no drivers|ERROR_INCOMPATIBLE_DRIVER|ERROR_INITIALIZATION_FAILED'; then
            fail "the Vulkan loader finds no usable NVIDIA driver on this host (the ICD file names a library the
container does not have). Brainiac needs Vulkan: destroy this instance and pick another offer or host."
        fi
        fail "vulkaninfo failed"
    }
    echo "$summary" | sed -n '/^Devices:/,$p'
    echo "$summary" | grep -qiE 'deviceName *= *.*NVIDIA' \
        || fail "Vulkan works but shows no NVIDIA device (only a CPU driver?)"
    echo "BRAINIAC_SETUP vulkan ok"
}

# ------------------------------------------------------------------ JDKs
# Gradle runs on JDK 21 (as in CI); the build compiles and tests with JDK 27 (a toolchain).
jdk() {
    local version=$1 dir="$OPT/jdk-$1"
    if [[ -x "$dir/bin/java" ]]; then echo "$dir"; return; fi
    local tgz=/tmp/jdk-$version.tar.gz attempt
    # a download can arrive cut off or corrupt (seen on rented hosts): test the archive and
    # download it again instead of failing later inside tar or with a broken JDK
    for attempt in 1 2 3; do
        rm -f "$tgz"
        jdk_download "$version" "$tgz" && gzip -t "$tgz" 2>/dev/null && tar -tzf "$tgz" >/dev/null 2>&1 && break
        echo "JDK $version download $attempt was incomplete or corrupt; trying again" >&2
        [[ $attempt -lt 3 ]] || fail "no intact JDK $version download after 3 tries"
        sleep 5
    done
    rm -rf "$dir" && mkdir -p "$dir"
    tar -xzf "$tgz" -C "$dir" --strip-components=1
    rm -f "$tgz"
    [[ -x "$dir/bin/java" ]] || fail "JDK $version unpacked without bin/java"
    echo "$dir"
}

# Downloads JDK $1 to the file $2: Adoptium's newest GA or EA build, else OpenJDK's newest
# early-access build on jdk.java.net
jdk_download() {
    local version=$1 out=$2 url
    for kind in ga ea; do
        url="https://api.adoptium.net/v3/binary/latest/$version/$kind/linux/x64/jdk/hotspot/normal/eclipse"
        if curl -fsSL --retry 3 "$url" -o "$out" 2>/dev/null; then return 0; fi
    done
    url=$(curl -fsSL "https://jdk.java.net/$version/" | grep -oE "https://download.java.net/[^\"]+linux-x64_bin.tar.gz" | head -n1)
    [[ -n "$url" ]] || { echo "no JDK $version download found" >&2; return 1; }
    curl -fsSL --retry 3 "$url" -o "$out"
}

jdks() {
    step "JDKs"
    JDK21=$(jdk 21)
    JDK27=$(jdk 27)
    "$JDK21/bin/java" -version 2>&1 | head -n1
    "$JDK27/bin/java" -version 2>&1 | head -n1
    export JAVA_HOME="$JDK21"
    # more room on LWJGL's thread stack (64 KB by default) for drivers that report a lot,
    # passed on to every JVM Gradle starts (tests, run tasks)
    export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Dorg.lwjgl.system.stackSize=512"
    export PATH="$JAVA_HOME/bin:$PATH"
    # so that any ./gradlew (for example in "run CMD") finds JDK 27 without extra flags
    mkdir -p ~/.gradle
    printf 'org.gradle.java.installations.paths=%s\norg.gradle.java.installations.auto-download=false\n' "$JDK27" > ~/.gradle/gradle.properties
    GRADLE=(./gradlew --console=plain "-Porg.gradle.java.installations.paths=$JDK27"
        -Porg.gradle.java.installations.auto-download=false)
}

# ------------------------------------------------------------------ jobs
build() {
    step "build"
    cd "$ROOT"
    time "${GRADLE[@]}" assemble -x javadoc
}

gpu_tests() {
    step "Vulkan backend tests"
    cd "$ROOT"
    time "${GRADLE[@]}" :brainiac-backend-vulkan:test
}

bench() {
    cd "$ROOT"
    step "benchmarks on the GPU"
    time "${GRADLE[@]}" :brainiac-benchmarks:run --args="--quick --device vulkan"
    step "matrix product speeds (MatmulBenchmarkTest)"
    "${GRADLE[@]}" :brainiac-backend-vulkan:test --tests '*MatmulBenchmark*' -Pbench --rerun-tasks || true
    step "training: MNIST mlp and cnn, 1 epoch each, on the GPU"
    time "${GRADLE[@]}" :brainiac-examples:run --args="bench mnist mlp cnn --epochs 1 --device gpu"
}

worker() {
    local port="${BRAINIAC_WORKER_PORT:-7870}"
    local args=(--worker --listen 127.0.0.1 --port "$port" --data "$DATA")
    while (($#)); do
        case "$1" in
            --models) args+=(--models "$2"); mkdir -p "$2"; shift 2 ;;
            *) args+=("$1"); shift ;;
        esac
    done
    mkdir -p "$DATA"
    [[ -n "${BRAINIAC_WORKER_TOKEN:-}" ]] || echo "note: no BRAINIAC_WORKER_TOKEN; the worker makes its own token in $DATA/worker/worker.token"
    step "studio worker on 127.0.0.1:$port (reach it through an SSH tunnel)"
    cd "$ROOT"
    "${GRADLE[@]}" :brainiac-studio:installDist -x javadoc -q
    # announce once the port answers; the token is never printed here
    ( for _ in $(seq 1 300); do
          if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then echo "BRAINIAC_WORKER ready port=$port"; exit; fi
          sleep 1
      done ) &
    export JAVA_HOME="$JDK27"
    brainiac-studio/build/install/brainiac-studio/bin/brainiac-studio "${args[@]}" </dev/null &
    local pid=$!
    if [[ -n "${BRAINIAC_WORKER_UNTIL_EOF:-}" ]]; then
        # the launcher keeps this input open while it runs: when it goes away (even killed),
        # the input ends and the worker stops with it instead of holding the GPU
        ( cat >/dev/null; kill "$pid" 2>/dev/null ) &
    fi
    wait "$pid"
}

case "$JOB" in
    check) packages; vulkan ;;
    smoke) packages; vulkan; jdks; build; gpu_tests; bench ;;
    tests) packages; vulkan; jdks; step "./gradlew build"; cd "$ROOT"; time "${GRADLE[@]}" build ;;
    bench) packages; vulkan; jdks; bench ;;
    worker) packages; vulkan; jdks; worker "$@" ;;
    run) packages; vulkan; jdks; cd "$ROOT"; step "$*"; "$@" ;;
    *) fail "unknown job $JOB: use check, smoke, tests, bench, worker or run CMD..." ;;
esac
step "BRAINIAC_SETUP done: $JOB"
