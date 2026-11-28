cd "$LOCALAPPDATA/Temp/kalimo-test"; CP=$(cat cp.txt); J="/c/Program Files/Java/jdk-27/bin/java"; O="C:\Users\micha\AppData\Local\Temp\kalimo-test\game3n0"
R() { "$J" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Dfile.encoding=UTF-8 -Dbrainiac.vulkan.memory=10000 "-Dref.de=$O\reference-de.wav" "-Dreftext.de=$O\reference-de.txt" -cp "classes;$CP" "$@"; }
gen() { for i in $(seq 1 40); do R io.github.zzuegg.brainiac.audio.Game2 "$O" "$@" > game3n0-gen.log 2>&1 && return 0; echo "restart gen $i"; done; }
judge() { for i in $(seq 1 15); do R JudgeLang "$O\takes" > game3n0-judge.log 2>&1 && return 0; echo "restart judge $i"; done; }
gen lines.tsv 0 4 de; judge; python src/game3n0_pick.py 1
if [ -s game3n0/retry.tsv ]; then gen retry.tsv 4 4 de; judge; python src/game3n0_pick.py 2; fi
if [ -s game3n0/retry.tsv ]; then gen retry.tsv 8 4 de; judge; fi
python src/game3n0_pick.py final
