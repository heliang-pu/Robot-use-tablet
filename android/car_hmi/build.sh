#!/usr/bin/env bash
# Builds the cockpit mock APK with the plain SDK tools (no Gradle).
#   ./build.sh            build build/carhmi.apk
#   ./build.sh install    build, install on the connected tablet and start it
# Needs: JDK 11+, Android SDK with build-tools 35.0.0 and platform android-35 (default ~/Android/Sdk).
set -euo pipefail
cd "$(dirname "$0")"
SDK="${ANDROID_HOME:-$HOME/Android/Sdk}"
BT="$SDK/build-tools/35.0.0"
JAR="$SDK/platforms/android-35/android.jar"
PKG=com.fmc3.carhmi

rm -rf build
mkdir -p build/gen build/cls build/dex
"$BT/aapt2" compile --dir res -o build/res.zip
"$BT/aapt2" link -o build/base.apk -I "$JAR" --manifest AndroidManifest.xml -A assets --java build/gen \
  --min-sdk-version 26 --target-sdk-version 34 --version-code 1 --version-name 1.0 build/res.zip
javac -source 11 -target 11 -Xlint:-options -cp "$JAR" -d build/cls $(find src build/gen -name '*.java')
"$BT/d8" --release --min-api 26 --lib "$JAR" --output build/dex $(find build/cls -name '*.class')
(cd build/dex && zip -q ../base.apk classes.dex)
"$BT/zipalign" -f -p 4 build/base.apk build/aligned.apk
# The signing key stays next to the project so that later builds install over earlier ones.
[ -f debug.keystore ] || keytool -genkeypair -keystore debug.keystore -storepass android -keypass android \
  -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US" >/dev/null 2>&1
"$BT/apksigner" sign --ks debug.keystore --ks-pass pass:android --key-pass pass:android --out build/carhmi.apk build/aligned.apk
ls -la build/carhmi.apk

if [ "${1:-}" = install ]; then
  adb install -r build/carhmi.apk
  adb shell am start -n "$PKG/.MainActivity"
fi
