# 车机模拟 L8（Cockpit mock）

机器人点按测试用的安卓 App，仿照理想 L8 中控屏的界面。**只做界面和显示，不实现真实车控**——每个控件点了画面会变（切页、开关、加减、滑条、拨号、歌词等），方便机器人点完从屏幕上读到结果。

参考来源：UXSync 竞品对标平台的理想 L8（Li OS 7.5.0）页面，以及公开的实拍图。界面为浅色主题的新版方向；部分小字和图标在清晰参考到手前为推测。

## 结构

- `assets/www/` — 界面本体，一个全屏 WebView 加载的单页应用
  - `index.html` / `style.css`
  - `app.js` — 页面与交互逻辑、触摸日志、音频播放
  - `data.js` — 设置项、联系人、曲目等显示内容
  - `icons.js` — 自绘的线条图标
  - `music/` — 导入的歌曲（不进 git，见下）
- `src/com/fmc3/carhmi/MainActivity.java` — 全屏 WebView 外壳
- `AndroidManifest.xml`、`res/` — 清单与资源（图标、主题）
- `build.sh` — 用 Android SDK 命令行工具直接打包（不需要 Gradle）
- `import_music.py` — 把一个文件夹里的 mp3（含 .lrc 歌词）导入音乐页

## 构建与安装

需要 JDK 11+、Android SDK（build-tools 35.0.0、platform android-35，默认 `~/Android/Sdk`）。

```bash
./build.sh            # 打包到 build/carhmi.apk
./build.sh install    # 打包并安装到已连接的平板后启动
```

包名 `com.fmc3.carhmi`。签名用的 `debug.keystore` 由 `build.sh` 首次构建时自动生成（不进 git）。

## 加音乐

歌曲是使用者自己的文件，不随仓库提交（`.gitignore` 排除了 `assets/www/music/`）。导入：

```bash
python3 import_music.py /path/to/音乐文件夹   # 需要 ffmpeg / ffprobe
./build.sh install
```

会把 mp3 拷进 `assets/www/music/`，读取内嵌封面和同名 `.lrc` 歌词，生成 `music/list.js`。没有导入时音乐页用占位曲目（无声）。

## 触摸日志

页面所有事件都写到 logcat，标签 `CarHMI`：每次按下/抬起的实际像素坐标、触发的控件、以及当前页所有控件的名称和位置。可用来算机器人的点击误差。

```bash
adb -s <ip:port> logcat -s CarHMI:I
```
