# 测试截图的来源

这里的三张“平板截图”都是**合成图**，由 `scripts/smoke_vlm_cpu.py --make-screens` 用 PIL 画出来
（字体：系统自带的 Noto Sans CJK SC，SIL OFL 许可）。没有任何图片来自网络，所以没有来源网址。

为什么没用真实截图：2026-11-23 在 Wikimedia Commons 上按关键词和分类找过可自由使用的横屏安卓平板截图
（"Android tablet screenshot home screen"、"Android 12L screenshot"、"Android 13 tablet screenshot"、
"Android music player tablet screenshot"、"Android tablet settings screenshot"，以及
Category:Android 12/13/14 screenshots 等分类），没找到同时满足“横屏平板、有音乐播放页/桌面/设置页、许可清楚”的图
（唯一命中的横屏图是 File:Elgato SAT)IP app on Nexus 7.png，是一个电视应用的界面，用不上），
之后 Commons 的接口开始限流，就改用合成图。

| 文件 | 内容 | 标注 |
|---|---|---|
| home.png | 桌面：6x2 个带中文标签的应用图标（音乐、相机、设置、图库、浏览器…）、搜索栏、停靠栏 | home.json |
| player.png | 音乐播放页（暂停）：封面、歌名、进度条、随机/上一首/播放/下一首/循环 | player.json |
| settings.png | 设置页（双栏）：左栏分类，右栏“声音和振动”的音量滑块和开关 | settings.json |

每个 json 里：`elements` 是各个可点元素的真值框 `[x0, y0, x1, y1]`（像素，1280x800，含边界），
`instructions` 是要问模型的中文指令和它对应的目标元素。

注意：合成图比真实截图干净、简单（没有真实壁纸、没有真实应用的图标细节），这里量到的命中率只能当作
链路是否通、以及画质退化影响大小的第一手参考，不能当作真实平板上的成功率。
