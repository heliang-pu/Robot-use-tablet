'use strict';
// Display content of the mock. Everything here is made up for the bench.

// Songs come from music/list.js (written by import_music.py); the stations have no sound.
const GRAD = [['#5b8cff', '#8fd3ff'], ['#ff8a5b', '#ffd08f'], ['#6fcf97', '#b8f2d0'], ['#9b6bff', '#d6b8ff'], ['#ff6b9d', '#ffc1d6'], ['#f2c94c', '#fff0b3']];
const RADIO = [
  { t: '交通广播', a: 'FM 103.9 · 正在直播', d: 3600, c: GRAD[0], g: 1 },
  { t: '晨间新闻', a: '新闻综合频率', d: 1800, c: GRAD[1], g: 1 },
];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
// lrc: timed lyric lines as [seconds, text]
const SONGS = (typeof MUSIC !== 'undefined' ? MUSIC : []).map((m, i) => ({ t: esc(m.t), a: esc(m.a), d: m.d, f: m.f, cover: m.cover, lrc: m.lrc || [], c: GRAD[i % GRAD.length], g: 0 }));
// Without imported songs the list falls back to silent placeholders.
const TRACKS = (SONGS.length ? SONGS : [
  { t: '轻音乐精选', a: '纯音乐电台', d: 245, c: GRAD[2], g: 0 },
  { t: '公路旅行歌单', a: '精选合辑', d: 212, c: GRAD[3], g: 0 },
  { t: '钢琴时光', a: '古典音乐', d: 328, c: GRAD[4], g: 0 },
]).concat(RADIO);

const CONTACTS = [
  ['A', 'Aaron', '138 0000 0101'], ['A', '阿姨', '138 0000 0102'],
  ['B', '爸爸', '138 0000 0103'], ['B', 'Bella', '138 0000 0104'],
  ['D', '道路救援', '400 000 0105'], ['G', '公司前台', '010 0000 0106'],
  ['K', '客服中心', '400 000 0107'], ['L', '李四', '138 0000 0108'],
  ['M', '妈妈', '138 0000 0109'], ['W', '王五', '138 0000 0110'],
  ['Z', '张三', '138 0000 0111'], ['Z', '赵六', '138 0000 0112'],
];

const RECENTS = [
  ['妈妈', '138 0000 0109', '今天 09:12', 'out'], ['公司前台', '010 0000 0106', '今天 08:40', 'in'],
  ['张三', '138 0000 0111', '昨天 18:25', 'miss'], ['道路救援', '400 000 0105', '昨天 12:03', 'out'],
  ['李四', '138 0000 0108', '周三 20:16', 'in'], ['客服中心', '400 000 0107', '周二 10:48', 'out'],
];

const PLACES = [
  ['家', '滨河路 18 号', '8.6', '18'], ['公司', '科技大道 66 号', '12.4', '25'],
  ['中心公园', '公园东路 1 号', '3.2', '9'], ['人民医院', '健康路 120 号', '5.7', '14'],
  ['购物中心', '创新路 88 号', '4.1', '11'], ['体育中心', '体育场路 9 号', '9.8', '21'],
  ['高铁站', '站前大道 1 号', '16.3', '32'], ['超级充电站', '科技大道 20 号', '2.4', '7'],
];

const APPS = [
  ['nav', '导航', 'nav', '#2f7bff'], ['music', '音乐', 'music', '#ff4d6a'], ['phone', '电话', 'phone', '#1fb85a'],
  ['settings', '车辆设置', 'gear', '#6b7a90'], ['climate', '空调', 'fan', '#19a7e0'], ['video', '视频', 'video', '#8a5cf6'],
  ['camera', '相机', 'camera', '#ff8a1f'], ['weather', '天气', 'cloud', '#3aa0ff'], ['calendar', '日历', 'cal', '#f04a4a'],
  ['charge', '充电地图', 'bolt', '#12b886'], ['manual', '用户手册', 'book', '#5d6cf0'], ['tasks', '任务大师', 'list', '#e0a100'],
  ['game', '游戏', 'game', '#e64980'], ['browser', '浏览器', 'globe', '#2aa9a0'], ['album', '相册', 'image', '#f76707'],
  ['energy', '能耗', 'gauge', '#40a02b'], ['btapp', '蓝牙', 'bt', '#1c7ed6'], ['dvr', '行车记录', 'video', '#495057'],
];

const AMB = ['#ff3b3b', '#ff8a1f', '#ffd60a', '#34c759', '#19c8d8', '#2f6bff', '#8a5cf6', '#ff4fa3'];

// --- settings pages -------------------------------------------------------
const seg = (id, label, opts, def, hint, auto) => ({ k: 'seg', id, label, opts, def, hint, auto });
const tg = (id, label, on, hint) => ({ k: 'seg', id, label, opts: ['开启', '关闭'], def: on ? 0 : 1, hint });
const sld = (id, label, def, icon, unit, auto) => ({ k: 'sl', id, label, def, icon, unit, auto });
const btn = (id, label, text, msg) => ({ k: 'btn', id, label, text, msg });
const info = (label, val) => ({ k: 'info', label, val });
const devs = (id, label, items, def) => ({ k: 'devs', id, label, items, def });

const SET = [
  ['vehicle', '车辆', [
    ['quick', '快捷', [
      { k: 'tiles', items: [['lock', '中控锁', 'lock', true], ['window', '车窗通风', 'window', false], ['trunk', '尾门', 'trunk', false],
        ['mirror', '后视镜折叠', 'mirror', false], ['shade', '遮阳帘', 'shade', false], ['child', '儿童锁', 'child', false],
        ['wiperm', '雨刮维修', 'wiper', false], ['fuelcap', '加油小门', 'fuel', false], ['chargecap', '充电口盖', 'bolt', false]] },
      seg('drive_mode', '驾驶模式', ['舒适', '标准', '运动'], 1),
    ]],
    ['light', '灯光', [
      { k: 'colors', id: 'amb_color', label: '氛围灯颜色', def: 5 },
      sld('amb_bright', '氛围灯亮度', 35, 'sun', '%'),
      seg('amb_mode', '氛围灯模式', ['单色模式', '多色模式'], 0, '多色模式下氛围灯跟随音乐变化'),
      seg('headlamp', '前大灯', ['关闭', '位置灯', '近光灯', '自动'], 3),
      tg('auto_high', '智能远光灯', true, '会车时自动切换为近光'),
    ]],
    ['door', '门窗', [
      seg('win_all', '车窗', ['全部关闭', '通风', '全部打开'], 0),
      tg('lock_winup', '锁车自动升窗', true),
      tg('near_unlock', '靠近解锁', true, '携带钥匙靠近车辆时自动解锁'),
      tg('leave_lock', '离车上锁', true),
      seg('unlock_way', '解锁方式', ['全车', '仅主驾'], 0),
    ]],
    ['drive', '驾驶', [
      seg('accel', '加速模式', ['舒适', '标准', '运动'], 1),
      seg('regen', '能量回收', ['舒适', '标准', '强'], 1),
      seg('steer', '转向助力', ['舒适', '标准', '运动'], 1),
      seg('susp', '悬架高度', ['低', '标准', '高'], 1),
      tg('auto_hold', '自动驻车', true),
    ]],
    ['mirr', '后视镜', [
      tg('mir_autofold', '锁车自动折叠', true),
      tg('mir_tilt', '倒车自动下翻', false),
      tg('mir_heat', '后视镜加热', false),
      btn('mir_adj', '后视镜调节', '开始调节', '请使用方向盘右侧按键调节'),
    ]],
    ['seats', '座椅', [
      tg('seat_welcome', '迎宾座椅', true, '开门时座椅自动后移'),
      tg('seat_autoheat', '座椅加热自动开启', false),
      seg('seat_mem', '主驾位置记忆', ['位置 1', '位置 2', '位置 3'], 0),
      btn('seat_save', '保存当前位置', '保存', '座椅位置已保存'),
    ]],
    ['charge', '充电', [
      sld('chg_limit', '充电上限', 90, 'bolt', '%'),
      tg('chg_sched', '预约充电', false, '23:00 至次日 07:00'),
      tg('v2l', '对外放电', false),
      tg('chg_capauto', '充电口盖自动关闭', true),
    ]],
    ['tail', '尾门', [
      sld('tail_height', '尾门开启高度', 80, 'trunk', '%'),
      tg('tail_kick', '脚踢感应开启', true),
      btn('tail_open', '尾门', '打开尾门', '尾门已打开'),
    ]],
    ['maint', '维护', [
      tg('wash_mode', '洗车模式', false, '关闭车窗、折叠后视镜并关闭自动雨刮'),
      tg('tow_mode', '拖车模式', false),
      btn('tpms_reset', '胎压复位', '复位', '胎压已复位'),
      btn('wiper_fluid', '玻璃水', '查看', '玻璃水液位正常'),
    ]],
  ]],
  ['adas', '智能驾驶', [
    ['assist', '辅助驾驶', [
      tg('noa', '导航辅助驾驶', true),
      tg('lka', '车道保持辅助', true),
      seg('follow', '跟车距离', ['近', '中', '远'], 1),
      seg('lane_style', '变道风格', ['舒适', '标准', '高效'], 1),
      tg('speed_warn', '超速提醒', true),
    ]],
    ['safety', '主动安全', [
      tg('fcw', '前向碰撞预警', true),
      seg('fcw_sens', '预警灵敏度', ['低', '中', '高'], 1),
      tg('aeb', '自动紧急制动', true),
      tg('bsd', '盲区监测', true),
      tg('dow', '开门预警', true),
    ]],
    ['parking', '泊车', [
      tg('apa', '自动泊车', true),
      tg('rpa', '遥控泊车', false),
      tg('rvc_auto', '倒车影像自动开启', true),
      seg('radar_vol', '雷达提示音', ['关', '低', '高'], 1),
    ]],
  ]],
  ['navset', '导航', [
    ['pref', '偏好', [
      seg('route_pref', '路线偏好', ['智能推荐', '时间优先', '少收费', '不走高速'], 0),
      tg('avoid_jam', '躲避拥堵', true),
      tg('chg_plan', '充电规划', true, '长途导航时自动添加充电站'),
    ]],
    ['voice', '播报', [
      seg('guide_mode', '播报模式', ['详细', '简洁', '静音'], 0),
      sld('guide_vol', '导航音量', 60, 'vol', ''),
      tg('cam_warn', '电子眼播报', true),
    ]],
    ['mapset', '地图', [
      seg('map_view', '地图视角', ['2D 正北', '2D 车头', '3D'], 1),
      tg('traffic', '实时路况', true),
      seg('map_day', '昼夜模式', ['自动', '白天', '夜间'], 0),
    ]],
  ]],
  ['display', '显示', [
    ['screen', '屏幕', [
      sld('bright', '亮度', 100, 'sun', '', 'bright_auto'),
      seg('theme', '外观', ['浅色', '深色'], 0, '自动模式下根据日出日落切换', 'theme_auto'),
      seg('font', '字体大小', ['标准', '大'], 0),
      tg('screensaver', '屏保时钟', false, '驻车 5 分钟后显示时钟'),
      btn('clean', '屏幕清洁', '进入清洁模式', '清洁模式：屏幕触摸已暂停 30 秒'),
    ]],
    ['hud', 'HUD', [
      tg('hud_on', 'HUD 显示', true),
      sld('hud_bright', 'HUD 亮度', 60, 'sun', ''),
      sld('hud_height', 'HUD 高度', 50, 'up', ''),
      seg('hud_mode', '显示内容', ['简洁', '标准', '丰富'], 1),
    ]],
    ['wallp', '壁纸', [
      { k: 'wall', id: 'wall', label: '壁纸', def: 0 },
      tg('wall_dyn', '动态壁纸', false),
    ]],
  ]],
  ['sound', '声音', [
    ['volume', '音量', [
      sld('vol_media', '媒体音量', 45, 'vol', ''),
      sld('vol_call', '通话音量', 55, 'vol', ''),
      sld('vol_voice', '语音助手音量', 60, 'vol', ''),
      tg('speed_vol', '随速音量', true),
    ]],
    ['effect', '音效', [
      seg('eq', '音效模式', ['标准', '影院', '人声', '自定义'], 0),
      seg('field', '声场位置', ['全车', '主驾', '前排', '后排'], 0),
      tg('surround', '全景声', true),
    ]],
    ['tone', '提示音', [
      tg('key_tone', '按键音', true),
      tg('lock_tone', '锁车提示音', true),
      seg('tone_vol', '提示音量', ['低', '中', '高'], 1),
    ]],
  ]],
  ['conn', '连接', [
    ['bt', '蓝牙', [
      tg('bt_on', '蓝牙', true),
      devs('bt_dev', '已配对的设备', [['我的手机', '已连接'], ['平板电脑', '未连接'], ['蓝牙耳机', '未连接']], 0),
    ]],
    ['wlan', 'WLAN', [
      tg('wlan_on', 'WLAN', true),
      devs('wlan_dev', '可用网络', [['Home-5G', '已连接'], ['Office-WiFi', '已保存'], ['Guest', '开放'], ['Parking-Free', '开放']], 0),
    ]],
    ['hotspot', '热点', [
      tg('hot_on', '车载热点', false),
      info('热点名称', 'L8-0001'),
      info('密码', '12345678'),
      seg('hot_band', '频段', ['2.4G', '5G'], 1),
    ]],
  ]],
  ['assistant', '理想同学', [
    ['wake', '唤醒', [
      tg('wake_on', '语音唤醒', true),
      tg('wake_free', '免唤醒指令', true, '可直接说"上一首""接听电话"'),
      tg('cont_talk', '连续对话', true),
      seg('wake_zone', '唤醒音区', ['全车', '仅主驾'], 0),
    ]],
    ['timbre', '音色', [
      seg('tts_voice', '音色', ['标准女声', '温柔女声', '标准男声'], 0),
      seg('tts_speed', '语速', ['慢', '标准', '快'], 1),
      seg('lang', '语言', ['普通话', 'English', 'Deutsch'], 0),
    ]],
  ]],
  ['secure', '安全', [
    ['sentry', '哨兵', [
      tg('sentry_on', '哨兵模式', false, '车辆锁车后监控周围环境'),
      seg('sentry_sens', '灵敏度', ['低', '中', '高'], 1),
      tg('dvr_on', '行车记录仪', true),
    ]],
    ['kids', '儿童锁', [
      tg('child_l', '左后门儿童锁', false),
      tg('child_r', '右后门儿童锁', false),
      tg('rear_lock', '后排屏幕锁', false),
    ]],
  ]],
  ['system', '系统', [
    ['general', '通用', [
      seg('unit_dist', '距离单位', ['公里', '英里'], 0),
      seg('unit_temp', '温度单位', ['℃', '℉'], 0),
      seg('time_fmt', '时间格式', ['24 小时', '12 小时'], 0),
      btn('factory', '恢复出厂设置', '恢复', '演示界面：未执行恢复'),
    ]],
    ['ota', '版本', [
      info('系统版本', 'Li OS 7.5.0'),
      btn('ota_check', '系统更新', '检查更新', '当前已是最新版本'),
      tg('ota_auto', '自动下载更新', true),
    ]],
    ['about', '关于', [
      info('车型', 'L8'),
      info('车辆识别码', 'LW000000000000001'),
      info('存储空间', '已用 38 GB / 共 128 GB'),
      info('蓝牙名称', 'L8-0001'),
    ]],
  ]],
];
