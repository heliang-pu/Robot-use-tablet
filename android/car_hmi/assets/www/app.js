'use strict';
// Cockpit screen mock for the tap bench. Display and touch feedback only: nothing here controls anything.
// Every touch and every action is written to the console as a line starting with "HMI " (logcat tag CarHMI on the tablet).

const stage = document.getElementById('stage');
let H = {};
const LOG = o => { try { console.log('HMI ' + JSON.stringify(o)); } catch (e) { /* ignore */ } };
const pad2 = n => String(n).padStart(2, '0');
const mmss = s => pad2(Math.floor(s / 60)) + ':' + pad2(Math.floor(s % 60));
const kid = k => (k === '*' ? 'star' : k === '#' ? 'hash' : k);

const S = {
  page: 'home', prev: 'home', stub: null, gear: 'P',
  set: { cat: 'vehicle', sub: {} },
  v: { bright_auto: true, theme_auto: false, park: true, nap: false, wheel: false, rdef: false },
  phone: { tab: 'keypad', num: '', call: null, hide: false, recents: RECENTS.slice() },
  m: { i: 0, play: false, pos: 0, fav: { 0: true }, tab: 0, mode: 0, lyr: true },
  nav: { st: 'idle', dest: null, zoom: 2, traffic: true, view: 0, opt: 0 },
  c: { on: true, auto: true, ac: true, max: false, tL: 24, tR: 24, fan: 3, mode: 0, sync: true,
    ron: false, rlock: false, rauto: true, rsync: true, rt: 24, rfan: 2, rmode: 0,
    fdef: false, rec: false, pure: true, scent: false, ion: false,
    // per seat: [heating level, ventilation level, massage level or folded]
    seat: { fl: [0, 2, 0], fr: [0, 2, 0], ml: [0, 0, 0], mr: [0, 0, 0], rl: [0, 0, 0], rr: [0, 0, 0] } },
  scn: {},
  toast: null,
};
SET.forEach(c => {
  S.set.sub[c[0]] = c[2][0][0];
  c[2].forEach(s => s[2].forEach(it => {
    if (it.k === 'tiles') it.items.forEach(t => { S.v[t[0]] = t[3]; });
    else if (it.id) S.v[it.id] = it.def;
  }));
});

// One tappable control. `name` is the readable name that goes into the screen log.
function b(id, cls, html, fn, name, attr) {
  H[id] = fn;
  return '<div class="b ' + cls + '" data-id="' + id + '"' + (name ? ' data-n="' + name + '"' : '') + (attr ? ' ' + attr : '') + '>' + html + '</div>';
}
function toast(msg) {
  S.toast = msg;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { S.toast = null; const e = document.getElementById('toast'); if (e) e.remove(); }, 2200);
}
function go(p) { if (S.page !== p) S.prev = S.page; S.page = p; }
function tgl(k, on, off) { S.v[k] = !S.v[k]; if (on) toast(S.v[k] ? on : off); }
function cyc(a, i) { a[i] = a[i] === 0 ? 3 : a[i] - 1; }
function tempTxt(t) { return t <= 16 ? '低' : t >= 32 ? '高' : t.toFixed(1); }
function track(d) {
  const n = TRACKS.length;
  if (S.m.mode === 2 && n > 2) { let j = S.m.i; while (j === S.m.i) j = Math.floor(Math.random() * n); S.m.i = j; }
  else S.m.i = (S.m.i + d + n) % n;
  S.m.pos = 0;
}

// Real playback for the tracks that have a file (music/NN.mp3); the radio entries stay silent.
const AU = new Audio();
AU.preload = 'auto';
let auIdx = -1;
function syncAudio() {
  const t = TRACKS[S.m.i];
  AU.volume = Math.max(0, Math.min(1, S.v.vol_media / 100));
  if (!t.f) { if (!AU.paused) AU.pause(); auIdx = -1; return; }
  if (auIdx !== S.m.i) { auIdx = S.m.i; AU.src = 'music/' + t.f + '.mp3'; }
  if (S.m.play && AU.paused) AU.play().catch(e => LOG({ ev: 'audio', st: 'error', msg: String(e) }));
  if (!S.m.play && !AU.paused) AU.pause();
}
AU.addEventListener('playing', () => LOG({ ev: 'audio', st: 'playing', track: TRACKS[S.m.i].t, at: Math.round(AU.currentTime) }));
AU.addEventListener('pause', () => LOG({ ev: 'audio', st: 'paused', track: TRACKS[S.m.i].t, at: Math.round(AU.currentTime) }));
AU.addEventListener('error', () => LOG({ ev: 'audio', st: 'error', track: TRACKS[S.m.i].t }));
// Lyrics: the line that belongs to the current playing time, a 7-line window around it, and seeking by line.
let lyrShown = -2;
function lyrIdx(t) {
  const L = t.lrc;
  if (!L || !L.length) return -1;
  const now = (t.f && auIdx === S.m.i ? AU.currentTime : S.m.pos) + 0.15;
  let i = -1;
  for (let k = 0; k < L.length && L[k][0] <= now; k++) i = k;
  return i;
}
function seek(sec) { if (TRACKS[S.m.i].f) { AU.currentTime = sec; S.m.pos = sec; LOG({ ev: 'audio', st: 'seek', to: sec }); } }
function lyrNow() {
  const t = TRACKS[S.m.i], cur = lyrIdx(t);
  lyrShown = cur;
  return cur >= 0 ? esc(t.lrc[cur][1]) : '';
}
function lyrLines() {
  const t = TRACKS[S.m.i], L = t.lrc || [];
  if (!L.length) return '<div class="ll">暂无歌词</div>';
  const cur = lyrIdx(t);
  lyrShown = cur;
  let h = '';
  for (let k = cur - 3; k <= cur + 3; k++) {
    if (k < 0 || k >= L.length) { h += '<div class="ll"></div>'; continue; }
    const txt = L[k][1], on = k === cur;
    const fs = on && txt.length > 15 ? Math.max(20, Math.floor(470 / txt.length)) : 0;
    h += b('music.lyric.' + k, 'll' + (on ? ' on' : ''), esc(txt), () => seek(L[k][0]), '歌词第' + (k + 1) + '行', fs ? 'style="font-size:' + fs + 'px"' : '');
  }
  return h;
}
AU.addEventListener('timeupdate', () => {
  const t = TRACKS[S.m.i];
  if (!t.f || lyrIdx(t) === lyrShown) return;
  const p = document.getElementById('lyr'), c = document.getElementById('cline');
  if (p) p.innerHTML = lyrLines();
  if (c) c.innerHTML = lyrNow();
});
AU.addEventListener('ended', () => {
  if (S.m.mode === 1) { AU.currentTime = 0; AU.play(); return; }
  track(1);
  render();
});

// ---------------------------------------------------------------- status bar
function statusbar() {
  const d = new Date();
  const tell = (c, n) => '<span style="color:' + c + '">' + ic(n, 34) + '</span>';
  const call = S.phone.call && S.phone.hide ? b('sb.call', 'incall', ic('phone', 26) + '通话中', () => { S.phone.hide = false; }, '返回通话') : '';
  return '<div id="sb"><div class="sbl">' +
    (S.v.park ? tell('#f03a3a', 'park') : '') + (S.v.headlamp > 0 ? tell('#19b35a', 'lamp') : '') + tell('#f03a3a', 'belt') +
    (S.v.trunk ? tell('#ff8a1f', 'trunk') : '') + (S.v.lock ? '' : tell('#ff8a1f', 'unlock')) + call +
    '</div><div class="sbr">' + ic('mic', 30) + ic('pin', 30) +
    (S.v.bt_on === 0 ? '<span style="color:var(--blue)">' + ic('bt', 30) + '</span>' : '') +
    '<span class="sbt">5G</span>' + (S.v.wlan_on === 0 ? ic('wifi', 30) : '') +
    '<span class="sbt">24°C</span><span class="sbt" id="clock">' + pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + '</span></div></div>';
}

// ---------------------------------------------------------------- left panel
function carTop(w, h) {
  const mir = S.v.mirror
    ? '<path d="M31 112l-9 10v14l9-6z" fill="#23272e"/><path d="M169 112l9 10v14l-9-6z" fill="#23272e"/>'
    : '<path d="M31 110 8 119v12l23-5z" fill="#23272e"/><path d="M169 110l23 9v12l-23-5z" fill="#23272e"/>';
  return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 200 340">' +
    '<defs><linearGradient id="cb" x1="0" x2="1"><stop offset="0" stop-color="#14171c"/><stop offset=".5" stop-color="#3b4049"/><stop offset="1" stop-color="#14171c"/></linearGradient></defs>' +
    '<ellipse cx="100" cy="176" rx="88" ry="160" fill="rgba(20,30,50,.10)"/>' + mir +
    '<rect x="30" y="10" width="140" height="320" rx="48" fill="url(#cb)"/>' +
    '<path d="M46 98Q100 76 154 98l-8 42Q100 126 54 140z" fill="#0a0c10"/>' +
    '<rect x="52" y="146" width="96" height="116" rx="14" fill="#0f1217"/>' +
    '<path d="M54 270q46 12 92 0l6 28q-52 14-104 0z" fill="#0a0c10"/>' +
    '<path d="M54 19Q100 9 146 19" stroke="#e6eeff" stroke-width="3" fill="none" stroke-linecap="round"/>' +
    '<path d="M52 323q48 8 96 0" stroke="#ff4d4f" stroke-width="3" fill="none" stroke-linecap="round"/>' +
    (S.v.trunk ? '<path d="M40 306q60 44 120 0" stroke="#0a6bff" stroke-width="7" fill="none" stroke-linecap="round"/>' : '') +
    (S.v.chargecap ? '<path d="M172 252h26" stroke="#0a6bff" stroke-width="5" stroke-linecap="round"/>' : '') +
    '</svg>';
}
function cover(t, size) {
  if (t.cover) return '<div class="cover" style="background:url(music/' + t.f + '.jpg) center/cover"></div>';
  return '<div class="cover" style="background:linear-gradient(135deg,' + t.c[0] + ',' + t.c[1] + ')">' + ic('music', size) + '</div>';
}
function mediaCard() {
  const t = TRACKS[S.m.i];
  return '<div class="media"><div class="hd"></div>' + cover(t, 56) + '<div class="t1">' + t.t + '</div><div class="t2">' + t.a + '</div><div class="t3" id="cline">' + lyrNow() + '</div><div class="mctl">' +
    b('media.prev', 'mb', ic('prev', 50), () => track(-1), '上一首') +
    b('media.play', 'mb', ic(S.m.play ? 'pause' : 'play', 54), () => { S.m.play = !S.m.play; }, '播放暂停') +
    b('media.next', 'mb', ic('next', 50), () => track(1), '下一首') +
    b('media.fav', 'mb' + (S.m.fav[S.m.i] ? ' fav' : ''), ic('heart', 50), () => { S.m.fav[S.m.i] = !S.m.fav[S.m.i]; }, '收藏') +
    b('media.open', 'mb src', ic('music', 50), () => go('music'), '打开音乐') + '</div></div>';
}
function callStatus(c) {
  return (c.st === 'dial' ? '正在呼叫…' : '通话中 ' + mmss((Date.now() - c.t0) / 1000)) + (c.dtmf ? ' · ' + c.dtmf : '');
}
function callCard(c) {
  const keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '0', '#'];
  return '<section id="left"><div class="cnum">' + (c.name || c.num) + '</div><div class="cst" id="cst">' + callStatus(c) + '</div><div class="ckeys">' +
    keys.map(k => b('call.key.' + kid(k), 'ck', k, () => { c.dtmf = (c.dtmf || '') + k; }, '通话键盘' + k)).join('') + '</div>' +
    b('call.hangup', 'chang', ic('hang', 56), endCall, '挂断') +
    b('call.hide', 'chide', '隐藏', () => { S.phone.hide = true; }, '隐藏通话') + '</section>';
}
function startCall(num, name) {
  if (!num) { toast('请输入号码'); return; }
  const c = { num, name, st: 'dial', t0: Date.now() };
  S.phone.call = c; S.phone.hide = false;
  setTimeout(() => { if (S.phone.call === c && c.st === 'dial') { c.st = 'talk'; c.t0 = Date.now(); render(); } }, 2500);
}
function endCall() {
  const c = S.phone.call;
  if (!c) return;
  S.phone.recents.unshift([c.name || c.num, c.num, '刚刚', 'out']);
  S.phone.call = null; S.phone.hide = false; S.phone.num = '';
  toast('通话已结束');
}
function leftPanel() {
  const call = S.phone.call;
  if (call && !S.phone.hide) return callCard(call);
  const note = S.v.trunk ? '尾门已打开' : S.v.mirror ? '后视镜已折叠' : (S.v.lock ? '' : '车辆未上锁');
  return '<section id="left"><div class="gear">' + S.gear + '</div><div class="carwrap">' + carTop(206, 350) + '</div><div class="carnote">' + note + '</div>' +
    '<div class="quick">' +
    b('quick.trunk', 'qb' + (S.v.trunk ? ' act' : ''), ic('trunk', 54), () => tgl('trunk', '尾门已打开', '尾门已关闭'), '尾门') + '<i></i>' +
    b('quick.mirror', 'qb' + (S.v.mirror ? ' act' : ''), ic('mirror', 54), () => tgl('mirror', '后视镜已折叠', '后视镜已展开'), '后视镜折叠') + '<i></i>' +
    b('quick.park', 'qb' + (S.v.park ? ' act' : ''), ic('park', 54), () => tgl('park', '电子驻车已启用', '电子驻车已释放'), '电子驻车') +
    '</div>' + mediaCard() + '</section>';
}

// ---------------------------------------------------------------- dock
function dock() {
  const L = [['home', 'home', '桌面'], ['settings', 'car', '车辆设置'], ['nav', 'nav', '导航'], ['music', 'music', '音乐'], ['phone', 'phone', '电话'], ['apps', 'apps', '应用中心']];
  const dl = L.map(x => b('dock.' + x[0], 'di' + (S.page === x[0] || (x[0] === 'apps' && S.page === 'stub') ? ' on' : ''), ic(x[1], 50), () => go(x[0]), x[2])).join('');
  const fl = S.c.seat.fl;
  const lv = n => (n ? '<span class="lv"><b>' + n + '</b></span>' : '');
  const tt = tempTxt(S.c.tL);
  const dr =
    b('dock.lock', 'di' + (S.v.lock ? '' : ' hot'), ic(S.v.lock ? 'lock' : 'unlock', 48), () => tgl('lock', '车辆已上锁', '车辆已解锁'), '中控锁') +
    b('dock.temp', 'di dtemp', tt + (isNaN(+tt) ? '' : '<small>°</small>'), () => go('climate'), '主驾温度') +
    '<div class="dsep"></div>' +
    b('dock.climate', 'di' + (S.page === 'climate' ? ' on' : '') + (S.c.on ? ' act' : ''), ic('fan', 50), () => go(S.page === 'climate' ? S.prev : 'climate'), '空调') +
    b('dock.seatheat', 'di' + (fl[0] ? ' hot' : ''), ic('seatheat', 50) + lv(fl[0]), () => cyc(fl, 0), '主驾座椅加热') +
    b('dock.seatvent', 'di' + (fl[1] ? ' act' : ''), ic('seatvent', 50) + lv(fl[1]), () => cyc(fl, 1), '主驾座椅通风') +
    b('dock.wheel', 'di' + (S.v.wheel ? ' hot' : ''), ic('wheel', 50), () => tgl('wheel', '方向盘加热已开启', '方向盘加热已关闭'), '方向盘加热') +
    b('dock.rdef', 'di' + (S.v.rdef ? ' act' : ''), ic('rdef', 50), () => tgl('rdef', '后除霜已开启', '后除霜已关闭'), '后除霜') +
    b('dock.park', 'di' + (S.v.park ? ' act' : ''), ic('park', 50), () => tgl('park', '电子驻车已启用', '电子驻车已释放'), '电子驻车') +
    b('dock.winup', 'dtx', '一键升窗', () => { S.v.window = false; S.v.win_all = 0; toast('车窗已全部关闭'); }, '一键升窗') +
    b('dock.mirror', 'dtx' + (S.v.mirror ? ' act' : ''), '后视镜折叠', () => tgl('mirror', '后视镜已折叠', '后视镜已展开'), '后视镜折叠') +
    b('dock.nap', 'dtx' + (S.v.nap ? ' act' : ''), '小憩模式', () => tgl('nap', '小憩模式已开启', '小憩模式已关闭'), '小憩模式');
  return '<div id="dockL">' + dl + '</div><div id="dockR">' + dr + '</div>';
}

// ---------------------------------------------------------------- settings
function onSet(id) {
  if (id === 'window') S.v.win_all = S.v.window ? 1 : 0;
  if (id === 'win_all') S.v.window = S.v.win_all === 1;
  if (id === 'trunk') toast(S.v.trunk ? '尾门已打开' : '尾门已关闭');
  if (id === 'tail_open') S.v.trunk = true;
  if (id === 'lock') toast(S.v.lock ? '车辆已上锁' : '车辆已解锁');
  if (id === 'traffic') S.nav.traffic = S.v.traffic === 0;
}
function autoPill(key) {
  return b('set.' + key, 'pill' + (S.v[key] ? ' on' : ''), '自动', () => { S.v[key] = !S.v[key]; }, key);
}
function slider(it) {
  const v = S.v[it.id];
  return '<div class="sl" data-id="set.' + it.id + '" data-n="' + it.label + '" data-sl="' + it.id + '" data-u="' + (it.unit || '') + '"><div class="f" style="width:' + v + '%"></div><div class="i">' + ic(it.icon, 34) + '</div><div class="n">' + v + (it.unit || '') + '</div></div>';
}
function setRow(it) {
  const lab = '<div class="lab">' + it.label + '</div>';
  let h = '';
  if (it.k === 'seg') {
    h = lab + '<div class="ctl"><div class="seg">' +
      it.opts.map((o, i) => b('set.' + it.id + '.' + i, 'o' + (S.v[it.id] === i ? ' on' : ''), o, () => { S.v[it.id] = i; onSet(it.id); }, it.label + ':' + o)).join('') +
      '</div>' + (it.auto ? autoPill(it.auto) : '') + '</div>';
  } else if (it.k === 'sl') {
    h = lab + '<div class="ctl">' + slider(it) + (it.auto ? autoPill(it.auto) : '') + '</div>';
  } else if (it.k === 'btn') {
    h = lab + '<div class="ctl">' + b('set.' + it.id, 'pill', it.text, () => { toast(it.msg); onSet(it.id); }, it.label + ':' + it.text) + '</div>';
  } else if (it.k === 'info') {
    h = '<div class="inf"><span>' + it.label + '</span>' + it.val + '</div>';
  } else if (it.k === 'tiles') {
    h = '<div class="tiles">' + it.items.map(t => b('set.tile.' + t[0], 'tile' + (S.v[t[0]] ? ' on' : ''), ic(t[2], 52) + t[1], () => { S.v[t[0]] = !S.v[t[0]]; onSet(t[0]); }, t[1])).join('') + '</div>';
  } else if (it.k === 'colors') {
    h = lab + '<div class="ctl" style="gap:14px">' + AMB.map((c, i) => b('set.' + it.id + '.' + i, 'sw' + (S.v[it.id] === i ? ' on' : ''), '', () => { S.v[it.id] = i; }, it.label + ':' + (i + 1), 'style="background:' + c + '"')).join('') + '</div>';
  } else if (it.k === 'wall') {
    const W = [['默认', 'linear-gradient(115deg,#c5d8ef,#e6d9f1)'], ['暖阳', 'linear-gradient(115deg,#f3d9c9,#ead7ee)'], ['青山', 'linear-gradient(115deg,#c9ecdd,#cfd8f3)']];
    h = lab + '<div class="ctl">' + W.map((w, i) => b('set.wall.' + i, 'wallo' + (S.v.wall === i ? ' on' : ''), w[0], () => { S.v.wall = i; }, '壁纸:' + w[0], 'style="background:' + w[1] + '"')).join('') + '</div>';
  } else if (it.k === 'devs') {
    h = lab + it.items.map((d, i) => {
      const on = S.v[it.id] === i;
      return b('set.' + it.id + '.' + i, 'li' + (on ? ' on' : ''), d[0] + '<span>' + (on ? '已连接' : (d[1] === '已连接' ? '未连接' : d[1])) + '</span>', () => { S.v[it.id] = i; toast('已连接 ' + d[0]); }, it.label + ':' + d[0]);
    }).join('');
  }
  return '<div class="row">' + h + '</div>';
}
function settingsPage() {
  const cat = SET.find(c => c[0] === S.set.cat);
  const sub = cat[2].find(s => s[0] === S.set.sub[cat[0]]);
  const c1 = SET.map(c => b('set.cat.' + c[0], 'sitem' + (c === cat ? ' on' : ''), c[1], () => { S.set.cat = c[0]; }, c[1])).join('');
  const c2 = cat[2].map(s => b('set.sub.' + cat[0] + '.' + s[0], 'sitem' + (s === sub ? ' on' : ''), s[1], () => { S.set.sub[cat[0]] = s[0]; }, cat[1] + '/' + s[1])).join('');
  return '<div class="snav"><div class="scol">' + c1 + '</div><div class="scol">' + c2 + '</div></div>' +
    '<div class="scont scroll" data-scroll="set.' + cat[0] + '.' + sub[0] + '">' + sub[2].map(setRow).join('') + '</div>';
}

// ---------------------------------------------------------------- phone
function phonePage() {
  const p = S.phone;
  const tabs = [['recent', '最近通话'], ['contacts', '通讯录'], ['keypad', '拨号键盘'], ['bt', '蓝牙设置']];
  const nav = tabs.map(t => b('phone.tab.' + t[0], 'sitem' + (p.tab === t[0] ? ' on' : ''), t[1], () => { p.tab = t[0]; }, '电话/' + t[1])).join('');
  let c = '';
  if (p.tab === 'keypad') {
    const K = [['1', ''], ['2', 'ABC'], ['3', 'DEF'], ['4', 'GHI'], ['5', 'JKL'], ['6', 'MNO'], ['7', 'PQRS'], ['8', 'TUV'], ['9', 'WXYZ'], ['*', ''], ['0', '+'], ['#', '']];
    c = '<div class="pnum' + (p.num ? '' : ' ph') + '">' + (p.num || '请输入号码') + '</div><div class="kgrid">' +
      K.map(k => b('phone.key.' + kid(k[0]), 'k', k[0] + '<small>' + k[1] + '</small>', () => { if (p.num.length < 15) p.num += k[0]; }, '拨号' + k[0])).join('') +
      b('phone.call', 'k c call', ic('phone', 44), () => startCall(p.num, ''), '拨打') +
      b('phone.del', 'k c', '删除', () => { p.num = p.num.slice(0, -1); }, '删除') +
      b('phone.clear', 'k c', '清空', () => { p.num = ''; }, '清空') + '</div>';
  } else if (p.tab === 'contacts') {
    let last = '';
    c = '<div class="sbar">' + ic('search', 32) + '搜索联系人</div><div class="scroll" data-scroll="contacts" style="height:850px">' + CONTACTS.map((x, i) => {
      const sec = x[0] !== last ? '<div class="sec">' + x[0] + '</div>' : '';
      last = x[0];
      return sec + b('phone.contact.' + i, 'pli', '<div class="ava">' + x[1][0] + '</div><div class="nm">' + x[1] + '</div><div class="s">' + x[2] + '</div>', () => startCall(x[2], x[1]), '联系人:' + x[1]);
    }).join('') + '</div>';
  } else if (p.tab === 'recent') {
    c = '<div class="scroll" data-scroll="recents" style="height:940px">' + p.recents.slice(0, 12).map((x, i) =>
      b('phone.recent.' + i, 'pli' + (x[3] === 'miss' ? ' miss' : ''), '<div class="ava">' + ic(x[3] === 'out' ? 'up' : 'down', 30) + '</div><div><div class="nm">' + x[0] + '</div><div class="n2">' + x[1] + '</div></div><div class="s">' + x[2] + '</div>', () => startCall(x[1], x[0] === x[1] ? '' : x[0]), '最近通话:' + x[0])).join('') + '</div>';
  } else {
    c = '<div style="padding-top:20px">' + SET.find(s => s[0] === 'conn')[2][0][2].map(setRow).join('') + '</div>';
  }
  return '<div class="ptitle">电话</div><div class="pnav">' + nav + '</div><div class="pcont">' + c + '</div>';
}

// ---------------------------------------------------------------- music
function musicPage() {
  const m = S.m, t = TRACKS[m.i];
  const nav = ['推荐', '电台', '我的收藏', '本地音乐'].map((x, i) => b('music.tab.' + i, 'sitem' + (m.tab === i ? ' on' : ''), x, () => { m.tab = i; }, '音乐/' + x)).join('');
  const list = TRACKS.map((x, i) => [x, i]).filter(e => m.tab === 0 || (m.tab === 1 && e[0].g === 1) || (m.tab === 2 && m.fav[e[1]]) || (m.tab === 3 && e[0].g === 0));
  const rows = list.map((e, n) => b('music.track.' + e[1], 'tl' + (e[1] === m.i ? ' on' : ''), '<div class="ix">' + (e[1] === m.i && m.play ? ic('vol', 30) : n + 1) + '</div>' + e[0].t + '<div class="a">' + e[0].a + '</div>', () => { m.i = e[1]; m.pos = 0; m.play = true; }, '曲目:' + e[0].t)).join('');
  return '<div class="ptitle">音乐</div><div class="pnav">' + nav + '</div><div class="pcont">' +
    '<div class="np">' + cover(t, 110) + '<div><div class="t1">' + t.t + '</div><div class="t2">' + t.a + '</div>' +
    '<div class="prog"><div id="mbar" style="width:' + (m.pos / t.d * 100) + '%"></div></div><div class="ptime"><span id="mpos">' + mmss(m.pos) + '</span><span>' + mmss(t.d) + '</span></div></div></div>' +
    '<div class="pc">' + b('music.prev', 'pcb', ic('prev', 50), () => track(-1), '上一首') +
    b('music.play', 'pcb main', ic(m.play ? 'pause' : 'play', 52), () => { m.play = !m.play; }, '播放暂停') +
    b('music.next', 'pcb', ic('next', 50), () => track(1), '下一首') +
    b('music.fav', 'pcb' + (m.fav[m.i] ? ' fav' : ''), ic('heart', 48), () => { m.fav[m.i] = !m.fav[m.i]; }, '收藏') +
    b('music.mode', 'pcb tx', ['列表循环', '单曲循环', '随机播放'][m.mode], () => { m.mode = (m.mode + 1) % 3; }, '播放模式') +
    b('music.lyrbtn', 'pcb tx' + (m.lyr ? ' onb' : ''), '歌词', () => { m.lyr = !m.lyr; }, '歌词开关') + '</div>' +
    '<div class="mlow"><div class="scroll mlist' + (m.lyr ? ' narrow' : '') + '" data-scroll="tracks">' + (rows || '<div class="tl" style="color:var(--tx3)">暂无内容</div>') + '</div>' +
    (m.lyr ? '<div class="lyr" id="lyr">' + lyrLines() + '</div>' : '') + '</div></div>';
}

// ---------------------------------------------------------------- app centre
function openApp(a) {
  if (['nav', 'music', 'phone', 'settings', 'climate'].includes(a[0])) go(a[0]);
  else if (a[0] === 'btapp') { S.set.cat = 'conn'; S.set.sub.conn = 'bt'; go('settings'); }
  else { S.stub = a; go('stub'); }
}
function appsPage() {
  return '<div class="ptitle">应用中心</div><div class="agrid">' + APPS.map(a => b('apps.' + a[0], 'app', '<div class="ai" style="background:' + a[3] + '">' + ic(a[2], 72) + '</div>' + a[1], () => openApp(a), '应用:' + a[1])).join('') + '</div>';
}
function stubPage() {
  const a = S.stub || APPS[5];
  return b('stub.back', 'backb', ic('back', 36) + '返回', () => go('apps'), '返回') +
    '<div class="stub"><div class="ai" style="background:' + a[3] + '">' + ic(a[2], 100) + '</div><div class="nm">' + a[1] + '</div><div class="ds">演示界面 · 此应用仅用于显示</div></div>';
}

// ---------------------------------------------------------------- map
const DEST_XY = [[100, 700], [1090, 160], [330, 340], [860, 160], [1090, 340], [860, 700], [1330, 160], [860, 520]];
function mapSvg(w, h, mini) {
  const n = S.nav, z = mini ? 1 : [0.6, 0.8, 1, 1.3, 1.7][n.zoom];
  const vw = w / z, vh = h / z, cx = 616, cy = 520;
  const hs = [-300, -60, 160, 340, 700, 900, 1120, 1340], vs = [-360, -120, 100, 330, 860, 1090, 1330, 1580, 1800];
  let r = '<rect x="-700" y="-700" width="2800" height="2500" fill="var(--map)"/>' +
    '<path d="M-700 840C0 770 300 1010 700 810S1500 610 2100 770" stroke="var(--water)" stroke-width="64" fill="none"/>' +
    '<rect x="350" y="180" width="246" height="140" rx="22" fill="var(--green)"/><rect x="-100" y="180" width="180" height="140" rx="22" fill="var(--green)"/>' +
    '<rect x="1110" y="720" width="200" height="160" rx="22" fill="var(--green)"/>';
  hs.forEach(y => { r += '<path d="M-700 ' + y + 'H2100" stroke="var(--road)" stroke-width="14"/>'; });
  vs.forEach(x => { r += '<path d="M' + x + ' -700V1800" stroke="var(--road)" stroke-width="14"/>'; });
  r += '<path d="M-700 520H2100M616 -700V1800" stroke="var(--road2)" stroke-width="26"/>';
  if (n.traffic) {
    r += '<path d="M616 506V174" stroke="#2fbf5a" stroke-width="8"/><path d="M630 520H1076" stroke="#f04a3a" stroke-width="8"/>' +
      '<path d="M344 520H602" stroke="#f5b62a" stroke-width="8"/><path d="M616 534V886" stroke="#2fbf5a" stroke-width="8"/>';
  }
  const labels = [[473, 258, '中心公园'], [740, 258, '人民医院'], [975, 438, '购物中心'], [215, 438, '第一中学'], [975, 618, '体育中心'], [215, 618, '图书馆'],
    [740, 618, '科技园'], [473, 438, '市民广场'], [1210, 258, '高铁站'], [-10, 258, '湿地公园'], [1210, 500, '科技大道'], [260, 964, '滨河路'], [1210, 808, '体育公园']];
  labels.forEach(l => { r += '<text x="' + l[0] + '" y="' + l[1] + '" font-size="24" fill="var(--mtx)" text-anchor="middle">' + l[2] + '</text>'; });
  if (n.dest != null && n.st !== 'idle' && n.st !== 'search') {
    const d = DEST_XY[n.dest], path = 'M616 520' + (d[1] !== 520 ? 'V' + d[1] : '') + 'H' + d[0];
    r += '<path d="' + path + '" stroke="#fff" stroke-width="26" fill="none" stroke-linejoin="round" stroke-linecap="round"/>' +
      '<path d="' + path + '" stroke="#0a6bff" stroke-width="16" fill="none" stroke-linejoin="round" stroke-linecap="round"/>' +
      '<g transform="translate(' + d[0] + ',' + d[1] + ')"><path d="M0 0c-14-16-22-26-22-38a22 22 0 0 1 44 0c0 12-8 22-22 38z" fill="#f5323c" stroke="#fff" stroke-width="3"/><circle cy="-38" r="8" fill="#fff"/></g>';
  }
  r += '<g transform="translate(616,520)"><circle r="46" fill="rgba(10,107,255,.16)"/><path d="M0-26 19 22 0 11-19 22z" fill="#0a6bff" stroke="#fff" stroke-width="4" stroke-linejoin="round"/></g>';
  const tilt = !mini && n.view ? ' style="transform:perspective(1500px) rotateX(42deg) scale(1.6);transform-origin:50% 62%"' : '';
  return '<svg width="' + w + '" height="' + h + '" viewBox="' + (cx - vw / 2) + ' ' + (cy - vh / 2) + ' ' + vw + ' ' + vh + '"' + tilt + '>' + r + '</svg>';
}
function navPage() {
  const n = S.nav, dest = n.dest == null ? null : PLACES[n.dest];
  let ov = '';
  if (n.st === 'idle') {
    ov = b('nav.search', 'msearch fl', ic('search', 36) + '搜索目的地', () => { n.st = 'search'; }, '搜索目的地') +
      '<div class="mchips">' + [['home', '回家', 0], ['bldg', '去公司', 1], ['bolt', '充电站', 7]].map(c => b('nav.chip.' + c[2], 'mchip fl', ic(c[0], 30) + c[1], () => { n.dest = c[2]; n.st = 'route'; }, c[1])).join('') + '</div>' +
      '<div class="mroad">' + ic('pin', 28) + '科技大道</div>';
  } else if (n.st === 'search') {
    ov = '<div class="mcard fl"><div class="hdr">' + b('nav.search.back', 'x', ic('back', 40), () => { n.st = 'idle'; }, '返回') + '选择目的地</div>' +
      PLACES.slice(0, 7).map((p, i) => b('nav.place.' + i, 'pl', ic('pin', 36) + '<div>' + p[0] + '<div class="ad">' + p[1] + '</div></div><div class="ds">' + p[2] + ' 公里</div>', () => { n.dest = i; n.st = 'route'; }, '目的地:' + p[0])).join('') + '</div>';
  } else if (n.st === 'route') {
    ov = '<div class="mcard fl"><div class="hdr">' + ic('pin', 40) + dest[0] + '</div><div style="font-size:24px;color:var(--tx3)">' + dest[1] + '</div>' +
      '<div class="seg" style="margin-top:24px">' + ['推荐', '最快', '少收费'].map((o, i) => b('nav.opt.' + i, 'o' + (n.opt === i ? ' on' : ''), o, () => { n.opt = i; }, '路线:' + o)).join('') + '</div>' +
      '<div class="kv" style="font-size:28px"><span>全程 <b>' + dest[2] + '</b> 公里</span><span>约 <b>' + (+dest[3] + [0, -2, 4][n.opt]) + '</b> 分钟</span></div>' +
      b('nav.start', 'go', '开始导航', () => { n.st = 'guide'; toast('开始导航，前往' + dest[0]); }, '开始导航') +
      b('nav.cancel', 'cancel', '取消', () => { n.st = 'idle'; n.dest = null; }, '取消') + '</div>';
  } else {
    const mins = +dest[3] + [0, -2, 4][n.opt], a = new Date(Date.now() + mins * 60000);
    ov = '<div class="guide">' + ic('turn', 84) + '<div><div class="g1">300 米</div><div class="g2">右转进入 创新路</div></div></div>' +
      '<div class="eta fl"><span><b>' + dest[2] + '</b>公里　<b>' + mins + '</b>分钟　预计 <b>' + pad2(a.getHours()) + ':' + pad2(a.getMinutes()) + '</b> 到达 ' + dest[0] + '</span>' +
      b('nav.exit', 'exit', '退出导航', () => { n.st = 'idle'; n.dest = null; toast('导航已结束'); }, '退出导航') + '</div>';
  }
  const col = '<div class="mcol">' +
    b('nav.traffic', 'mbtn fl' + (n.traffic ? ' act' : ''), ic('layers', 36) + '路况', () => { n.traffic = !n.traffic; S.v.traffic = n.traffic ? 0 : 1; }, '路况') +
    b('nav.view', 'mbtn fl', '<b style="font-size:30px">' + (n.view ? '3D' : '2D') + '</b>', () => { n.view = 1 - n.view; }, '视角') +
    b('nav.locate', 'mbtn fl', ic('locate', 40), () => { n.zoom = 2; toast('已回到当前位置'); }, '定位') + '</div>' +
    '<div class="mzoom">' + b('nav.zoomin', 'mbtn fl', ic('plus', 40), () => { n.zoom = Math.min(4, n.zoom + 1); }, '放大') +
    b('nav.zoomout', 'mbtn fl', ic('minus', 40), () => { n.zoom = Math.max(0, n.zoom - 1); }, '缩小') + '</div>' +
    (n.st === 'guide' ? '' : '<div class="mscale">' + ['1 公里', '500 米', '200 米', '100 米', '50 米'][n.zoom] + '</div>');
  return '<div class="map">' + mapSvg(1232, 1000, false) + '</div>' + ov + col;
}

// ---------------------------------------------------------------- desktop
function homePage() {
  const scn = [['nap', '小憩模式', 'nap'], ['camp', '露营模式', 'leaf'], ['wash', '洗车模式', 'drop'], ['pet', '宠物模式', 'heart']];
  const on = k => (k === 'nap' ? S.v.nap : S.scn[k]);
  const tp = (x, y, v) => '<div class="tp" style="left:' + x + 'px;top:' + y + 'px">' + v + '<small>bar</small></div>';
  return '<div class="hgrid">' +
    b('home.map', 'hc wide', mapSvg(782, 470, true) + '<div class="msearch fl">' + ic('search', 36) + '搜索目的地</div><div class="mroad">' + ic('pin', 28) + '科技大道</div>', () => go('nav'), '导航卡片') +
    '<div class="hc"><h4>' + ic('cloud', 32) + '天气</h4><div class="big">24<small>°C 多云</small></div><div class="kv">最高 / 最低<b>27° / 18°</b></div><div class="kv">空气质量<b style="color:#17a34a">优 32</b></div><div class="kv">湿度<b>56%</b></div><div class="kv">风力<b>东南风 2 级</b></div></div>' +
    '<div class="hc"><h4>' + ic('bolt', 32) + '续航</h4><div class="big">1,120<small>km</small></div><div class="kv">电量 82%<b>纯电 172 km</b></div><div class="bar"><div style="width:82%;background:#17a34a"></div></div><div class="kv">油量 75%<b>增程 948 km</b></div><div class="bar"><div style="width:75%;background:#0a6bff"></div></div></div>' +
    '<div class="hc"><h4>' + ic('tire', 32) + '胎压</h4><div style="position:absolute;left:138px;top:120px">' + carTop(104, 177) + '</div>' + tp(34, 150, '2.5') + tp(256, 150, '2.5') + tp(34, 310, '2.4') + tp(256, 310, '2.5') + '<div class="kv" style="position:absolute;left:32px;right:32px;bottom:28px"><span>状态</span><b style="color:#17a34a">正常</b></div></div>' +
    '<div class="hc"><h4>' + ic('apps', 32) + '场景</h4><div class="scn">' + scn.map(s => b('home.scene.' + s[0], 'scb' + (on(s[0]) ? ' on' : ''), ic(s[2], 46) + s[1], () => {
      if (s[0] === 'nap') S.v.nap = !S.v.nap; else S.scn[s[0]] = !S.scn[s[0]];
      toast(s[1] + (on(s[0]) ? '已开启' : '已关闭'));
    }, s[1])).join('') + '</div></div></div>';
}

// ---------------------------------------------------------------- climate (uses the whole width)
function climatePage() {
  const c = S.c;
  const cell = (id, on, html, fn, name) => b('clim.' + id, 'ccell' + (on ? ' on' : ''), html, fn, name);
  const stp = (id, name, val, sub, up, dn) => '<div class="cstp">' + b('clim.' + id + '.up', 'cbtn', ic('up', 46), up, name + '加') +
    '<div class="cval">' + val + '<small>' + sub + '</small></div>' + b('clim.' + id + '.down', 'cbtn', ic('down', 46), dn, name + '减') + '</div>';
  const clampT = t => Math.max(16, Math.min(32, t));
  const fanTo = (k, d, a) => { c[k] = Math.max(1, Math.min(7, c[k] + d)); c[a] = false; };
  const setL = d => { c.tL = clampT(c.tL + d); if (c.sync) c.tR = c.tL; if (c.rsync) c.rt = c.tL; };
  const front = '<div class="cgrp" style="top:40px">' +
    '<div class="ccol">' + cell('power', c.on, ic('power', 44) + '空调', () => { c.on = !c.on; }, '前排空调开关') +
    cell('auto', c.auto && c.on, '<b style="font-size:30px">AUTO</b>自动', () => { c.auto = !c.auto; if (c.auto) c.on = true; }, '前排自动') + '</div>' +
    '<div class="ccol">' + cell('ac', c.ac, '<b style="font-size:30px">A/C</b>制冷', () => { c.ac = !c.ac; if (!c.ac) c.max = false; }, 'A/C 制冷') +
    cell('max', c.max, '<b style="font-size:26px">MAX A/C</b>最大制冷', () => { c.max = !c.max; if (c.max) { c.ac = true; c.on = true; c.fan = 7; c.tL = 16; if (c.sync) c.tR = 16; } }, '最大制冷') + '</div>' +
    stp('tl', '主驾温度', tempTxt(c.tL), '主驾', () => setL(0.5), () => setL(-0.5)) +
    stp('fan', '前排风量', c.fan, '风量', () => fanTo('fan', 1, 'auto'), () => fanTo('fan', -1, 'auto')) +
    '<div class="ccol cmode">' + [['face', '吹面'], ['facefoot', '吹面吹脚'], ['foot', '吹脚']].map((m, i) => cell('mode.' + i, c.mode === i, ic(m[0], 52), () => { c.mode = i; c.auto = false; }, '前排' + m[1])).join('') + '</div></div>';
  const sb = (id, on, icon, label, fn) => b('clim.' + id, 'csb' + (on ? ' on' : ''), ic(icon, 46) + label, fn, label);
  const strip = '<div class="cstrip">' + sb('fdef', c.fdef, 'fdef', '前除霜', () => { c.fdef = !c.fdef; }) + sb('rdef', S.v.rdef, 'rdef', '后除霜', () => { S.v.rdef = !S.v.rdef; }) +
    sb('rec', c.rec, 'recirc', '内循环', () => { c.rec = !c.rec; }) + sb('pure', c.pure, 'leaf', '空气净化', () => { c.pure = !c.pure; }) +
    sb('scent', c.scent, 'drop', '香氛', () => { c.scent = !c.scent; }) + sb('ion', c.ion, 'sun', '负离子', () => { c.ion = !c.ion; }) + '</div>';
  const rear = '<div class="cgrp" style="top:650px">' +
    '<div class="ccol">' + cell('rpower', c.ron, ic('power', 44) + '后排', () => { c.ron = !c.ron; }, '后排空调开关') +
    cell('rlock', c.rlock, ic('lock', 40) + '后排锁定', () => { c.rlock = !c.rlock; }, '后排锁定') + '</div>' +
    '<div class="ccol">' + cell('rauto', c.rauto, '<b style="font-size:26px">后排自动</b>AUTO', () => { c.rauto = !c.rauto; }, '后排自动') +
    cell('rsync', c.rsync, '<b style="font-size:26px">后排同步</b>SYNC', () => { c.rsync = !c.rsync; if (c.rsync) c.rt = c.tL; }, '后排同步') + '</div>' +
    stp('rt', '后排温度', tempTxt(c.rt), '后排', () => { c.rt = clampT(c.rt + 0.5); c.rsync = false; }, () => { c.rt = clampT(c.rt - 0.5); c.rsync = false; }) +
    stp('rfan', '后排风量', c.rfan, '风量', () => fanTo('rfan', 1, 'rauto'), () => fanTo('rfan', -1, 'rauto')) +
    '<div class="ccol cmode">' + [['face', '吹面'], ['foot', '吹脚']].map((m, i) => cell('rmode.' + i, c.rmode === i, ic(m[0], 52), () => { c.rmode = i; }, '后排' + m[1])).join('') + '</div></div>';
  // seat map: id, name, left, top, height, kind (3 = heat+vent+massage, 2 = heat+vent, 4 = heat+fold)
  const seatDefs = [['fl', '主驾', 40, 70, 330, 3], ['fr', '副驾', 310, 70, 330, 3], ['ml', '二排左', 40, 420, 240, 2], ['mr', '二排右', 310, 420, 240, 2], ['rl', '三排左', 40, 680, 240, 4], ['rr', '三排右', 310, 680, 240, 4]];
  const sbtn = (k, i, icon, cls, name) => {
    const a = c.seat[k];
    return b('clim.seat.' + k + '.' + ['heat', 'vent', 'massage'][i], 'sbn' + (a[i] ? ' ' + cls : ''), ic(icon, 44) + (a[i] ? '<span class="lv">' + a[i] + '</span>' : ''), () => cyc(a, i), name);
  };
  const seats = seatDefs.map(d => {
    const a = c.seat[d[0]];
    let bs = sbtn(d[0], 0, 'seatheat', 'h', d[1] + '座椅加热');
    if (d[5] !== 4) bs += sbtn(d[0], 1, 'seatvent', 'v', d[1] + '座椅通风');
    if (d[5] === 3) bs += sbtn(d[0], 2, 'massage', 'v', d[1] + '座椅按摩');
    if (d[5] === 4) bs += b('clim.seat.' + d[0] + '.fold', 'sbn' + (a[2] ? ' v' : ''), ic('fold', 44), () => { a[2] = a[2] ? 0 : 1; }, d[1] + '座椅放倒');
    return '<div class="seatc' + (a[0] ? ' warm' : a[1] ? ' cool' : '') + '" style="left:' + d[2] + 'px;top:' + d[3] + 'px;height:' + d[4] + 'px">' + d[1] + bs + '</div>';
  }).join('');
  const wheel = b('clim.wheel', 'sbn' + (S.v.wheel ? ' h' : ''), ic('wheel', 46), () => { S.v.wheel = !S.v.wheel; }, '方向盘加热', 'style="position:absolute;left:218px;top:26px"');
  const right = '<div class="cright">' + stp('tr', '副驾温度', tempTxt(c.tR), '副驾', () => { c.tR = clampT(c.tR + 0.5); c.sync = false; }, () => { c.tR = clampT(c.tR - 0.5); c.sync = false; }) + '</div>' +
    '<div class="pm">PM2.5<b>12 优</b>车内</div>' +
    '<div class="csync">' + cell('sync', c.sync, ic('sync', 46) + '温区同步', () => { c.sync = !c.sync; if (c.sync) c.tR = c.tL; }, '温区同步') + '</div>';
  return '<section id="full">' + front + strip + rear + '<div class="smap">' + wheel + seats + '</div>' + right + '</section>';
}

// ---------------------------------------------------------------- frame
function rightPage() {
  switch (S.page) {
    case 'settings': return settingsPage();
    case 'nav': return navPage();
    case 'music': return musicPage();
    case 'phone': return phonePage();
    case 'apps': return appsPage();
    case 'stub': return stubPage();
    default: return homePage();
  }
}
const dimOf = v => ((100 - v) / 100 * 0.65).toFixed(3);
function render() {
  const sc = {};
  stage.querySelectorAll('[data-scroll]').forEach(el => { sc[el.dataset.scroll] = el.scrollTop; });
  H = {};
  stage.className = (S.v.theme === 1 ? 'dark ' : '') + 'wall' + S.v.wall + (S.v.font === 1 ? ' bigf' : '');
  let h = statusbar();
  h += S.page === 'climate' ? climatePage() : leftPanel() + '<section id="right">' + rightPage() + '</section>';
  h += dock() + '<div id="dim" style="opacity:' + dimOf(S.v.bright) + '"></div>' + (S.toast ? '<div id="toast">' + S.toast + '</div>' : '');
  stage.innerHTML = h;
  stage.querySelectorAll('[data-scroll]').forEach(el => { if (sc[el.dataset.scroll]) el.scrollTop = sc[el.dataset.scroll]; });
  syncAudio();
  logScreen();
}

// Scale of the stage on the screen; pointer positions are reported in stage pixels (= tablet pixels at 1920x1200).
let scale = 1;
function fit() {
  scale = Math.min(innerWidth / 1920, innerHeight / 1200);
  stage.style.transform = 'scale(' + scale + ')';
  stage.style.left = ((innerWidth - 1920 * scale) / 2) + 'px';
  stage.style.top = ((innerHeight - 1200 * scale) / 2) + 'px';
}
function pt(e) {
  const r = stage.getBoundingClientRect();
  return { x: Math.round((e.clientX - r.left) / scale), y: Math.round((e.clientY - r.top) / scale) };
}
function controls() {
  const r = stage.getBoundingClientRect(), out = [];
  stage.querySelectorAll('[data-id]').forEach(el => {
    const q = el.getBoundingClientRect();
    const x = (q.left - r.left) / scale, y = (q.top - r.top) / scale, w = q.width / scale, h = q.height / scale;
    if (w < 2 || y + h < 64 || y > 1200) return;
    out.push([el.dataset.id, el.dataset.n || '', Math.round(x), Math.round(y), Math.round(w), Math.round(h)]);
  });
  return out;
}
function logScreen() {
  const out = controls(), total = Math.ceil(out.length / 16);
  for (let i = 0; i < total; i++) LOG({ ev: 'screen', page: S.page, part: i + 1, of: total, controls: out.slice(i * 16, i * 16 + 16) });
}

// A tap counts when the pointer goes up within 48 px of where it went down, however long it stayed down.
let down = null, slide = null;
function slideSet(e) {
  const r = slide.el.getBoundingClientRect();
  const v = Math.round(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * 100);
  slide.v = v;
  slide.el.querySelector('.f').style.width = v + '%';
  slide.el.querySelector('.n').textContent = v + slide.el.dataset.u;
  if (slide.key === 'bright') document.getElementById('dim').style.opacity = dimOf(v);
  if (slide.key === 'vol_media') AU.volume = v / 100;
}
stage.addEventListener('pointerdown', e => {
  const t = e.target.closest('[data-id]'), p = pt(e);
  LOG({ ev: 'down', x: p.x, y: p.y, id: t ? t.dataset.id : null });
  const sl = e.target.closest('.sl');
  if (sl) { slide = { el: sl, key: sl.dataset.sl }; slideSet(e); return; }
  if (!t) return;
  down = { id: t.dataset.id, x: p.x, y: p.y, el: t };
  t.classList.add('press');
});
stage.addEventListener('pointermove', e => { if (slide) slideSet(e); });
function release(e, cancelled) {
  const p = pt(e);
  LOG({ ev: cancelled ? 'cancel' : 'up', x: p.x, y: p.y });
  if (slide) {
    S.v[slide.key] = slide.v;
    LOG({ ev: 'act', id: 'set.' + slide.key, value: slide.v });
    slide = null;
    render();
    return;
  }
  if (!down) return;
  const d = down;
  down = null;
  d.el.classList.remove('press');
  if (cancelled || Math.hypot(p.x - d.x, p.y - d.y) > 48) return;
  const fn = H[d.id];
  if (!fn) return;
  fn();
  LOG({ ev: 'act', id: d.id, page: S.page });
  render();
}
stage.addEventListener('pointerup', e => release(e, false));
stage.addEventListener('pointercancel', e => release(e, true));
document.addEventListener('contextmenu', e => e.preventDefault());
window.addEventListener('resize', fit);

setInterval(() => {
  const d = new Date(), el = document.getElementById('clock');
  if (el) el.textContent = pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  const c = S.phone.call, ce = document.getElementById('cst');
  if (c && ce) ce.textContent = callStatus(c);
  if (S.m.play) {
    const t = TRACKS[S.m.i];
    if (t.f) S.m.pos = AU.currentTime; // a real song: the audio element keeps the time and moves on by itself
    else {
      S.m.pos += 1;
      if (S.m.pos >= t.d) { if (S.m.mode !== 1) track(1); else S.m.pos = 0; render(); return; }
    }
    const mb = document.getElementById('mbar'), mp = document.getElementById('mpos');
    if (mb) { mb.style.width = (S.m.pos / t.d * 100) + '%'; mp.textContent = mmss(S.m.pos); }
  }
}, 1000);

// Hooks for the Android shell and for a debugger attached to the WebView.
window.hmiBack = () => { if (S.page === 'home') return false; go('home'); render(); return true; };
window.hmiPause = () => { if (S.m.play) { S.m.play = false; render(); } };
window.hmi = { S, go, render, controls, state: () => JSON.stringify(S) };

fit();
render();
