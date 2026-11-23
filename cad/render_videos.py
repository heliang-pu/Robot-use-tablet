"""Render the v3 turntables. Run with python3 -B render_videos.py.
The controller runs Blender CPU workers sequentially with a 300 s timeout.
An interrupted run reuses completed PNGs on the next invocation.
"""
from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'out' / 'videos'
WORK = OUT / '.work'
BLEND = ROOT / 'out' / 'tablet_tap_parts.blend'
BLENDER = '/snap/blender/current/blender'
FPS, TOTAL, THREADS = 24, 240, 12
W, H = 1280, 720
MOVIES = ('pen_holder', 'tablet_stand')
PAUSE = 36  # within the 144-frame assembled shot
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def ease(x):
    t = max(0., min(1., x))
    return t*t*(3-2*t)


def pose(movie, frame):
    """Azimuth, elevation, projected roll, axial displacement, stand gap, tablet opacity."""
    if movie == 'pen_holder':
        if frame < 96:
            return (26+360*frame/96, 23, 64, 0, 118, 0)
        j = frame-96
        rotation = min(j, PAUSE)+max(0, j-PAUSE-48)
        angle = 9+360*(rotation-PAUSE)/96
        travel = 2*(1-math.cos(2*math.pi*(j-PAUSE)/24)) if PAUSE<=j<PAUSE+48 else 0
        return (angle, 12, 75, travel, 118, 0)
    if frame < 120:
        return (-59+360*frame/120, 28, 0, 0, 118, 1)
    j = frame-120
    if j <= 34:
        gap=118-28*ease((j-8)/26)
    elif j <= 74:
        gap=90+60*ease((j-34)/40)
    else:
        gap=150-32*ease((j-74)/24)
    alpha=1-ease(j/7) if j<8 else ease((j-99)/17)
    return (-59, 28, 0, 0, gap, alpha)


def flexure(holder):
    """Add longitudinal sections to the in-memory mesh. Rest shape is unchanged."""
    import bmesh
    from mathutils import Vector
    holder.data=holder.data.copy()
    bm=bmesh.new()
    bm.from_mesh(holder.data)
    volume=bm.calc_volume()
    for k in range(1,25):
        plane_x=-48+37*k/25
        geom=list(bm.verts)+list(bm.edges)+list(bm.faces)
        bmesh.ops.bisect_plane(bm,geom=geom,plane_co=(plane_x,0,0),plane_no=(1,0,0),
                               dist=1e-6,clear_outer=False,clear_inner=False)
    assert abs(bm.calc_volume()-volume)<.02 and all(e.is_manifold for e in bm.edges)
    bm.to_mesh(holder.data)
    bm.free()
    holder.shape_key_add(name='Unloaded')
    key=holder.shape_key_add(name='Guided stroke 4 mm')
    for vertex in key.data:
        x,y,_=vertex.co
        # Local print Y = -Z_H. The crossarm and jaw block end at Z_H=3.5.
        if -y>6:
            vertex.co+=Vector((0,4*ease((x+48)/37),0))
    return key


def render_worker(movie, frames, samples, probe=False):
    import bpy
    import numpy as np
    from mathutils import Matrix,Vector
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    sc=bpy.context.scene
    sc.render.engine='CYCLES'
    sc.cycles.device='CPU'
    sc.cycles.samples=samples
    sc.cycles.use_denoising=True
    sc.cycles.denoiser='OPENIMAGEDENOISE'
    if hasattr(sc.cycles,'denoising_use_gpu'):
        sc.cycles.denoising_use_gpu=False
    sc.cycles.max_bounces=5
    sc.cycles.transparent_max_bounces=8
    sc.render.threads_mode='FIXED'
    sc.render.threads=THREADS
    sc.render.use_persistent_data=True
    sc.render.resolution_x=W
    sc.render.resolution_y=H
    sc.render.resolution_percentage=100
    sc.render.image_settings.file_format='PNG'
    sc.render.image_settings.color_mode='RGB'
    sc.render.image_settings.compression=12
    sc.render.fps=FPS
    obj=bpy.data.objects
    cam=sc.camera
    lights=[o for o in sc.objects if o.type=='LIGHT']
    lights=sorted(lights,key=lambda o: ['Key','Fill','Rim'].index(o.name.split()[0]))
    for o in sc.objects:
        o.hide_render=o not in lights and o!=cam
    pen=movie==MOVIES[0]
    if pen:
        bare=obj['pen_holder']; holder=obj['Holder • installed']
        mobile=[o for o in sc.objects if o.name.startswith(('Stylus •','Cable tie'))]
        fingers=[o for o in sc.objects if o.name.startswith(('Finger A','Finger B'))]
        assembled=[holder,*fingers,*mobile]
        moving_origins={o:o.location.copy() for o in mobile}
        key=flexure(holder)
        groups=([bare],assembled)
    else:
        left=obj['Left saddle • assembled']; right=obj['Right saddle • assembled']
        rods=[o for o in sc.objects if o.name.startswith('Crossbar • installed')]
        shell=obj['Tablet • translucent viewing shell']; glass=obj['Tablet • glass']
        glass_base=glass.data.materials[0].node_tree.nodes['Principled BSDF']
        transparent_glass=shell.data.materials[0].copy()
        bsdf=transparent_glass.node_tree.nodes['Principled BSDF']
        for field in ('Base Color','Metallic','Roughness'):
            bsdf.inputs[field].default_value=glass_base.inputs[field].default_value
        glass.data.materials[0]=transparent_glass
        shell.hide_set(False)
        ghost=[(o.data.materials[0].node_tree.nodes.get('Mix Shader').inputs[0],opacity)
               for o,opacity in ((shell,.31),(glass,.21))]
        groups=([left,right,*rods,shell,glass],)*2
        obj['Backdrop'].hide_render=False
        obj['Backdrop'].location.z=-.05
    bpy.context.view_layer.update()

    def axes(f):
        a,e,roll,*_=pose(movie,f)
        a,e,roll=map(math.radians,(a,e,roll))
        if pen:
            eye=Vector((math.sin(a)*math.cos(e),-math.sin(e),math.cos(a)*math.cos(e)))
            up=Vector((0,-1,0))
        else:
            eye=Vector((math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)))
            up=Vector((0,0,1))
        r=up.cross(eye).normalized(); v=eye.cross(r)
        return math.cos(roll)*r+math.sin(roll)*v,-math.sin(roll)*r+math.cos(roll)*v,eye

    # Fit the complete motion before rendering. Use constant magnification per shot
    # and one fixed camera position throughout the width adjustment.
    def positions(o):
        co=np.empty(len(o.data.vertices)*3,dtype=np.float32)
        o.data.vertices.foreach_get('co',co)
        m=np.array(o.matrix_world)
        return co.reshape(-1,3)@m[:3,:3].T+m[:3,3]

    vertices={o:positions(o) for group in groups for o in group}
    boundary=96 if pen else 120
    camera_track={}
    for i,(lo,hi) in enumerate(((0,boundary),(boundary,TOTAL))):
        group=groups[i]
        points=np.concatenate([vertices[o] for o in group])
        if i and not pen:
            points=np.concatenate([points,vertices[right]+[-28,0,0],vertices[right]+[32,0,0]])
        if i and pen:
            points=np.concatenate([points,*[vertices[o]+[0,4,0] for o in mobile]])
        scales=[]
        for f in range(lo,hi):
            r,v,eye=axes(f)
            basis=np.array([r,v,eye])
            p=points@basis.T
            low,high=p.min(axis=0),p.max(axis=0)
            width,height,_=high-low
            scales.append(float(max(width/.90,height*W/H/.85)))
            camera_track[f]=(r,v,eye,Vector(((low+high)/2)@basis))
        for f in range(lo,hi):
            # A short look-ahead smooths silhouette changes while keeping the assembly
            # large in the wide frame. The fixed rail shot uses the whole motion envelope.
            j=f-lo
            size=max(scales[max(0,j-2):min(len(scales),j+3)])
            camera_track[f]+= (size,)
    selected=None
    directory=OUT/('frames_'+movie)
    directory.mkdir(exist_ok=True)
    for f in frames:
        path=(WORK/'probe.png') if probe else directory/f'{f:04}.png'
        if not probe and path.is_file(): continue
        index=int(f>=boundary)
        if index!=selected:
            for o in set(groups[0]+groups[1]): o.hide_render=o not in groups[index]
            selected=index
        _,_,_,travel,gap,alpha=pose(movie,f)
        if pen:
            key.value=travel/4
            for o,start in moving_origins.items(): o.location=start+Vector((0,travel,0))
        else:
            right.location.x=70+gap
            for socket,opacity in ghost: socket.default_value=opacity*alpha
            shell.hide_render=glass.hide_render=alpha<1e-7
        r,v,eye,center,scale=camera_track[f]
        target=center-v*(.035*scale*H/W)
        cam.matrix_world=Matrix((r,v,eye)).transposed().to_4x4()
        cam.location=target+eye*scale*3
        cam.data.type='ORTHO';cam.data.ortho_scale=scale;cam.data.clip_end=5000
        # Elevate the studio lights in the stand's world frame for a readable contact shadow.
        lr=r if pen else Vector((r.x,r.y,0)).normalized()
        lv=v if pen else Vector((0,0,1))
        le=eye if pen else lr.cross(lv).normalized()
        for lamp,(x,y,z),diam,power in zip(lights,[(-.72,1,.9),(.86,.55,.5),(.55,1.30,-.85)],
                                          [1,.85,.8],[19,6,13]):
            offset=scale*(x*lr+y*lv+z*le)
            lamp.location=center+offset
            lamp.rotation_euler=(-offset).to_track_quat('-Z','Y').to_euler()
            lamp.data.size=scale*diam
            lamp.data.energy=power*scale**2
        partial=path.with_suffix('.render.png')
        sc.render.filepath=str(partial)
        start=time.monotonic()
        bpy.ops.render.render(write_still=True)
        elapsed=time.monotonic()-start
        partial.replace(path)
        print(f'FRAME {movie} {f} {elapsed:.4f}',flush=True)
        if not probe:
            with (WORK/(movie+'.times')).open('a') as out: print(f,elapsed,file=out)


def invoke(movie, frame_spec, samples, probe=False):
    env=os.environ.copy()
    for key,sub in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('BLENDER_USER_CONFIG','config')]:
        folder=WORK/sub;folder.mkdir(exist_ok=True)
        env[key]=str(folder)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='12',MKL_NUM_THREADS='1')
    args=['timeout','300',BLENDER,'-b','-noaudio','--factory-startup','--python',str(Path(__file__).resolve()),
          '--','--worker',movie,frame_spec,str(samples)]
    if probe: args.append('--probe')
    for _ in range(3):
        result=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(result.stdout,flush=True)
        if result.returncode==0 and 'WORKER COMPLETE' in result.stdout:
            return [float(line.split()[-1]) for line in result.stdout.splitlines() if line.startswith('FRAME ')]
        if result.returncode not in (-15,124,143):
            break
    raise RuntimeError(f'Blender worker failed: {result.returncode}')


def captions(movie):
    if movie==MOVIES[0]:
        clips=[(0,95,'电容笔夹（打印件）'),(96,95+PAUSE,'装配：夹爪手指 + 电容笔'),
               (96+PAUSE,143+PAUSE,'弹性行程 4 mm'),(144+PAUSE,239,'装配：夹爪手指 + 电容笔')]
    else: clips=[(0,119,'平板支架 12°'),(120,239,'托座间距可调')]
    return ','.join(
        f"drawtext=fontfile='{FONT}':text='{text}':fontsize=26:fontcolor=0x203239:"
        f"box=1:boxcolor=white@0.96:boxborderw=10:x=42:y=h-60:enable='between(n,{a},{b})'"
        for a,b,text in clips)


def encode(movie):
    directory=OUT/('frames_'+movie)
    assert all((directory/f'{f:04}.png').is_file() for f in range(TOTAL))
    path=WORK/(movie+'.mp4')
    base=['/usr/bin/ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-filter_threads','2']
    subprocess.run(base+['-framerate','24','-threads','2','-i',str(directory/'%04d.png'),
                         '-vf',captions(movie),'-frames:v',str(TOTAL),'-c:v','libx264',
                         '-threads','6','-preset','medium','-crf','18','-pix_fmt','yuv420p',
                         '-movflags','+faststart',str(path)],check=True)
    info=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,nb_frames,avg_frame_rate,pix_fmt,duration',
            '-of','json',str(path)]))['streams'][0]
    assert (info['codec_name'],info['width'],info['height'],info['nb_frames'],info['avg_frame_rate'],
            info['pix_fmt'],float(info['duration']))==('h264',W,H,'240','24/1','yuv420p',10.)
    poster=WORK/(movie+'_poster.png')
    index=120 if movie==MOVIES[0] else 0
    subprocess.run(base+['-i',str(path),'-vf',f'select=eq(n\\,{index})','-frames:v','1',
                         '-threads','2',str(poster)],check=True)
    path.replace(OUT/path.name); poster.replace(OUT/poster.name)
    shutil.rmtree(directory)


def controller():
    WORK.mkdir(parents=True,exist_ok=True)
    if not Path(FONT).is_file(): raise FileNotFoundError(FONT)
    # Benchmark a full-resolution transparent assembly before selecting final samples.
    benchmark=invoke('tablet_stand','0:1',6,True)[0]
    samples=max(1,min(10,int(6*2.65/max(.1,benchmark))))
    print(f'BENCHMARK 6 spp: {benchmark:.3f} s; production: {samples} spp',flush=True)
    # Render representative poses early for visual review; the subsequent ranges reuse them.
    for movie,spec in zip(MOVIES,['0,48,96,144,194','0,60,154,194,239']):
        invoke(movie,spec,samples)
    for movie in MOVIES:
        for start in range(0,TOTAL,8):
            invoke(movie,f'{start}:{min(start+8,TOTAL)}',samples)
        encode(movie)
        rows=[line.split() for line in (WORK/(movie+'.times')).read_text().splitlines()]
        times={int(frame):float(seconds) for frame,seconds in rows}.values()
        print(f'FINISHED {movie}: mean {sum(times)/len(times):.3f} s, max {max(times):.3f} s',flush=True)

    for name in ('tmp','cache','config'):
        shutil.rmtree(WORK/name,ignore_errors=True)
    for name in ('probe.png',*(movie+'.times' for movie in MOVIES)):
        (WORK/name).unlink(missing_ok=True)
    if not any(WORK.iterdir()): WORK.rmdir()


if __name__=='__main__':
    if '--worker' in sys.argv:
        movie,spec,spp=sys.argv[sys.argv.index('--worker')+1:][:3]
        frames=range(*map(int,spec.split(':'))) if ':' in spec else list(map(int,spec.split(',')))
        try:
            render_worker(movie,frames,int(spp),'--probe' in sys.argv)
        except Exception:
            import traceback
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        print('WORKER COMPLETE',flush=True)
    elif '--encode' in sys.argv:
        encode(sys.argv[sys.argv.index('--encode')+1])
    else:
        controller()
import os, sys; sys.stdout.flush(); os._exit(0)
