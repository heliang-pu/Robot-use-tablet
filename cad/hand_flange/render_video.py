import os,sys,subprocess,numpy as np,trimesh,xml.etree.ElementTree as ET
os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW','vtkOSOpenGLRenderWindow')
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy,numpy_to_vtkIdTypeArray
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation as R
W,H,FPS=1280,720,25
OUT='/home/phl/workspace/tablet_tap/cad/hand_flange/out/'
B='/tmp/claude-1000/-home-phl-workspace-tablet-tap/f11248c1-b455-41b8-812f-077fd0309b66/scratchpad/telehand_x/telehand_lite/urdf/'
FONT=sys.argv[1]; TEST=len(sys.argv)>2
ren=vtk.vtkRenderer(); ren.SetBackground(.93,.94,.96); ren.SetBackground2(.80,.83,.88); ren.GradientBackgroundOn()
win=vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(W,H); win.SetMultiSamples(0); win.AddRenderer(ren); ren.SetUseFXAA(True)
def light(p,i):
    l=vtk.vtkLight(); l.SetLightTypeToCameraLight(); l.SetPosition(*p); l.SetFocalPoint(0,0,0); l.SetIntensity(i); ren.AddLight(l)
ren.AutomaticLightCreationOff(); light((-1,1.2,1.2),.75); light((1.5,.3,.8),.38); light((0,-1,.5),.18)
def actor(m,col,op=1.,smooth=35):
    pd=vtk.vtkPolyData(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.asarray(m.vertices,dtype=np.float32),deep=True))
    f=np.c_[np.full(len(m.faces),3),m.faces].astype(np.int64).ravel(); ca=vtk.vtkCellArray(); ca.SetCells(len(m.faces),numpy_to_vtkIdTypeArray(f,deep=True))
    pd.SetPoints(pts); pd.SetPolys(ca)
    n=vtk.vtkPolyDataNormals(); n.SetInputData(pd); n.SetFeatureAngle(smooth); n.SplittingOn(); n.Update()
    mp=vtk.vtkPolyDataMapper(); mp.SetInputData(n.GetOutput()); a=vtk.vtkActor(); a.SetMapper(mp)
    p=a.GetProperty(); p.SetColor(*col); p.SetOpacity(op); p.SetSpecular(.25); p.SetSpecularPower(25); p.SetAmbient(.12); ren.AddActor(a); return a
def cyl(r,z0,z1,c=(0,0),n=64):
    m=trimesh.creation.cylinder(r,z1-z0,sections=n); m.apply_translation([*c,(z0+z1)/2]); return m
def uni(*a): return trimesh.boolean.union(list(a),engine='manifold')
def dif(a,*b): return trimesh.boolean.difference([a,*b],engine='manifold')
def at(m,T): m=m.copy(); m.apply_transform(T); return m
pol=lambda r,al:(r*np.sin(np.radians(al)),r*np.cos(np.radians(al)))
hexs=lambda af,z0,z1:cyl(af/np.sqrt(3),z0,z1,n=6)

P=trimesh.load('piper_end_frame.stl'); P.apply_translation([0,0,-1]); P.merge_vertices(); A_arm=actor(P,(.25,.26,.29))
rg=trimesh.load('ring.stl'); rg.apply_transform(trimesh.transformations.rotation_matrix(np.radians(45),[0,0,1])); A_ring=actor(rg,(.66,.68,.72))
fl=trimesh.load(OUT+'hand_flange.stl'); A_fl=actor(fl,(.25,.50,.85))
cone=lambda: None
def csk(L):
    c=trimesh.creation.revolve([[0,-1.65],[1.45,-1.65],[3,-.1],[3,0],[0,0]],sections=64); c.merge_vertices(); c.fix_normals()
    return dif(uni(cyl(1.45,-L,-.5),c),hexs(2.,-1.2,1))
def rotx(al):  # tool axis +Z -> radial direction al
    d=np.array([np.sin(np.radians(al)),np.cos(np.radians(al)),0]); R0=R.align_vectors([d,[0,0,1]],[[0,0,1],[1,0,0]])[0]; T=np.eye(4); T[:3,:3]=R0.as_matrix(); return d,T
key=hexs(2.,-1.2,70); key25=hexs(2.5,1.5,70); key15=hexs(1.5,0,70)
Ars=[]
for al in (45,135,225,315):
    d,T=rotx(al); Ars.append((actor(at(csk(14),T),(.90,.72,.25)),d,d*28.42+[0,0,7.05],actor(at(key,T),(.35,.36,.40))))
cap=dif(uni(cyl(1.45,-25,.2),cyl(2.75,0,3)),hexs(2.5,1.4,4)); wash=dif(cyl(3,-1.4,0),cyl(1.6,-2,1)); Ahs=[]
for al in (90,270):
    d,T=rotx(al); Ahs.append((actor(at(uni(cap,wash),T),(.90,.72,.25)),d,d*35.7+[0,0,25.9],actor(at(key25,T),(.35,.36,.40))))
# set screws (side B): axis from the mesh of the part is drawn by the report's picture: tips on the spigot
gs=dif(cyl(1.45,0,8),hexs(1.5,6.6,9)); Ags=[]
for al in (250,290):
    d=np.array([np.sin(np.radians(al)),np.cos(np.radians(al)),0]); R0=R.align_vectors([d,[0,0,1]],[[0,0,1],[1,0,0]])[0]; T=np.eye(4); T[:3,:3]=R0.as_matrix(); Ags.append((actor(at(gs,T),(.80,.35,.25)),d,d*13.02+[0,0,25.9],actor(at(key15,T),(.35,.36,.40))))
def tube(P,r):
    sp=vtk.vtkParametricSpline(); p=vtk.vtkPoints(); [p.InsertNextPoint(*q) for q in P]; sp.SetPoints(p); src=vtk.vtkParametricFunctionSource(); src.SetParametricFunction(sp); src.SetUResolution(90)
    t=vtk.vtkTubeFilter(); t.SetInputConnection(src.GetOutputPort()); t.SetRadius(r); t.SetNumberOfSides(20); t.CappingOn(); m=vtk.vtkPolyDataMapper(); m.SetInputConnection(t.GetOutputPort()); a=vtk.vtkActor(); a.SetMapper(m); ren.AddActor(a); return a
Acab=[]
for i,c in enumerate([(.8,.1,.1),(.8,.1,.1),(.1,.1,.1),(.1,.1,.1),(.95,.8,.1),(.1,.55,.2)]):
    th=2*np.pi*i/6; dx,dz=2.0*np.cos(th),1.9*np.sin(th)
    a=tube([(dx,dz*.6-1,31),(dx,dz*.6,25),(dx,2+dz*.4,20),(dx,8,16.6+dz*.9),(dx,15,15.4+dz),(dx,24,15.2+dz),(dx,32,15+dz),(dx,42,12+dz),(dx,54,-6+dz)],.95); a.GetProperty().SetColor(*c); Acab.append(a)
# hand in pointing pose
t=ET.parse(B+'telehand_lite.urdf').getroot(); J={}
for j in t.findall('joint'):
    o=j.find('origin'); v=lambda s:np.array([float(x) for x in s.split()]); T=np.eye(4); T[:3,3]=v(o.get('xyz')); T[:3,:3]=R.from_euler('xyz',v(o.get('rpy'))).as_matrix()
    J[j.find('child').get('link')]=(j.get('name'),j.find('parent').get('link'),T,v(j.find('axis').get('xyz')))
q={}
for f in ('middle','ring','pinky'):
    q.update({f+'_mcp':1.30,f+'_pip':1.30,f+'_dip':1.30})
def fk(l):
    if l not in J: return np.eye(4)
    n,p,T,a=J[l]; M=np.eye(4); M[:3,:3]=R.from_rotvec(a*q.get(n,0)).as_matrix(); return fk(p)@T@M
PH=np.array([[0,0,1000,-10.02],[1000,0,0,-1.02],[0,1000,0,81.02],[0,0,0,1.]])
Ahand=[]
for l in t.findall('link'):
    f=l.find('visual/geometry/mesh')
    if f is None: continue
    m=trimesh.load(B+f.get('filename').replace('./','')); m.apply_transform(PH@fk(l.get('name'))); m.merge_vertices()
    if 'index_dip'==l.get('name'): print('index tip (flange frame)',m.vertices[m.vertices[:,2].argmax()].round(1))
    Ahand.append(actor(m,(.36,.38,.42) if l.get('name')!='root' else (.44,.46,.50)))
F1=ImageFont.truetype(FONT,34); F2=ImageFont.truetype(FONT,22)
cam=ren.GetActiveCamera()
def view(az,el,dist,foc,fov=28):
    a,e=np.radians(az),np.radians(el); cam.SetFocalPoint(*foc); cam.SetPosition(*(np.array(foc)+dist*np.array([np.cos(e)*np.sin(a),np.cos(e)*np.cos(a),np.sin(e)]))); cam.SetViewUp(0,0,1); cam.SetViewAngle(fov); ren.ResetCameraClippingRange()
w2i=vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.ReadFrontBufferOff(); w2i.SetInputBufferTypeToRGB()
ff=None if TEST else subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-movflags','+faststart',OUT+'hand_flange_v2.mp4'],stdin=subprocess.PIPE)
N=[0]
def shot(title,sub,step):
    win.Render(); w2i.Modified(); w2i.Update(); o=w2i.GetOutput(); d=o.GetDimensions(); im=vtk_to_numpy(o.GetPointData().GetScalars()).reshape(d[1],d[0],3)[::-1]
    I=Image.fromarray(im.copy()); D=ImageDraw.Draw(I,'RGBA'); D.rectangle((0,H-104,W,H),fill=(20,26,36,205)); D.text((36,H-94),title,font=F1,fill='white'); D.text((36,H-44),sub,font=F2,fill=(205,215,230))
    if TEST: I.save(f'vt_{N[0]:02d}.png')
    else: ff.stdin.write(np.asarray(I).tobytes())
    N[0]+=1
ease=lambda x:(lambda x:x*x*(3-2*x))(min(max(x,0),1))
def seg(sec,title,sub,fn):
    n=int(sec*FPS)
    for i in ([int(n*.4),n-1] if TEST else range(n)): fn(i/max(n-1,1)); shot(title,sub,i)
def show(L,on=True): [a.SetVisibility(on) for a in L]

ars=[x[0] for x in Ars]; ark=[x[3] for x in Ars]; ahs=[x[0] for x in Ahs]; ahk=[x[3] for x in Ahs]; ags=[x[0] for x in Ags]; agk=[x[3] for x in Ags]
everything=[A_arm,A_ring,A_fl,*ars,*ark,*ahs,*ahk,*ags,*agk,*Acab,*Ahand]
def only(*L): show(everything,False); [show(x) for x in L]
def place(S,j,u,d0=40,base=0):  # put screw j of set S at progress u
    a,d,p,k=S[j]; e=d0*(1-ease(u*1.35)); q=p+d*e+[0,0,base]; a.SetPosition(*q); k.SetPosition(*q); a.SetVisibility(u>0); k.SetVisibility(0<u<.96)
def lift(z):
    A_fl.SetPosition(0,0,z); [a.SetPosition(0,0,z) for a in Ahand+Acab]
    for S in (Ahs,Ags):
        for a,d,p,k in S: a.SetPosition(*(p+[0,0,z]))
only([A_fl])
seg(5,'转接法兰第二版','一体铝件 58 g，装在松灵自带的法兰环上；换工具只动环上 4 颗螺钉',lambda s:view(25+360*s,26,185,(0,0,17)))
seg(3.5,'从手臂一侧看','Φ40 止口插进松灵的环，四个平面避开环的钥匙面（每 90° 一档）',lambda s:view(25+110*s,-36,185,(0,0,17)))
def g1(s):
    z=80*(1-ease(s*1.3)); [a.SetPosition(0,0,z) for a in Ahand]; view(58-28*s,16,430,(0,0,78)); show(Acab,s>.55)
only([A_fl],Ahand)
seg(4.5,'① 先在桌上：手的圆管插进新件','线从出线口穿出；手底面压在两块凸台上（走线为示意）',g1)
only([A_fl],Ahand,Acab)
def g2(s): view(62,10,200,(0,0,27)); place(Ahs,0,s,45)
seg(3,'② A 侧：M3×25 加 3 片碟簧、垫圈','把圆管拉到 A 侧孔壁，这是基准；约 0.2 N·m，碟簧将平未平',g2)
def g3(s): view(-62,14,200,(0,0,27)); place(Ags,0,min(s*2,1)); place(Ags,1,max(s*2-1,0))
seg(4,'③ B 侧：两颗顶丝拧到刚碰上就停','消除间隙，不要用力挤圆管',g3)
def g4(s): view(-62,10,200,(0,0,27)); place(Ahs,1,s,45)
seg(3,'④ B 侧：M3×25 同样拧紧','厚壁夹在螺钉和两颗顶丝之间，1 mm 薄壁不受力',g4)
only([A_arm,A_ring],[A_fl],Ahand,Acab,ahs,ags)
def g5(s): lift(50*(1-ease(s*1.25))); view(40+14*s,12,330,(0,0,40))
seg(4,'⑤ 整体装到松灵的环上','松灵的环留在臂上不动；除了插进环里的止口，新件都在环顶面以上，碰不到 J6 外壳',g5)
lift(0)
def g6(s):
    view(60+330*s,4+10*np.sin(np.pi*s),235,(0,0,14))
    for j in range(4): place(Ars,j,min(max(s*4-j,0),1),40)
only([A_arm,A_ring],[A_fl],Ahand,Acab,ahs,ags)
seg(8,'⑥ 4 颗 M3×14 沉头从环外侧拧入','0.8 N·m，对角分两次；钉头把肩面压紧在环顶面上',g6)
[show(ark,False),show(ars)]
def g7(s):
    [a.GetProperty().SetOpacity(.28) for a in (A_fl,)]; [a.GetProperty().SetOpacity(.3) for a in Ahand+[A_ring]]; view(62-110*s,9,190,(0,0,19))
seg(5,'透视：螺钉、顶丝和走线','黄：螺钉　红：顶丝　彩线：4 根电源 + 2 根 485',g7)
[a.GetProperty().SetOpacity(1) for a in (A_fl,A_ring,*Ahand)]
seg(6,'装好后：食指伸直点屏幕','内盘端面到食指尖 208.7 mm；手、新件、环和螺钉共 513 g（拇指姿势为示意）',lambda s:view(35+360*s,14,580,(0,-8,86)))
if ff: ff.stdin.close(); ff.wait()
print('frames',N[0])
