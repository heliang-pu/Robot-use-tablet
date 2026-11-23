"""Blender 5.3，毫米。默认重建模型、四个 STL 和八张预览；--model 仅重建几何。"""
from pathlib import Path
from math import asin, sin, cos, tan, radians, pi, sqrt
import sys
import bpy
import bmesh
from mathutils import Vector, Matrix, geometry

P = {
    # H 系。范围用两端坐标表示；重复尺寸从基准推导。
    "block_x": (-19., 19.), "block_y": (-13., 13.), "block_z": (-27., 3.5),  # 夹持块外界
    "tunnel_x": (-6.5,16.5), "tunnel_y": (-6.5,8.0), "tunnel_chamfer": 4.5,  # 封闭隧道四角 45° 倒角
    "tunnel_min_clearance": .9, "contact_tol": .02,  # Ø10 就座后净空下限、Ø8.5 接触公差
    "finger_tip": 4.97, "finger_root": -71.5, "finger_flat_from": -5.5,  # 输入件的指尖、指根、平面起点
    "finger_retreat_deg": 4.9, "finger_opening": (0., 100.),  # 指腹退让角与夹爪额定行程
    "finger_seat": 10.5, "pocket_depth": 2.5, "pocket_gap": .7,
    "mouth_chamfer": .5, "pocket_min_wall": 2.3,  # 定位槽贴合面、深度、间隙、入口及壁厚要求
    "finger_ref_y": (4.97,3.,0.,-2.,-5.,-8.,-10.8,-18.,-20.,-25.),
    "finger_ref_width": (10.,10.7,11.6,12.1,12.7,13.25,13.76,14.75,15.08,15.84),  # 核对用半宽
    "arm_x": (-56.,-19.), "arm_y": (-13.,0.), "arm_z": (-12.,3.5),  # 悬臂范围
    "post_x": (-56.,-48.), "post_y": (-13.,-1.), "post_z": (-12.,35.),  # 立柱范围
    "leaf_x": (-48.,-11.), "leaf_y": (-13.,-3.), "leaf_z": (8.5,34.2),  # 两片簧片的起始高度
    "leaf_t": .8, "root_r": 1., "leaf_min_straight": 35.,  # 片厚、根部圆角、等厚段下限
    "slide_x": (-11.,11.), "slide_y": (-13.,1.5), "slide_z": (8.5,35.),  # 滑座外界
    "v_angle": 90., "v_apex": -6.0104, "v_relief": .6,  # V 夹角、尖底 Y、根部让位半径
    "ties_z": (15.5,28.), "tie_width": 3.6, "tie_bottom": 1.6, "tie_side": 1.2,  # 扎带槽
    "pen_range": (7.,10.), "pen_d": 8.5, "pen_length": 95., "pen_tip": 65.,  # 笔径范围、检具直径、总长与笔尖 Z
    "stroke": 5., "error": 1.5, "min_wall": 2.0,  # 限位、深度误差和普通壁厚下限
    # S 系。后端 Y 由托面长度、起点及倾角计算。
    "stand_w": 70., "slope": 12., "support_len": 115., "p0": (6.,8.),  # 托座宽度和托面
    "lip_h": 4.5, "lip_t": 5., "rail_w": 4., "rail_s": (12.,50.), "rail_h": 4.5,  # 挡沿、挡边
    "hole_centers": (30.,108.), "hole_y": 12.4, "hole_z": 5.4, "hole_bottom": 2.,  # 横向连杆孔
    "screw_d": 2.8, "screw_x": 35., "screw_z": 4.7,  # 后端 M3 自攻底孔
    "outer_r": 2., "rim_r": 1.,  # 竖直外棱、顶部外棱圆角
    "bar_length": 200., "bar_w": 12., "bar_h": 5., "bar_chamfer": 1., "bar_count": 2,  # 独立连杆
    "assembly_gap": 118., "tablet_w": 250., "tablet_l": 165., "tablet_t": 7.,  # 装配验收位置和平板
}
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
N = 96
EPS = .02  # 仅用于切刀或相接实体的内部搭接，不移动外表面
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.unit_settings.system = 'METRIC'
sc.unit_settings.scale_length = .001
sc.unit_settings.length_unit = 'MILLIMETERS'
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = 32
sc.cycles.use_denoising = True
sc.render.threads_mode = 'FIXED'
sc.render.threads = 16
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
bpy.context.preferences.filepaths.save_version = 0
for directory in [OUT/'stl', OUT/'renders']:
    directory.mkdir(parents=True, exist_ok=True)

def collection(name):
    c = bpy.data.collections.new(name)
    sc.collection.children.link(c)
    return c

parts = collection("01 • Print parts | mm")
demo = collection("02 • Assembly references")
studio = collection("03 • Studio")
def move_collection(o,c):
    for old in list(o.users_collection):
        old.objects.unlink(o)
    c.objects.link(o)

def mesh(name, verts, faces, c=None):
    m = bpy.data.meshes.new(name)
    m.from_pydata(verts, [], faces)
    m.update()
    bm = bmesh.new()
    bm.from_mesh(m)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=.000005)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(m)
    bm.free()
    o = bpy.data.objects.new(name,m)
    (c or parts).objects.link(o)
    return o

def loft(name, rings):
    n, k = len(rings[0]), len(rings)
    faces = [list(reversed(range(n))), list(range((k-1)*n,k*n))]
    for j in range(k-1):
        for i in range(n):
            a, b = j*n+i, j*n+(i+1)%n
            faces.append((a,b,b+n,a+n))
    return mesh(name,[p for ring in rings for p in ring],faces)

def prism(name, poly, axis, lo, hi):
    # 坐标轮换：(Y,Z) 沿 X，(X,Z) 沿 Y，(X,Y) 沿 Z。
    funcs = [lambda a,b,t:(t,a,b),lambda a,b,t:(a,t,b),lambda a,b,t:(a,b,t)]
    return loft(name,[[funcs[axis](a,b,t) for a,b in poly] for t in (lo,hi)])

def box(name,x,y,z):
    return prism(name,[(x[0],y[0]),(x[1],y[0]),(x[1],y[1]),(x[0],y[1])],2,*z)

def circle(r,a=0,b=0):
    return [(a+r*cos(i*2*pi/N), b+r*sin(i*2*pi/N)) for i in range(N)]

def boolean(a,b,mode):
    mod = a.modifiers.new(mode,'BOOLEAN')
    mod.operation, mod.solver, mod.object = mode, 'MANIFOLD', b
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if not a.data.polygons:
        raise ValueError(f"空布尔结果 {a.name} {mode} {b.name}")
    bpy.data.objects.remove(b,do_unlink=True)
    return a

def add(a,b): return boolean(a,b,'UNION')
def cut(a,b): return boolean(a,b,'DIFFERENCE')

def material(name, color, metallic=0, rough=.42):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color,1)
    m.use_nodes=True
    bs=m.node_tree.nodes['Principled BSDF']
    bs.inputs['Base Color'].default_value=(*color,1)
    bs.inputs['Metallic'].default_value=metallic
    bs.inputs['Roughness'].default_value=rough
    return m

teal=material("Lagoon • printed body",(.026,.24,.30))
amber=material("Ochre • crossbars",(.90,.39,.075))
metal=material("Brushed aluminium",(.64,.72,.76),.76,.26)
dark=material("Graphite rubber",(.028,.035,.044))
finger_mat=material("Gripper • satin grey",(.31,.35,.39),.43)
glass=material("Display glass",(.042,.125,.195),.28,.22)
floor_mat=material("Studio • chalk",(.83,.86,.88))

def ink(o,m):
    o.data.materials.clear()
    o.data.materials.append(m)
    return o

def import_finger():
    bpy.ops.wm.stl_import(filepath=str(ROOT/'inputs/finger_link7_mm.stl'))
    o=bpy.context.object
    move_collection(o,demo)
    o.name="Finger A • measured STL"
    bm=bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(o.data)
    bm.free()
    return ink(o,finger_mat)

finger=import_finger()
def pocket_profile():
    # 手指的外侧轮廓在此深度内为直母线。按实物截取 z=2.5 的三角面，
    # 在所有分段端点采样，去掉共线的三角网格节点。
    lines=[]
    h=P['pocket_depth']
    for f in finger.data.polygons:
        pts=[finger.data.vertices[i].co for i in f.vertices]
        hits=[]
        for a,b in zip(pts,pts[1:]+pts[:1]):
            if (a.z<h)!=(b.z<h):
                v=a.lerp(b,(h-a.z)/(b.z-a.z))
                hits.append((v.y,abs(v.x)))
        if len(hits)==2 and abs(hits[0][0]-hits[1][0])>1e-7:
            lines.append(sorted(hits))
    start=P['block_z'][0]+P['finger_tip']-2
    end=max(q[1][0] for q in lines)-.0001
    ys=sorted({round(y,4) for line in lines for y,x in line if start<y<end}|{start,end})
    pts=[]
    for y in ys:
        w=max(a[1]+(y-a[0])*(b[1]-a[1])/(b[0]-a[0]) for a,b in lines if a[0]-.00011<=y<=b[0]+.00011)
        if not pts or y-pts[-1][0]>.003: pts.append((y,w))
    def simplify(p):
        a,b=p[0],p[-1]
        errs=[abs(x-a[1]-(y-a[0])*(b[1]-a[1])/(b[0]-a[0])) for y,x in p]
        k=max(range(len(p)),key=errs.__getitem__)
        return simplify(p[:k+1])[:-1]+simplify(p[k:]) if errs[k]>.008 else [a,b]
    return [(y-P['finger_tip'],w+P['pocket_gap']) for y,w in simplify(pts)]

def make_holder():
    bx,by,bz=[P['block_'+k] for k in 'xyz']
    block=box('Pen holder',bx,by,bz)
    left,right=P['tunnel_x']; bottom,top=P['tunnel_y']; c=P['tunnel_chamfer']
    opening=[(left+c,bottom),(right-c,bottom),(right,bottom+c),(right,top-c),
             (right-c,top),(left+c,top),(left,top-c),(left,bottom+c)]
    cut(block,prism('Octagonal tunnel',opening,2,bz[0]-1,bz[1]+1))
    trace=pocket_profile()
    for sign in (-1,1):
        rings=[]
        c=P['mouth_chamfer']
        for height,delta in [(P['finger_seat'],0),(by[1]-c,0),(by[1],c),(by[1]+1,c)]:
            side=[(w+delta,sign*height,z) for z,w in trace]
            side.append((side[-1][0],sign*height,P['pocket_gap']+delta))
            rings.append(side+[(-x,y,z) for x,y,z in reversed(side)])
        cut(block,loft('Finger pocket',rings))
    add(block,box('Cantilever',(P['arm_x'][0],bx[0]+EPS),P['arm_y'],P['arm_z']))
    add(block,box('Post',P['post_x'],P['post_y'],P['post_z']))
    x0,x1=P['leaf_x']
    t,r=P['leaf_t'],P['root_r']
    for z in P['leaf_z']:
        # 仅凹根有圆角；上下齐平处保留连续平面，限位端面不外伸。
        def edge_path(level,sgn):
            pts=[]
            for x,dir_,limits in [(x0,1,P['post_z']),(x1,-1,P['slide_z'])]:
                allowed=(limits[1]-level if sgn==1 else level-limits[0]) >= r-1e-6
                if allowed:
                    arc=[(x+dir_*r*(1-cos(i*pi/32)),level+sgn*r*(1-sin(i*pi/32))) for i in range(17)]
                else: arc=[(x,level)]
                if dir_==1: pts += [(x-EPS,arc[0][1])]+arc
                else: pts += arc[::-1]+[(x+EPS,arc[0][1])]
            return pts
        profile=edge_path(z,-1)+edge_path(z+t,1)[::-1]
        add(block,prism('Flexure',profile,1,*P['leaf_y']))
    sx,sy,sz=[P['slide_'+k] for k in 'xyz']
    half_angle=radians(P['v_angle']/2)
    w=(sy[1]-P['v_apex'])*tan(half_angle)
    # 在同一轮廓内构造 V 面与让位槽，避免相切切刀留下极细的三角面。
    first=pi/2-half_angle
    arc=[(P['v_relief']*cos(a),P['v_apex']+P['v_relief']*sin(a))
         for a in [first-(2*pi-2*half_angle)*i/72 for i in range(73)]]
    profile=[(sx[0],sy[0]),(sx[1],sy[0]),(sx[1],sy[1]),(w,sy[1])]+arc+[(-w,sy[1]),(sx[0],sy[1])]
    slide=prism('Carriage',profile,2,*sz)
    for z in P['ties_z']:
        zz=(z-P['tie_width']/2,z+P['tie_width']/2)
        cut(slide,box('Under strap',(sx[0]-1,sx[1]+1),(sy[0]-1,sy[0]+P['tie_bottom']),zz))
        for xx in [(sx[0]-1,sx[0]+P['tie_side']),(sx[1]-P['tie_side'],sx[1]+1)]:
            cut(slide,box('Side strap',xx,(sy[0]-1,sy[1]+1),zz))
    return ink(add(block,slide),teal)

ang=radians(P['slope'])
def slope(s,h=0):
    return (P['p0'][0]+s*cos(ang)-h*sin(ang),P['p0'][1]+s*sin(ang)+h*cos(ang))

def make_stand():
    w=P['stand_w']
    rear=slope(P['support_len'])[0]
    # 以沿托面 s 和法向 h 建模。前壁过 A、F，故每个 h 截面上的
    # 前边界 s_front 是线性的。圆角在实体内部生成，保持平板的靠止面。
    p0y,p0z=P['p0']
    s_a=-p0y*cos(ang)-p0z*sin(ang)
    h_a=p0y*sin(ang)-p0z*cos(ang)
    drift=(-P['lip_t']-s_a)/(P['lip_h']-h_a)
    rad=P['outer_r']
    def footprint(h,last,rolling=None):
        pts=[]
        for cx,at_front,start in [(w-rad,False,0),(rad,False,pi/2),
                                  (rad,True,pi),(w-rad,True,3*pi/2)]:
            for i in range(17):
                a=start+i*pi/32
                dx,ds=cos(a),sin(a)
                hh=h
                nx,ns,nh=Vector((dx,ds,-drift*ds if at_front else 0)).normalized()
                if rolling is not None:
                    # 球心位于 h=挡沿高度-R1。每条母线从侧面实际法向
                    # 转向托面法向，在斜前壁上也得到真 R1 弧。
                    rr=P['rim_r']
                    hh+=rr*nh
                cs=s_a+(hh-h_a)*drift+rad if at_front else last-rad
                x=cx+rad*dx
                s=cs+rad*ds
                if rolling is not None:
                    a0=asin(nh)
                    arc=a0+(pi/2-a0)*rolling
                    x+=rr*(dx*cos(arc)-nx)
                    s+=rr*(ds*cos(arc)-ns)
                    hh+=rr*(sin(arc)-nh)
                pts.append((x,*slope(s,hh)))
        return pts

    # 下楔块前壁与挡沿前壁共用同一组母线；后端另按全局竖直轴倒 R2。
    far=P['support_len']+30
    base=loft('Left saddle',[footprint(-80,far),footprint(0,far)])
    plan=[]
    for cx,cy,a0 in [(w-rad,rear-rad,0),(rad,rear-rad,pi/2)]:
        plan.extend((cx+rad*cos(a0+i*pi/32),cy+rad*sin(a0+i*pi/32)) for i in range(17))
    blank=prism('Base and rear trim',plan+[(0,-40),(w,-40)],2,0,100)
    boolean(base,blank,'INTERSECT')
    rings=[footprint(-EPS,0)]
    rings.extend(footprint(P['lip_h']-P['rim_r'],0,i/16) for i in range(17))
    add(base,loft('Bottom stop',rings))
    # 挡边平面为长圆形，R2 的两端与直段相切。顶部逐圈内收形成 R1 滚圆。
    rw=P['rail_w']; sa,sb=P['rail_s']; radius=P['outer_r']
    def rail_ring(inset,h):
        rr=radius-inset
        points=[]
        for cx,ss,start in [(rw-radius,sb-radius,0),(radius,sb-radius,pi/2),
                            (radius,sa+radius,pi),(rw-radius,sa+radius,3*pi/2)]:
            for i in range(17):
                a=start+i*pi/32
                p=(cx+rr*cos(a),*slope(ss+rr*sin(a),h))
                if not points or (Vector(p)-Vector(points[-1])).length>1e-5: points.append(p)
        if (Vector(points[-1])-Vector(points[0])).length<1e-5: points.pop()
        return points
    top=P['rail_h']; rr=P['rim_r']
    rings=[rail_ring(0,-EPS)]
    for i in range(17):
        a=i*pi/32
        rings.append(rail_ring(rr*(1-cos(a)),top-rr+rr*sin(a)))
    add(base,loft('Side fence',rings))
    bottom=P['hole_bottom']
    for y in P['hole_centers']:
        cut(base,box('Crossbar socket',(-1,w+1),(y-P['hole_y']/2,y+P['hole_y']/2),(bottom,bottom+P['hole_z'])))
    cut(base,prism('M3 pilot',circle(P['screw_d']/2,P['screw_x'],P['screw_z']),1,P['hole_centers'][-1],rear+1))
    return ink(base,teal)

def make_bar():
    l,w,h,c=[P[k] for k in ('bar_length','bar_w','bar_h','bar_chamfer')]
    def section(x,d):
        return [(x,y,z) for y,z in [(d,d),(w-d,d),(w-d,h-d),(d,h-d)]]
    return ink(loft('Tie bar',[section(0,c),section(c,0),section(l-c,0),section(l,c)]),amber)

def clone(o,name,c):
    obj=o.copy()
    c.objects.link(obj)
    obj.name=name
    return obj

def tri(o):
    bm=bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm,faces=list(bm.faces))
    bm.to_mesh(o.data); bm.free()

def export(o):
    name=o.name
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active=o
    bpy.ops.wm.stl_export(filepath=str(OUT/'stl'/f'{name}.stl'),export_selected_objects=True,
                          apply_modifiers=True,global_scale=1,use_scene_unit=False)

def T(x,y,z): return Matrix.Translation((x,y,z))
def flip_x(w): return T(w,0,0) @ Matrix.Diagonal((-1,1,1,1))
def bake(o,mat):
    o.data.transform(mat)
    if mat.to_3x3().determinant()<0: o.data.flip_normals()

holder=make_holder()
H=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,-P['block_y'][0]),(0,0,0,1)))
bake(holder,H); holder.name='pen_holder'
left=make_stand(); left.name='stand_left'
right=clone(left,'stand_right',parts); right.data=left.data.copy(); bake(right,flip_x(P['stand_w']))
bar=make_bar(); bar.name='tie_bar'
for ob in [holder,left,right,bar]:
    tri(ob)
    for face in ob.data.polygons:
        face.use_smooth=False
    ob.cycles.shadow_terminator_geometry_offset=0
    export(ob)

# 工程中的打印原件与两个演示装配分区放置。STL 保持各自独立的打印基准。
holder.location.x=330; left.location.x=375; right.location.x=460; bar.location=(375,-26,0)
hand_pose=T(-70,185,25)
ha=hand_pose @ H
held=clone(holder,'Holder • installed',demo); held.matrix_world=hand_pose
finger_map=Matrix(((1,0,0,0),(0,0,1,P['finger_seat']),(0,1,0,-P['finger_tip']),(0,0,0,1)))
finger.matrix_world=ha @ finger_map
finger_b=clone(finger,'Finger B • mirrored',demo)
finger_b.matrix_world=ha @ Matrix.Diagonal((1,-1,1,1)) @ finger_map
def in_hand(o,m):
    move_collection(o,demo); o.matrix_world=ha; return ink(o,m)
r=P['pen_d']/2; tip=P['pen_tip']; tail=tip-P['pen_length']
shaft=in_hand(prism('Stylus • alloy barrel',circle(r),2,tail+4,tip-r),metal)
cap=in_hand(prism('Stylus • end cap',circle(r),2,tail,tail+4),dark)
rings=[[(x,y,tip-r+r*sin(j*pi/40)) for x,y in circle(r*cos(j*pi/40))] for j in range(20)]
rings.append([(0,0,tip)]*N)
nib=in_hand(loft('Stylus • rubber tip',rings),dark)
clip=in_hand(prism('Stylus • pocket clip',[(r-.3,tail+.4),(r+2.6,tail+.4),(r+3.4,tail+2.5),
                   (r+3.4,tail+29),(r+2.55,tail+29.5),(r+2.25,tail+4),(r-.3,tail+4)],1,-1.25,1.25),dark)
def band(z):
    # 演示扎带顺着槽底、侧壁及笔杆外圆绕行。
    half=P['slide_x'][1]-P['tie_side']
    low=P['slide_y'][0]+P['tie_bottom']
    high=min(P['slide_y'][1],P['v_apex']+half/tan(radians(P['v_angle']/2)))
    xy=[Vector((x,y)) for x,y in circle(r)]+[Vector((x,y)) for x in (-half,half) for y in (low,high)]
    poly=[xy[i] for i in geometry.convex_hull_2d(xy)]
    inner=[tuple(v) for v in poly]
    outer=[]
    for i,v in enumerate(poly):
        a=(v-poly[i-1]).normalized(); b=(poly[(i+1)%len(poly)]-v).normalized()
        n=Vector((a.y,-a.x)); m=Vector((b.y,-b.x)); bis=(n+m).normalized()
        side=1 if sum(inner[j][0]*inner[(j+1)%len(poly)][1]-inner[j][1]*inner[(j+1)%len(poly)][0] for j in range(len(poly)))>0 else -1
        outer.append(tuple(v+side*.6*bis/bis.dot(n)))
    band=prism('Cable tie',outer,2,z-1.3,z+1.3)
    cut(band,prism('Tie interior',inner,2,z-2,z+2))
    return in_hand(band,dark)
ties=[band(z) for z in P['ties_z']]
hand=[held,finger,finger_b,shaft,cap,nib,clip,*ties]

# +X 正视时隧道侧墙会遮住笔杆。仅在 +X 剖视图显示中面处的演示切片，
# 标出实际限位端面；保存的装配和导出件仍使用完整网格。
section=clone(held,'Holder • X=0 longitudinal section',demo)
section.data=held.data.copy()
section.matrix_world=Matrix.Identity(4)
boolean(section,box('Section tool',(-1.,0.),(-100,100),(-100,100)),'INTERSECT')
section.matrix_world=held.matrix_world
section.data.materials.clear()
section.data.materials.append(teal)
section.data.materials.append(material('Section face',(.70,.33,.08)))
for face in section.data.polygons:
    face.material_index=int(all(abs(section.data.vertices[i].co.x)<.00001 for i in face.vertices))
dim_mat=material('Dimension ink',(.03,.06,.08))
guides=[]
low=P['block_z'][1]; high=P['slide_z'][0]
line_y=P['block_y'][0]-6
for z in (low,high):
    guides.append(in_hand(box('Gap • witness',(30,30.12),(line_y-1.4,P['slide_y'][0]-.15),(z-.055,z+.055)),dim_mat))
    guides.append(in_hand(prism('Gap • tick',[(line_y-.65,z-.7),(line_y-.55,z-.8),(line_y+.65,z+.7),(line_y+.55,z+.8)],0,30,30.12),dim_mat))
guides.append(in_hand(box('Gap • dimension',(30,30.12),(line_y-.065,line_y+.065),(low,high)),dim_mat))
font=bpy.data.curves.new('Gap label','FONT')
font.body=f'{high-low:.1f} mm';font.size=2.4;font.align_y='CENTER';font.align_x='RIGHT'
label=bpy.data.objects.new('Gap • label',font)
demo.objects.link(label)
label.matrix_world=ha @ T(30,line_y-2,(high+low)/2) @ Matrix(((0,0,1,0),(1,0,0,0),(0,1,0,0),(0,0,0,1)))
font.materials.append(dim_mat)
guides.append(label)
detail=[section,*hand[1:],*guides]
detail_only=[section,*guides]
for obj in detail_only: obj.hide_set(True)

ox,oy=0,0  # 装配使用 S 系作为世界坐标
sl=clone(left,'Left saddle • assembled',demo); sl.location=(ox,oy,0)
sr=clone(right,'Right saddle • assembled',demo); sr.location=(ox+P['stand_w']+P['assembly_gap'],oy,0)
bars=[]
total=2*P['stand_w']+P['assembly_gap']
for y in P['hole_centers']:
    b=clone(bar,'Crossbar • installed',demo)
    b.location=(ox+(total-P['bar_length'])/2,oy+y-P['bar_w']/2,P['hole_bottom']+(P['hole_z']-P['bar_h'])/2)
    bars.append(b)

def rounded(w,l,r):
    return [(cx+r*cos(a+i*pi/32),cy+r*sin(a+i*pi/32))
            for cx,cy,a in [(w-r,l-r,0),(r,l-r,pi/2),(r,r,pi),(w-r,r,3*pi/2)] for i in range(17)]
tabmat=T(ox+P['rail_w'],oy+P['p0'][0],P['p0'][1]) @ Matrix.Rotation(ang,4,'X')
def tablet_layer(name,w,l,h0,h1,inset,m):
    p=[(x+inset,y+inset) for x,y in rounded(w-2*inset,l-2*inset,max(.8,6-inset))]
    o=prism(name,p,2,h0,h1); move_collection(o,demo); o.matrix_world=tabmat
    return ink(o,m)
tw,tl,tt=P['tablet_w'],P['tablet_l'],P['tablet_t']
tablet=tablet_layer('Tablet • gauge envelope',tw,tl,0,tt,0,dark)
# 屏幕面用贴面材质，模型厚度保持指定值。
screen=mesh('Tablet • glass',[(x+4,y+4,tt+.001) for x,y in rounded(tw-8,tl-8,3)],
            [list(range(68))],demo)
screen.matrix_world=tabmat; ink(screen,glass)
def ghost_material(base,opacity):
    # 装配等轴图用半透明检具显示被平板遮挡的两根杆。
    mat=base.copy()
    mat.name=base.name+' • translucent assembly gauge'
    nodes=mat.node_tree.nodes
    links=mat.node_tree.links
    mix=nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value=opacity
    links.new(nodes.new('ShaderNodeBsdfTransparent').outputs[0],mix.inputs[1])
    links.new(nodes.get('Principled BSDF').outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],nodes.get('Material Output').inputs['Surface'])
    return mat
finger.data.materials[0]=ghost_material(finger_mat,.27)  # 两指共享材质
ghost_case=ghost_material(dark,.17)
ghost_screen=ghost_material(glass,.16)
# 透明展示时，平板底面与托座的三角面共面，会产生明暗三角纹。
# 保留完整平板用于普通视图；透视演示只显示其可见外壳和屏幕。
shell=clone(tablet,'Tablet • translucent viewing shell',demo)
shell.data=tablet.data.copy()
bm=bmesh.new()
bm.from_mesh(shell.data)
bmesh.ops.delete(bm,geom=[face for face in bm.faces if face.normal.z<-.99],context='FACES')
bm.to_mesh(shell.data)
bm.free()
shell.data.materials[0]=ghost_case
shell.hide_set(True)
detail_only.append(shell)
stand_assy=[sl,sr,*bars,tablet,screen]

sc.world=bpy.data.worlds.new("Soft daylight")
sc.world.use_nodes=True
sc.world.node_tree.nodes.get('Background').inputs[0].default_value=(.84,.89,.96,1)
sc.world.node_tree.nodes.get('Background').inputs[1].default_value=.46
# 相机看见的背景与环境补光分别控制，正视图也保持浅色底。
world_nodes=sc.world.node_tree.nodes
path=world_nodes.new('ShaderNodeLightPath')
back=world_nodes.new('ShaderNodeBackground')
back.inputs[0].default_value=(.83,.89,.94,1)
back.inputs[1].default_value=1.25
mix=world_nodes.new('ShaderNodeMixShader')
links=sc.world.node_tree.links
links.new(path.outputs['Is Camera Ray'],mix.inputs[0])
links.new(world_nodes.get('Background').outputs[0],mix.inputs[1])
links.new(back.outputs[0],mix.inputs[2])
links.new(mix.outputs[0],world_nodes.get('World Output').inputs['Surface'])
floor=mesh('Backdrop',[(-4000,-4000,0),(4000,-4000,0),(4000,4000,0),(-4000,4000,0)],[(0,1,2,3)],studio)
ink(floor,floor_mat)
floor.hide_set(True)
lamps=[]
for name in ['Key • softbox','Fill • softbox','Rim • softbox']:
    data=bpy.data.lights.new(name,'AREA')
    ob=bpy.data.objects.new(name,data); studio.objects.link(ob); lamps.append(ob); ob.hide_set(True)

# name, visible geometry, line from center to eye, view up, image size, floor
shots=[
    ('A1_holder_iso',[holder],(1,-1,1),(0,-1,0),(1100,820),True),
    ('A2_holder_top',[holder],(0,0,1),(0,-1,0),(1000,850),False),
    ('A3_holder_side',[holder],(1,0,0),(0,-1,0),(720,900),False),
    ('A4_holder_assembly',hand,(1.6,-.78,.85),(0,-1,0),(1050,920),False),
    ('A4b_assembly_plusX_section',detail,(1,0,0),(0,-1,0),(960,1000),False),
    ('B1_left_iso',[left],(1,-1,1),(0,0,1),(1100,820),True),
    ('B2_stand_assembly',[sl,sr,*bars,shell,screen],(1,1,1),(0,0,1),(1200,760),True),
    ('B3_stand_side',stand_assy,(-1,0,0),(0,0,1),(1200,600),False),
]
models=list(parts.objects)+list(demo.objects)
def light(ob,at,offset,size,strength):
    ob.location=at+offset
    ob.rotation_euler=(-offset).to_track_quat('-Z','Y').to_euler()
    ob.data.shape='DISK'
    ob.data.size=size
    ob.data.energy=strength

def setup_shot(spec):
    name,objs,direction,up,(width,height),ground=spec
    ghost=name=='B2_stand_assembly'
    tablet.data.materials[0]=ghost_case if ghost else dark
    screen.data.materials[0]=ghost_screen if ghost else glass
    for obj in models: obj.hide_render=obj not in objs
    bpy.context.view_layer.update()
    # 剖视图专注夹持区；手指远端不参与取景。
    framing=[section,shaft,nib,*guides[:-1]] if name=='A4b_assembly_plusX_section' else objs
    pts=[o.matrix_world @ v.co for o in framing for v in o.data.vertices]
    if name=='A4b_assembly_plusX_section':
        pts.extend(ha @ Vector((0,y,-41)) for y in (-27,32))
    eye=Vector(direction).normalized()
    right=Vector(up).cross(eye).normalized()
    top=eye.cross(right).normalized()
    ranges=[(min(v.dot(axis) for v in pts),max(v.dot(axis) for v in pts)) for axis in (right,top,eye)]
    center=sum((axis*((lo+hi)/2) for axis,(lo,hi) in zip((right,top,eye),ranges)),Vector((0,0,0)))
    span=max(ranges[0][1]-ranges[0][0],(ranges[1][1]-ranges[1][0])*width/height)
    cam=bpy.data.objects.get(name)
    if cam is None:
        data=bpy.data.cameras.new(name); cam=bpy.data.objects.new(name,data); studio.objects.link(cam)
        cam.hide_set(True)
    cam.matrix_world=Matrix((right,top,eye)).transposed().to_4x4()
    cam.location=center+eye*(3*span)
    cam.data.type='ORTHO'
    cam.data.ortho_scale=span*1.13*max(1,height/width)
    cam.data.clip_end=5000
    sc.camera=cam
    sc.render.resolution_x=width; sc.render.resolution_y=height
    floor.hide_render=not ground
    floor.location.z=min(v.z for v in pts)-.035
    for ob,v,s,k in zip(lamps,[(-.8,-.75,1.6),(1.1,-.32,.9),(.3,1.2,1.45)],[1.1,1.05,.94],[18,4,9]):
        offset=Vector(v)*span
        light(ob,center,offset,span*s,span*span*k)
    sc.render.filepath=str(OUT/'renders'/f'{name}.png')

def main():
    for spec in shots: setup_shot(spec)
    setup_shot(shots[0])
    # 保存文件时使两个装配与所有打印原件均可见。
    for o in models: o.hide_render=o in detail_only
    for area in bpy.context.screen.areas:
        if area.type=='VIEW_3D':
            space=area.spaces.active
            space.overlay.show_overlays=False
            space.shading.color_type='MATERIAL'
            space.region_3d.view_rotation=sc.camera.matrix_world.to_quaternion()
            space.region_3d.view_distance=540
            space.region_3d.view_location=(244,66,11)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active=holder
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'tablet_tap_parts.blend'))
    if '--model' not in sys.argv:
        for spec in shots:
            setup_shot(spec)
            bpy.ops.render.render(write_still=True)
    print("Build complete",flush=True)

if __name__=='__main__':
    main()
import os, sys; sys.stdout.flush(); os._exit(0)
