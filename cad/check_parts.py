"""用任务书的独立检具检查最终 STL。运行：任务书提供的 .venv/bin/python -B check_parts.py。"""
from pathlib import Path
from math import pi, atan2, cos, sin, radians, degrees, sqrt
import os
import sys
import numpy as np
import trimesh as tm
import manifold3d as md
import pymeshlab as ml
from shapely import affinity
from shapely.geometry import Polygon,Point,LineString
from shapely.ops import unary_union,nearest_points

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out'
STL=OUT/'stl'
VOL_TOL=1e-4   # mm³，浮点接触面数值噪声
DIM_TOL=2e-4   # mm
log=[]
results=[]

def record(label,ok,detail):
    ok=bool(ok)
    results.append(ok)
    log.append(f"{'PASS' if ok else 'FAIL'}  {label}  |  {detail}")

def load(name):
    return tm.load_mesh(STL/(name+'.stl'))

def solid(m):
    raw=md.Mesh64(np.ascontiguousarray(m.vertices,dtype=np.float64),
                  np.ascontiguousarray(m.faces,dtype=np.uint64))
    s=md.Manifold(raw)
    if s.status()!=md.Error.NoError: raise ValueError(str(s.status()))
    return s

def box(lo,hi):
    return md.Manifold.cube(np.subtract(hi,lo)).translate(lo)

def vol(s):
    return max(0.,s.volume())

def intervals(mesh,origin,axis):
    """从指定截线的负端发射射线，返回全部材料边界的轴坐标。"""
    q=np.array(origin,dtype=float)
    q[axis]=mesh.bounds[0,axis]-10.
    u=np.eye(3)[axis]
    v,_,_=mesh.ray.intersects_location([q],[u])
    return np.unique(np.round(v[:,axis],7))

def near(vals,ref,tol=DIM_TOL):
    return np.max(np.abs(np.asarray(vals)-ref))<=tol

def fmt(vals,places=4):
    return ' / '.join(f'{v:.{places}f}' for v in vals)

def topology(name,m):
    counts=np.bincount(m.edges_unique_inverse)
    ms=ml.MeshSet()
    ms.add_mesh(ml.Mesh(np.asarray(m.vertices),np.asarray(m.faces,dtype=np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    count=np.count_nonzero(ms.current_mesh().face_selection_array())
    n=len(m.split(only_watertight=False))
    bad=np.count_nonzero(counts!=2)
    record(name+' 单实体、水密、法向一致、无自交',
           n==1 and bad==count==0 and m.is_volume and m.area_faces.min()>0,
           f'{n} 个连通分量，非双邻面边 {bad}，自交面 {count}，体积 {m.volume:.3f} mm³')

def finger_solid():
    m=tm.load_mesh(ROOT/'inputs/finger_link7_mm.stl')
    pieces=m.split(only_watertight=False)
    shapes=[]
    for part in pieces:
        tm.repair.fix_normals(part,multibody=True)
        shapes.append(solid(part))
    return m,md.Manifold.batch_boolean(shapes,md.OpType.Add)

def bridge_rows(face,bank,axis):
    """检查桥的各个宽度变化位置；桥下实物截面须覆盖两侧落丝点。"""
    # 统一以 u 为搭桥方向，v 为逐行方向。
    if axis==1:
        face=affinity.affine_transform(face,[0,1,1,0,0,0])
        bank=affinity.affine_transform(bank,[0,1,1,0,0,0])
    u0,v0,u1,v1=face.bounds
    levels={v0,v1}
    for g in [face,bank]:
        for p in getattr(g,'geoms',[g]):
            for ring in [p.exterior,*p.interiors]:
                levels.update(y for x,y in ring.coords if v0<y<v1)
    levels=sorted(levels)
    widths=[]
    both=[True,True]
    for a,b in zip(levels,levels[1:]):
        if b-a<.00001: continue
        inset=min(.000005,(b-a)/4)
        for v in [a+inset,(a+b)/2,b-inset]:
            cut=face.intersection(LineString([(u0-1,v),(u1+1,v)]))
            for line in getattr(cut,'geoms',[cut]):
                if line.length<.000001: continue
                lo,_,hi,_=line.bounds
                widths.append(hi-lo)
                both[0] &= bank.covers(Point(lo-.025,v))
                both[1] &= bank.covers(Point(hi+.025,v))
    return min(widths),max(widths),both

def audit_bridges(m,name,expected):
    # 每个独立的水平朝下区域单列。底面自身有热床承托，跨度为零。
    down=m.face_normals[:,2]<-1+1e-8
    heights=np.round(m.triangles_center[:,2],5)
    seen=[]
    log.append(f'    {name}热床接触面：Z=0，桥跨 0，面下全由热床承托。')
    for z in np.unique(heights[down&(heights>DIM_TOL)]):
        triangles=m.triangles[down&(heights==z)]
        roof=unary_union([Polygon(t[:,:2]) for t in triangles])
        bank=Polygon()
        for p in solid(m).slice(z-.002).to_polygons():
            bank=bank.symmetric_difference(Polygon(p))
        for face in sorted(getattr(roof,'geoms',[roof]),key=lambda f:f.centroid.y):
            # 以实测高度及平面位置辨认特征，仍枚举未识别的水平面。
            matches=[r for r in expected if near(z,r[1]) and r[2][0]<face.centroid.y<r[2][1]]
            if len(matches)!=1:
                record(f'{name} 未识别的朝下水平面',False,f'Z={z}，面积 {face.area:.3f} mm²')
                continue
            label,level,window,axis,span_min,span_max=matches[0]
            lower,upper,ends=bridge_rows(face,bank,axis)
            seen.append(label)
            span_ok=span_max is None or (near(upper,span_max,.015) and lower>=span_min-.015)
            record(f'{name} {label}',all(ends) and span_ok,
                   f'打印 Z={z:.3f}，沿打印 {"XY"[axis]} 跨 {lower:.4f}–{upper:.4f} mm，负向端/正向端支承 {"有" if ends[0] else "无"}/{"有" if ends[1] else "无"}，面积 {face.area:.2f} mm²')
    return sorted(seen)==sorted(r[0] for r in expected)

def plane_check(m,label):
    # 从托面中央的射线命中面出发，按真实网格邻接扩展至轮廓与挡边棱线。
    # 提取过程不按到目标平面的距离筛点，保留整个连通托面的顶点和窄三角形。
    _,_,ids=m.ray.intersects_location([[35,60,100]],[[0,0,-1]],multiple_hits=False)
    adj=m.face_adjacency[m.face_adjacency_angles<radians(25)]
    groups=tm.graph.connected_components(adj,nodes=np.arange(len(m.faces)))
    faces=next(g for g in groups if ids[0] in g)
    points=m.vertices[np.unique(m.faces[faces])]
    centered=points-points.mean(axis=0)
    normal=np.linalg.svd(centered,full_matrices=False)[2][-1]
    if normal[2]<0: normal=-normal
    residual=np.abs(centered@normal).max()
    nc=m.face_normals[faces]@normal
    angle=degrees(np.arccos(np.clip(nc.min(),-1,1)))
    record('B3 '+label+'托面全顶点平面度及面法向',
           residual<=.001 and nc.min()>0 and angle<=.01 and m.area_faces[faces].sum()>7800,
           f'{len(faces)} 面、{len(points)} 顶点、面积 {m.area_faces[faces].sum():.3f} mm²，拟合平面最大偏距 {residual:.9f} mm，最大法向偏角 {angle:.6f}°，反向面 {(nc<=0).sum()}（法向一致性数值界限 0.01°）')


def check_holder():
    printmesh=load('pen_holder')
    topology('A1 笔夹',printmesh)
    h=printmesh.copy()
    h.apply_transform([[1,0,0,0],[0,0,1,-13],[0,-1,0,0],[0,0,0,1]])
    body=solid(h)
    dims=h.extents
    record('A6 H 系包络',near(dims,[75,26,62]),fmt(dims,3)+' mm')
    bed=np.isclose(printmesh.triangles[:,:,2],0,atol=DIM_TOL).all(axis=1)
    record('A5a 落床基准',abs(h.bounds[0,1]+13)<DIM_TOL and bed.any(),
           f'最低 Y_H={h.bounds[0,1]:.4f}，STL 最低 Z={printmesh.bounds[0,2]:.4f}，落床面积 {printmesh.area_faces[bed].sum():.2f} mm²')
    # 在最薄的顶唇处实测左右两侧，并测出另外几条可能的薄壁。
    widths=[]
    for level in (15.5,28,20):
        xs=intervals(h,[0,1.5-.000001,level],0)
        widths.extend((xs[1]-xs[0],xs[-1]-xs[-2]))
    tunnel=intervals(h,[0,.01,-8],0)
    right_wall=tunnel[-1]-tunnel[-2]
    faces_y=intervals(h,[1,0,-8],1)
    skins=[faces_y[1]-faces_y[0],faces_y[-1]-faces_y[-2]]
    tunnel_top=8.0  # 独立验收基准，H 系，打印 Z=Y_H+13
    record('A1 隧道顶至上指尖定位槽底壁厚 ≥2.5',
           near(faces_y[-2:], [tunnel_top,10.5]) and skins[1]>=2.5-DIM_TOL,
           f'隧道顶 Y_H={faces_y[-2]:.4f}，上槽底 Y_H={faces_y[-1]:.4f}，壁厚 {skins[1]:.4f} mm')
    base_y=intervals(h,[0,0,15.5],1)
    base=base_y[-1]-base_y[0]
    rawfinger,fs=finger_solid()
    ga=fs.transform([[1,0,0,0],[0,0,1,10.5],[0,1,0,-4.97]])
    gb=ga.mirror((0,1,0))
    iv=[vol(body^f) for f in (ga,gb)]
    record('A2a 两根实测手指布尔交集',max(iv)<VOL_TOL,fmt(iv,7)+' mm³（修正原 STL 两个壳体的法向后并集）')
    floors=intervals(h,[2,0,-6.3],1)
    fingerseat=intervals(rawfinger,[2,-6.3+4.97,0],2)[0]+10.5
    gap=[fingerseat-max(floors),min(floors)+fingerseat]
    record('A2b 两侧指腹贴底',near(gap,0),f'槽底 Y={min(floors):.4f} / +{max(floors):.4f}，间隙 '+fmt(gap)+' mm')
    side=[]
    for y in np.linspace(-22.029,4.966,121):
        ref=intervals(rawfinger,[0,y,2.5],0)
        for Y in [11.98,-11.98]:
            xs=intervals(h,[0,Y,y-4.97],0)
            side.extend([xs[-2]-ref[-1],ref[0]-xs[1]])
    record('A2c 楔形槽侧向间隙',min(side)>=.6 and max(side)<=.8,
           f'两槽左右壁，121 个纵向截面，相对手指 z=2.5 截面 {min(side):.4f}–{max(side):.4f} mm（入口倒角以下）')
    # 直接取两个槽的外口棱线和倒角起点，避免内移探针高估口沿厚度。
    wall=[]
    for yy in (12.5,13.0):
        fronts=[]; sides=[]
        for sign in (-1,1):
            vs=h.vertices[np.abs(h.vertices[:,1]-sign*yy)<.00001]
            rim=vs[(np.abs(vs[:,0])>5)&(np.abs(vs[:,0])<18)&(vs[:,2]<3)]
            outer=intervals(h,[18.7,sign*12.0,0],2)
            fronts.append(outer[-1]-rim[:,2].max())
            outer_x=intervals(h,[0,sign*12.0,-23.5],0)
            sides.extend((rim[:,0].min()-outer_x[0],outer_x[-1]-rim[:,0].max()))
        wall.append((min(fronts),min(sides)))
    record('A2d 定位槽前墙和侧墙 ≥2.3（含入口）',np.min(wall)>=2.3-DIM_TOL,
           f'两槽取小值，直壁处前 / 侧 {fmt(wall[0])}；入口处前 / 侧 {fmt(wall[1])} mm')
    wall_min=min([*widths,*skins,right_wall,base,*np.ravel(wall)])
    record('A1 普通结构壁厚及扎带槽内唇边 ≥2.0',wall_min>=2-DIM_TOL,
           f'两处扎带槽左右顶唇 {fmt(widths[:4])}；槽外左右 {fmt(widths[4:])}；隧道右墙 {right_wall:.4f}，隧道至两定位槽底 {fmt(skins)}，让位圆槽至扎带槽底 {base:.4f}；计入定位槽前侧墙后最薄 {wall_min:.4f} mm')
    pen=md.Manifold.cylinder(95,4.25,circular_segments=720).translate((0,0,-30))
    block=body^box((-19,-14,-28),(20,14,3.5))
    carriage=body^box((-11,-14,8.5),(11,14,36))
    fixed=body^box((-57,-14,-28),(20,14,3.5))
    # 以真实斜面到笔轴的法向距离判断线接触；布尔截面独立使用 720 边名义笔。
    normal=h.face_normals
    fc=h.triangles_center
    face_data=[]
    for sgn in [-1,1]:
        mask=(sgn*normal[:,0]>.69)&(normal[:,1]>.69)&(abs(normal[:,2])<.00001)&(fc[:,2]>10)
        i=np.flatnonzero(mask)[np.argmax(h.area_faces[mask])]
        face_data.append((abs(np.dot(normal[i],fc[i])),normal[i]))
    dist=np.array([d for d,n in face_data]); depth=4.25-dist
    dv=vol(carriage^pen)
    # 从左右实测 V 面反解笔轴就座位置，不假定槽尖的名义高度。
    normals=np.array([n[:2] for _,n in face_data])
    def seated(diameter):
        return np.linalg.solve(normals,diameter/2-dist)
    nominal=seated(8.5)
    record('A3a Ø8.5 原点轴线两面相切（±0.02）',near(depth,0,.02),
           f'轴到两斜面 {fmt(dist,7)}，法向嵌入 {fmt(depth,7)} mm，笔与滑座交集 {dv:.7f} mm³；相切轴线 Y={nominal[1]:+.7f} mm')
    ranges=[]
    for diameter in (7,8.5,10):
        xc,yc=seated(diameter)
        gauge=md.Manifold.cylinder(95,diameter/2,circular_segments=720).translate((xc,yc,-30))
        ranges.append(vol(gauge^(block+carriage)))
    record('A3 V 槽笔径范围',max(ranges)<VOL_TOL,
           f'Ø7/8.5/10 各自相切就座时对夹持块和滑座的交集 {fmt(ranges,7)} mm³')
    for delta in [0,-5]:
        for size,p in [(8.5,pen),(10,gauge)]:
            shifted=p.translate((0,0,delta))
            v=[vol(shifted^block),vol(shifted^ga),vol(shifted^gb)]
            record(f'A3b Ø{size} 笔对夹持块及两指，ΔZ={delta}',max(v)<VOL_TOL,
                   '交集 '+fmt(v,7)+' mm³')
    clear=[]; ceiling=[]
    center=Point(*seated(10))
    for level in [-26.999,-12.1,-.1,3.499]:
        sec=h.section(plane_origin=[0,0,level],plane_normal=[0,0,1])
        loops=[Polygon(r[:,:2]) for r in sec.discrete]
        aperture=min((p for p in loops if p.covers(center)),key=lambda p:p.area)
        boundary=aperture.boundary
        nearest=nearest_points(center,boundary)[1]
        clear.append(center.distance(nearest)-5)
        ceiling.append(aperture.bounds[3]-center.y-5)
    record('A3c Ø10 与八边形隧道的全周最小净空 ≥0.9',min(clear)>=.9,
           f'就座轴线 Y={center.y:+.6f}，四个贯通截面净空 {fmt(clear,6)}；最接近左上 45° 面，垂足 (X,Y)=({nearest.x:.4f},{nearest.y:.4f})。水平顶壁净空 {min(ceiling):.6f} mm；后移沿等截面，不改变径向距离。')
    sliding=carriage.translate((0,0,-5))
    v=[vol(sliding^fixed),vol(sliding^ga),vol(sliding^gb)]
    record('A3d 滑座移动 −5 后限位、无体积干涉',max(v)<VOL_TOL,
           '滑座与固定端 / 指 A / 指 B 的交集 '+fmt(v,7)+' mm³')
    z=intervals(h,[-7.7,-8.1,0],2)
    stroke=z[-2]-z[-3]
    gapbox=vol(body^box((-11,-13,3.5),(11,1.5,8.5)))
    record('A4a 滑座与夹持块间行程',near(stroke,5) and gapbox<VOL_TOL,
           f'端面差 {stroke:.4f}，滑座全宽限位区内材料 {gapbox:.7f} mm³；几何居中预压 {stroke/2:.1f} ± 1.5 → {stroke/2-1.5:.1f}–{stroke/2+1.5:.1f} mm')
    z=intervals(h,[-29,-7.8,0],2)
    z=z[z>7]  # 排除同一条截线穿过的悬臂
    th=z[1::2]-z[::2]
    section=h.section(plane_origin=[0,-8,0],plane_normal=[0,1,0])
    # 找到四条等高直边在 X 上的覆盖交集，即两侧均保持 0.8 的有效长度。
    spans=[]
    for level in z:
        seg=section.vertices
        pt=seg[(abs(seg[:,2]-level)<.00002)&(seg[:,0]>=-48)&(seg[:,0]<=-11)]
        spans.append((pt[:,0].min(),pt[:,0].max()))
    lengths=[min(a[1],b[1])-max(a[0],b[0]) for a,b in zip(spans[::2],spans[1::2])]
    record('A4b 两片簧片尺寸与等厚段',len(th)==2 and near(th,.8) and min(lengths)>=35-DIM_TOL and near(z,[8.5,9.3,34.2,35]),
           f'厚 {fmt(th)}，等厚段 {fmt(lengths,3)} mm；截面高度 '+fmt(intervals(h,[-29,0,8.6],1),2))
    root_radii=[]
    for xcenter,sign in [(-47,-1),(-12,1)]:
        for zcenter,zsign in [(10.3,-1),(33.2,1)]:
            xy=h.vertices[:,[0,2]]
            mask=(abs(h.vertices[:,1]+3)<.00002)
            for d in [(xy[:,0]-xcenter)*sign,(xy[:,1]-zcenter)*zsign]:
                mask &= (d>.001)&(d<.999)
            arc=np.unique(xy[mask],axis=0)
            # 圆方程的线性最小二乘，求真实 STL 节点的拟合半径。
            if len(arc)<3: raise ValueError('簧片圆角缺少圆弧节点')
            rel=arc-arc.mean(axis=0)
            a,b,c=np.linalg.lstsq(np.c_[2*rel,np.ones(len(rel))],(rel*rel).sum(axis=1),rcond=None)[0]
            root_radii.append(sqrt(c+a*a+b*b))
    record('A4b 簧片内侧根部圆角',near(root_radii,1,.002),'四个内侧根部拟合 R='+fmt(root_radii)+' mm')
    slits=[box((-29.1,-14,lo-.1),(-28.9,0,hi+.1)) for lo,hi in zip(z[::2],z[1::2])]
    counts=[len(s.decompose()) for s in (body-slits[0],body-slits[1],body-slits[0]-slits[1])]
    record('A4c 仅两片簧片连接',counts==[1,1,2],f'分别切断第 1 片 / 第 2 片 / 两片后的连通数 {counts}')
    zs=intervals(h,[10.4,-8,0],2)
    tie_edges=zs[(zs>10)&(zs<33)]
    margin=min(tie_edges[0]-z[1],z[-2]-tie_edges[-1])
    record('A4d 扎带槽不碰簧片',near(tie_edges,[13.7,17.3,26.2,29.8]) and margin>0,
           f'两槽 Z 边界 {fmt(tie_edges,2)}，距簧片最近 {margin:.2f} mm')
    expected=[
        ('下指尖定位槽',2.5,(-2,28),0,None,None),
        ('隧道顶',13+tunnel_top,(-5,28),0,14,14),
        ('后侧扎带底槽 Z_H=15.5',1.6,(-17,-14),1,3.6,3.6),
        ('前侧扎带底槽 Z_H=28',1.6,(-30,-26),1,3.6,3.6)]
    found=audit_bridges(printmesh,'A5b',expected)
    nz=printmesh.face_normals[:,2]
    sloping=nz[(nz<-.000001)&(nz>-1+1e-8)]
    angle=degrees(np.arcsin(-sloping.min()))
    record('A5c 朝下表面审计',found and angle<=45.01,
           f'预期四个非落床水平区的逐区核对结果见前四行；其他朝下斜面相对竖直的最大悬垂角 {angle:.5f}°')
    log.append('注：上述行程为刚体几何检具结果。实际触屏力、材料疲劳及打印桥接质量需要实物试验。')

def check_stand():
    left,right,bar=[load(k) for k in ('stand_left','stand_right','tie_bar')]
    for name,m in [('左托座',left),('右托座',right),('连杆',bar)]:
        topology('B1 '+name,m)
        record('B1 '+name+' 落床',near(m.bounds[0,2],0),f'最低 Z={m.bounds[0,2]:.4f} mm')
    L,R=solid(left),solid(right).translate((188,0,0))
    mirror=L.mirror((1,0,0)).translate((70,0,0))
    diff=vol(mirror-solid(right))+vol(solid(right)-mirror)
    record('B1 镜像与连杆外形',diff<.05 and near(bar.extents,[200,12,5]),
           f'左右镜像体积对称差 {diff:.5f} mm³（STL 浮点）；连杆长宽高 {fmt(bar.extents,2)} mm')
    angles=[]; slopes=[]
    for m in [left,right]:
        heights=[intervals(m,[27.3,y,0],2)[-1] for y in (20.,95.)]
        k=(heights[1]-heights[0])/75; slopes.append(k)
        angles.append(degrees(atan2(heights[1]-heights[0],75)))
    record('B3 托面倾角',near(angles,12,.1),'左右实测 '+fmt(angles,5)+'°')
    theta=radians(12.)
    def pos(s,n): return (6+s*cos(theta)-n*sin(theta),8+s*sin(theta)+n*cos(theta))
    y,z0=pos(0,0)
    tab=box((0,0,0),(250,165,7)).rotate((12,0,0)).translate((4,y,z0))
    iv=[vol(tab^stand) for stand in [L,R]]
    # 对贴合的大面积面先报告原始交集，再检查是否侵入检具表层公差以内的内核。
    t=DIM_TOL
    core=box((t,t,t),(250-t,165-t,7-t)).rotate((12,0,0)).translate((4,y,z0))
    deep=[vol(core^stand) for stand in (L,R)]
    y,z=pos(30,2)
    left_edge=intervals(left,[0,y,z],0)[-1]
    right_edge=188+intervals(right,[0,y,z],0)[0]
    separation=R.bounding_box()[0]-L.bounding_box()[3]
    record('B2a 118 间距、250 平板无超公差干涉',
           near(separation,118) and near([left_edge,right_edge],[4,254]) and max(deep)<VOL_TOL,
           f'托座净距 {separation:.3f}；挡边内面 X={left_edge:.3f}/{right_edge:.3f}，净宽 {right_edge-left_edge:.3f}；平板 X=4–254。原始浮点交集 {fmt(iv,7)} mm³；检具六面各内缩 {t} mm 后的内核交集 {fmt(deep,7)} mm³')
    slope_n=np.array([0,-sin(theta),cos(theta)])
    heights=(left.vertices-np.array([0,6,8]))@slope_n
    along=(left.vertices-np.array([0,6,8]))@np.array([0,cos(theta),sin(theta)])
    lip_h=max(heights[along<1]); rail_h=max(heights[along>10])
    py,pz=pos(20,0)
    backgap=pz-intervals(left,[27.3,py,0],2)[-1]
    # 沿斜面从挡沿前方穿到挡沿内侧，用同一平板局部坐标测靠止面。
    local=left.copy()
    local.apply_translation([0,-6,-8])
    local.apply_transform(tm.transformations.rotation_matrix(-theta,[1,0,0]))
    stop=intervals(local,[27.3,0,2.1],1)
    stopgap=min(abs(stop))
    record('B2b 背面贴合、下沿靠止，挡沿/挡边低于正面',near(backgap,0) and near(stopgap,0) and max(lip_h,rail_h)<7 and near([lip_h,rail_h],4.5),
           f'托面间隙 {abs(backgap):.6f}，前挡面间隙 {stopgap:.6f}；挡沿 / 挡边法向高度 {fmt([lip_h,rail_h])}，低于正面 {7-max(lip_h,rail_h):.4f} mm')
    all_clear=[]; engagements=[]; collisions=[]
    for cy in [30,108]:
        rod=solid(bar).translate((29,cy-6,2.2))
        ends=rod.bounding_box()
        for m,stand in [(left,L),(right,R)]:
            q=stand.bounding_box()
            engagements.append(min(q[3],ends[3])-max(q[0],ends[0]))
            collisions.append(vol(rod^stand))
            yy=intervals(m,[36.9,cy,4.7],1)
            zz=intervals(m,[36.9,cy,0],2)
            pair=[max(yy[yy<cy]),min(yy[yy>cy]),zz[1],zz[2]]
            all_clear.extend([ends[1]-pair[0],pair[1]-ends[4],ends[2]-pair[2],pair[3]-ends[5]])
    record('B2c 两根杆均插入两座、孔四周间隙',near(all_clear,.2) and near(engagements,41) and max(collisions)<VOL_TOL,
           f'16 处间隙 {min(all_clear):.4f}–{max(all_clear):.4f}；每端插入 {min(engagements):.2f} mm；最大布尔交集 {max(collisions):.7f} mm³')
    for name,m in [('左',left),('右',right)]:
        hz=intervals(m,[35,116,0],2)
        passage=intervals(m,[35,0,4.7],1)
        record('B2d '+name+'后端锁孔',near(hz[2]-hz[1],2.8) and near(passage[-1],101.8),
               f'孔径 {hz[2]-hz[1]:.4f}，中心 Z={(hz[2]+hz[1])/2:.4f}，中心射线从后端通至后孔前壁 Y={passage[-1]:.4f} mm')
    for name,m in [('B1 左座',left),('B1 右座',right)]:
        found=audit_bridges(m,name,[('前连杆孔',7.4,(29,31),1,12.4,12.4),
                                    ('后连杆孔',7.4,(107,109),1,12.4,12.4)])
        record(name+' 水平面清单完整',found,'床面外为两个连杆孔。后端 Ø2.8 横孔顶部为圆弧斜面。')
        plane_check(m,name[-2:])
    found=audit_bridges(bar,'B1 连杆',[])
    record('B1 连杆水平面清单',found,'除热床接触面外无水平朝下面。')
    files=('stand_left.stl','stand_right.stl','tie_bar.stl')
    same=[(STL/f).read_bytes()==(ROOT/'out_v2'/'stl'/f).read_bytes() for f in files]
    record('B1 支架打印几何与 v2 比对',all(same),f'左座/右座/连杆 STL 逐字节相同：{same}')
    log.append('    托面结论：两版 STL 相同，托面平面偏差及法向量见上。v1 工程的两个托座均为平直着色，无自定义法向。平直着色重渲后的 B2 仍有菱形纹。对照渲染中只去掉透明演示平板与托面重合的背面后，菱形纹消失，确认是共面渲染伪影。v2 采用平直着色、32 采样；实体假平板与托座的位置保持贴合。')

def main():
    log.extend(['电容笔夹与平板支架 | v3 独立 STL 验收（长度 mm，体积 mm³）',
                '输入：out/stl 的打印坐标网格、inputs/finger_link7_mm.stl。名义尺寸来自任务书及 v2、v3 修改单。八边形隧道及 V 槽按指定坐标保留。',
                '三角网格法向/连通/水密由 trimesh 验证，自交面由 MeshLab 检查；布尔使用 Manifold 双精度。',
                f'数值判零阈值：长度 {DIM_TOL:g} mm，体积 {VOL_TOL:g} mm³。侧隙、倾角按任务书公差。',''])
    check_holder()
    log.append('')
    check_stand()
    log.append(f'\n合计 {sum(results)} PASS，{len(results)-sum(results)} FAIL。')
    report='\n'.join(log)+'\n'
    (OUT/'check_report.txt').write_text(report)
    print(report,flush=True)
    if not all(results): sys.exit(1)

if __name__=='__main__':
    main()
import os, sys; sys.stdout.flush(); os._exit(0)
