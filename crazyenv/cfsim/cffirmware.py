# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_cffirmware', [dirname(__file__)])
        except ImportError:
            import _cffirmware
            return _cffirmware
        if fp is not None:
            try:
                _mod = imp.load_module('_cffirmware', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _cffirmware = swig_import_helper()
    del swig_import_helper
else:
    import _cffirmware
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



_cffirmware.M_PI_F_swigconstant(_cffirmware)
M_PI_F = _cffirmware.M_PI_F

def fsqr(x):
    return _cffirmware.fsqr(x)
fsqr = _cffirmware.fsqr

def radians(degrees):
    return _cffirmware.radians(degrees)
radians = _cffirmware.radians

def degrees(radians):
    return _cffirmware.degrees(radians)
degrees = _cffirmware.degrees
class vec(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vec, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vec, name)
    __repr__ = _swig_repr
    __swig_setmethods__["x"] = _cffirmware.vec_x_set
    __swig_getmethods__["x"] = _cffirmware.vec_x_get
    if _newclass:
        x = _swig_property(_cffirmware.vec_x_get, _cffirmware.vec_x_set)
    __swig_setmethods__["y"] = _cffirmware.vec_y_set
    __swig_getmethods__["y"] = _cffirmware.vec_y_get
    if _newclass:
        y = _swig_property(_cffirmware.vec_y_get, _cffirmware.vec_y_set)
    __swig_setmethods__["z"] = _cffirmware.vec_z_set
    __swig_getmethods__["z"] = _cffirmware.vec_z_get
    if _newclass:
        z = _swig_property(_cffirmware.vec_z_get, _cffirmware.vec_z_set)

    def __repr__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z)


    def __init__(self):
        this = _cffirmware.new_vec()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_vec
    __del__ = lambda self: None
vec_swigregister = _cffirmware.vec_swigregister
vec_swigregister(vec)


def mkvec(x, y, z):
    return _cffirmware.mkvec(x, y, z)
mkvec = _cffirmware.mkvec

def vrepeat(x):
    return _cffirmware.vrepeat(x)
vrepeat = _cffirmware.vrepeat

def vzero():
    return _cffirmware.vzero()
vzero = _cffirmware.vzero

def vscl(s, v):
    return _cffirmware.vscl(s, v)
vscl = _cffirmware.vscl

def vneg(v):
    return _cffirmware.vneg(v)
vneg = _cffirmware.vneg

def vdiv(v, s):
    return _cffirmware.vdiv(v, s)
vdiv = _cffirmware.vdiv

def vadd(a, b):
    return _cffirmware.vadd(a, b)
vadd = _cffirmware.vadd

def vsub(a, b):
    return _cffirmware.vsub(a, b)
vsub = _cffirmware.vsub

def vdot(a, b):
    return _cffirmware.vdot(a, b)
vdot = _cffirmware.vdot

def vmag2(v):
    return _cffirmware.vmag2(v)
vmag2 = _cffirmware.vmag2

def vmag(v):
    return _cffirmware.vmag(v)
vmag = _cffirmware.vmag

def vdist2(a, b):
    return _cffirmware.vdist2(a, b)
vdist2 = _cffirmware.vdist2

def vdist(a, b):
    return _cffirmware.vdist(a, b)
vdist = _cffirmware.vdist

def vnormalize(v):
    return _cffirmware.vnormalize(v)
vnormalize = _cffirmware.vnormalize

def vcross(a, b):
    return _cffirmware.vcross(a, b)
vcross = _cffirmware.vcross

def vprojectunit(a, b_unit):
    return _cffirmware.vprojectunit(a, b_unit)
vprojectunit = _cffirmware.vprojectunit

def vorthunit(a, b_unit):
    return _cffirmware.vorthunit(a, b_unit)
vorthunit = _cffirmware.vorthunit

def vabs(v):
    return _cffirmware.vabs(v)
vabs = _cffirmware.vabs

def vmin(a, b):
    return _cffirmware.vmin(a, b)
vmin = _cffirmware.vmin

def vmax(a, b):
    return _cffirmware.vmax(a, b)
vmax = _cffirmware.vmax

def vminkowski(v):
    return _cffirmware.vminkowski(v)
vminkowski = _cffirmware.vminkowski

def veq(a, b):
    return _cffirmware.veq(a, b)
veq = _cffirmware.veq

def vneq(a, b):
    return _cffirmware.vneq(a, b)
vneq = _cffirmware.vneq

def vless(a, b):
    return _cffirmware.vless(a, b)
vless = _cffirmware.vless

def vleq(a, b):
    return _cffirmware.vleq(a, b)
vleq = _cffirmware.vleq

def vgreater(a, b):
    return _cffirmware.vgreater(a, b)
vgreater = _cffirmware.vgreater

def vgeq(a, b):
    return _cffirmware.vgeq(a, b)
vgeq = _cffirmware.vgeq

def visnan(v):
    return _cffirmware.visnan(v)
visnan = _cffirmware.visnan

def vadd3(a, b, c):
    return _cffirmware.vadd3(a, b, c)
vadd3 = _cffirmware.vadd3

def vsub2(a, b, c):
    return _cffirmware.vsub2(a, b, c)
vsub2 = _cffirmware.vsub2

def vload(d):
    return _cffirmware.vload(d)
vload = _cffirmware.vload

def vstore(v, d):
    return _cffirmware.vstore(v, d)
vstore = _cffirmware.vstore

def vloadf(f):
    return _cffirmware.vloadf(f)
vloadf = _cffirmware.vloadf

def vstoref(v, f):
    return _cffirmware.vstoref(v, f)
vstoref = _cffirmware.vstoref
class mat33(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, mat33, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, mat33, name)
    __repr__ = _swig_repr
    __swig_setmethods__["m"] = _cffirmware.mat33_m_set
    __swig_getmethods__["m"] = _cffirmware.mat33_m_get
    if _newclass:
        m = _swig_property(_cffirmware.mat33_m_get, _cffirmware.mat33_m_set)

    def __init__(self):
        this = _cffirmware.new_mat33()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_mat33
    __del__ = lambda self: None
mat33_swigregister = _cffirmware.mat33_swigregister
mat33_swigregister(mat33)


def mzero():
    return _cffirmware.mzero()
mzero = _cffirmware.mzero

def diag(a, b, c):
    return _cffirmware.diag(a, b, c)
diag = _cffirmware.diag

def eyescl(a):
    return _cffirmware.eyescl(a)
eyescl = _cffirmware.eyescl

def eye():
    return _cffirmware.eye()
eye = _cffirmware.eye

def mcolumns(a, b, c):
    return _cffirmware.mcolumns(a, b, c)
mcolumns = _cffirmware.mcolumns

def mrows(a, b, c):
    return _cffirmware.mrows(a, b, c)
mrows = _cffirmware.mrows

def mcrossmat(v):
    return _cffirmware.mcrossmat(v)
mcrossmat = _cffirmware.mcrossmat

def mcolumn(m, col):
    return _cffirmware.mcolumn(m, col)
mcolumn = _cffirmware.mcolumn

def mrow(m, row):
    return _cffirmware.mrow(m, row)
mrow = _cffirmware.mrow

def mtranspose(m):
    return _cffirmware.mtranspose(m)
mtranspose = _cffirmware.mtranspose

def mscale(s, a):
    return _cffirmware.mscale(s, a)
mscale = _cffirmware.mscale

def mneg(a):
    return _cffirmware.mneg(a)
mneg = _cffirmware.mneg

def madd(a, b):
    return _cffirmware.madd(a, b)
madd = _cffirmware.madd

def msub(a, b):
    return _cffirmware.msub(a, b)
msub = _cffirmware.msub

def mvmult(a, v):
    return _cffirmware.mvmult(a, v)
mvmult = _cffirmware.mvmult

def mmult(a, b):
    return _cffirmware.mmult(a, b)
mmult = _cffirmware.mmult

def maddridge(a, d):
    return _cffirmware.maddridge(a, d)
maddridge = _cffirmware.maddridge

def misnan(m):
    return _cffirmware.misnan(m)
misnan = _cffirmware.misnan

def set_block33_rowmaj(block, stride, m):
    return _cffirmware.set_block33_rowmaj(block, stride, m)
set_block33_rowmaj = _cffirmware.set_block33_rowmaj
class quat(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, quat, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, quat, name)
    __repr__ = _swig_repr
    __swig_setmethods__["x"] = _cffirmware.quat_x_set
    __swig_getmethods__["x"] = _cffirmware.quat_x_get
    if _newclass:
        x = _swig_property(_cffirmware.quat_x_get, _cffirmware.quat_x_set)
    __swig_setmethods__["y"] = _cffirmware.quat_y_set
    __swig_getmethods__["y"] = _cffirmware.quat_y_get
    if _newclass:
        y = _swig_property(_cffirmware.quat_y_get, _cffirmware.quat_y_set)
    __swig_setmethods__["z"] = _cffirmware.quat_z_set
    __swig_getmethods__["z"] = _cffirmware.quat_z_get
    if _newclass:
        z = _swig_property(_cffirmware.quat_z_get, _cffirmware.quat_z_set)
    __swig_setmethods__["w"] = _cffirmware.quat_w_set
    __swig_getmethods__["w"] = _cffirmware.quat_w_get
    if _newclass:
        w = _swig_property(_cffirmware.quat_w_get, _cffirmware.quat_w_set)

    def __init__(self):
        this = _cffirmware.new_quat()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_quat
    __del__ = lambda self: None
quat_swigregister = _cffirmware.quat_swigregister
quat_swigregister(quat)


def mkquat(x, y, z, w):
    return _cffirmware.mkquat(x, y, z, w)
mkquat = _cffirmware.mkquat

def quatvw(v, w):
    return _cffirmware.quatvw(v, w)
quatvw = _cffirmware.quatvw

def qeye():
    return _cffirmware.qeye()
qeye = _cffirmware.qeye

def qaxisangle(axis, angle):
    return _cffirmware.qaxisangle(axis, angle)
qaxisangle = _cffirmware.qaxisangle

def quataxis(q):
    return _cffirmware.quataxis(q)
quataxis = _cffirmware.quataxis

def quatangle(q):
    return _cffirmware.quatangle(q)
quatangle = _cffirmware.quatangle

def rpy2quat_small(rpy):
    return _cffirmware.rpy2quat_small(rpy)
rpy2quat_small = _cffirmware.rpy2quat_small

def quatimagpart(q):
    return _cffirmware.quatimagpart(q)
quatimagpart = _cffirmware.quatimagpart

def quat2rpy(q):
    return _cffirmware.quat2rpy(q)
quat2rpy = _cffirmware.quat2rpy

def quat2axis(q):
    return _cffirmware.quat2axis(q)
quat2axis = _cffirmware.quat2axis

def quat2angle(q):
    return _cffirmware.quat2angle(q)
quat2angle = _cffirmware.quat2angle

def quat2rotmat(q):
    return _cffirmware.quat2rotmat(q)
quat2rotmat = _cffirmware.quat2rotmat

def qvrot(q, v):
    return _cffirmware.qvrot(q, v)
qvrot = _cffirmware.qvrot

def qqmul(q, p):
    return _cffirmware.qqmul(q, p)
qqmul = _cffirmware.qqmul

def qinv(q):
    return _cffirmware.qinv(q)
qinv = _cffirmware.qinv

def qneg(q):
    return _cffirmware.qneg(q)
qneg = _cffirmware.qneg

def qdot(a, b):
    return _cffirmware.qdot(a, b)
qdot = _cffirmware.qdot

def qnormalize(q):
    return _cffirmware.qnormalize(q)
qnormalize = _cffirmware.qnormalize

def quat_gyro_update(quat, gyro, dt):
    return _cffirmware.quat_gyro_update(quat, gyro, dt)
quat_gyro_update = _cffirmware.quat_gyro_update

def qnlerp(a, b, t):
    return _cffirmware.qnlerp(a, b, t)
qnlerp = _cffirmware.qnlerp

def qslerp(a, b, t):
    return _cffirmware.qslerp(a, b, t)
qslerp = _cffirmware.qslerp

def qload(d):
    return _cffirmware.qload(d)
qload = _cffirmware.qload

def qstore(q, d):
    return _cffirmware.qstore(q, d)
qstore = _cffirmware.qstore

def qloadf(f):
    return _cffirmware.qloadf(f)
qloadf = _cffirmware.qloadf

def qstoref(q, f):
    return _cffirmware.qstoref(q, f)
qstoref = _cffirmware.qstoref

_cffirmware.PP_DEGREE_swigconstant(_cffirmware)
PP_DEGREE = _cffirmware.PP_DEGREE

_cffirmware.PP_SIZE_swigconstant(_cffirmware)
PP_SIZE = _cffirmware.PP_SIZE

def polyval(p, t):
    return _cffirmware.polyval(p, t)
polyval = _cffirmware.polyval

def polylinear(p, duration, x0, x1):
    return _cffirmware.polylinear(p, duration, x0, x1)
polylinear = _cffirmware.polylinear

def poly5(poly, T, x0, dx0, ddx0, xf, dxf, ddxf):
    return _cffirmware.poly5(poly, T, x0, dx0, ddx0, xf, dxf, ddxf)
poly5 = _cffirmware.poly5

def polyscale(p, s):
    return _cffirmware.polyscale(p, s)
polyscale = _cffirmware.polyscale

def polyder(p):
    return _cffirmware.polyder(p)
polyder = _cffirmware.polyder

def polystretchtime(p, s):
    return _cffirmware.polystretchtime(p, s)
polystretchtime = _cffirmware.polystretchtime

def polyreflect(p):
    return _cffirmware.polyreflect(p)
polyreflect = _cffirmware.polyreflect
class poly4d(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, poly4d, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, poly4d, name)
    __repr__ = _swig_repr
    __swig_setmethods__["p"] = _cffirmware.poly4d_p_set
    __swig_getmethods__["p"] = _cffirmware.poly4d_p_get
    if _newclass:
        p = _swig_property(_cffirmware.poly4d_p_get, _cffirmware.poly4d_p_set)
    __swig_setmethods__["duration"] = _cffirmware.poly4d_duration_set
    __swig_getmethods__["duration"] = _cffirmware.poly4d_duration_get
    if _newclass:
        duration = _swig_property(_cffirmware.poly4d_duration_get, _cffirmware.poly4d_duration_set)

    def __init__(self):
        this = _cffirmware.new_poly4d()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_poly4d
    __del__ = lambda self: None
poly4d_swigregister = _cffirmware.poly4d_swigregister
poly4d_swigregister(poly4d)


def poly4d_zero(duration):
    return _cffirmware.poly4d_zero(duration)
poly4d_zero = _cffirmware.poly4d_zero

def poly4d_linear(duration, p0, p1, yaw0, yaw1):
    return _cffirmware.poly4d_linear(duration, p0, p1, yaw0, yaw1)
poly4d_linear = _cffirmware.poly4d_linear

def poly4d_scale(p, x, y, z, yaw):
    return _cffirmware.poly4d_scale(p, x, y, z, yaw)
poly4d_scale = _cffirmware.poly4d_scale

def poly4d_shift(p, x, y, z, yaw):
    return _cffirmware.poly4d_shift(p, x, y, z, yaw)
poly4d_shift = _cffirmware.poly4d_shift

def poly4d_shift_vec(p, pos, yaw):
    return _cffirmware.poly4d_shift_vec(p, pos, yaw)
poly4d_shift_vec = _cffirmware.poly4d_shift_vec

def poly4d_stretchtime(p, s):
    return _cffirmware.poly4d_stretchtime(p, s)
poly4d_stretchtime = _cffirmware.poly4d_stretchtime

def polyder4d(p):
    return _cffirmware.polyder4d(p)
polyder4d = _cffirmware.polyder4d

def poly4d_max_accel_approx(p):
    return _cffirmware.poly4d_max_accel_approx(p)
poly4d_max_accel_approx = _cffirmware.poly4d_max_accel_approx
class traj_eval(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, traj_eval, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, traj_eval, name)
    __repr__ = _swig_repr
    __swig_setmethods__["pos"] = _cffirmware.traj_eval_pos_set
    __swig_getmethods__["pos"] = _cffirmware.traj_eval_pos_get
    if _newclass:
        pos = _swig_property(_cffirmware.traj_eval_pos_get, _cffirmware.traj_eval_pos_set)
    __swig_setmethods__["vel"] = _cffirmware.traj_eval_vel_set
    __swig_getmethods__["vel"] = _cffirmware.traj_eval_vel_get
    if _newclass:
        vel = _swig_property(_cffirmware.traj_eval_vel_get, _cffirmware.traj_eval_vel_set)
    __swig_setmethods__["acc"] = _cffirmware.traj_eval_acc_set
    __swig_getmethods__["acc"] = _cffirmware.traj_eval_acc_get
    if _newclass:
        acc = _swig_property(_cffirmware.traj_eval_acc_get, _cffirmware.traj_eval_acc_set)
    __swig_setmethods__["omega"] = _cffirmware.traj_eval_omega_set
    __swig_getmethods__["omega"] = _cffirmware.traj_eval_omega_get
    if _newclass:
        omega = _swig_property(_cffirmware.traj_eval_omega_get, _cffirmware.traj_eval_omega_set)
    __swig_setmethods__["yaw"] = _cffirmware.traj_eval_yaw_set
    __swig_getmethods__["yaw"] = _cffirmware.traj_eval_yaw_get
    if _newclass:
        yaw = _swig_property(_cffirmware.traj_eval_yaw_get, _cffirmware.traj_eval_yaw_set)

    def __init__(self):
        this = _cffirmware.new_traj_eval()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_traj_eval
    __del__ = lambda self: None
traj_eval_swigregister = _cffirmware.traj_eval_swigregister
traj_eval_swigregister(traj_eval)


def traj_eval_invalid():
    return _cffirmware.traj_eval_invalid()
traj_eval_invalid = _cffirmware.traj_eval_invalid

def is_traj_eval_valid(ev):
    return _cffirmware.is_traj_eval_valid(ev)
is_traj_eval_valid = _cffirmware.is_traj_eval_valid

def poly4d_eval(p, t):
    return _cffirmware.poly4d_eval(p, t)
poly4d_eval = _cffirmware.poly4d_eval
class piecewise_traj(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, piecewise_traj, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, piecewise_traj, name)
    __repr__ = _swig_repr
    __swig_setmethods__["t_begin"] = _cffirmware.piecewise_traj_t_begin_set
    __swig_getmethods__["t_begin"] = _cffirmware.piecewise_traj_t_begin_get
    if _newclass:
        t_begin = _swig_property(_cffirmware.piecewise_traj_t_begin_get, _cffirmware.piecewise_traj_t_begin_set)
    __swig_setmethods__["timescale"] = _cffirmware.piecewise_traj_timescale_set
    __swig_getmethods__["timescale"] = _cffirmware.piecewise_traj_timescale_get
    if _newclass:
        timescale = _swig_property(_cffirmware.piecewise_traj_timescale_get, _cffirmware.piecewise_traj_timescale_set)
    __swig_setmethods__["shift"] = _cffirmware.piecewise_traj_shift_set
    __swig_getmethods__["shift"] = _cffirmware.piecewise_traj_shift_get
    if _newclass:
        shift = _swig_property(_cffirmware.piecewise_traj_shift_get, _cffirmware.piecewise_traj_shift_set)
    __swig_setmethods__["n_pieces"] = _cffirmware.piecewise_traj_n_pieces_set
    __swig_getmethods__["n_pieces"] = _cffirmware.piecewise_traj_n_pieces_get
    if _newclass:
        n_pieces = _swig_property(_cffirmware.piecewise_traj_n_pieces_get, _cffirmware.piecewise_traj_n_pieces_set)
    __swig_setmethods__["pieces"] = _cffirmware.piecewise_traj_pieces_set
    __swig_getmethods__["pieces"] = _cffirmware.piecewise_traj_pieces_get
    if _newclass:
        pieces = _swig_property(_cffirmware.piecewise_traj_pieces_get, _cffirmware.piecewise_traj_pieces_set)

    def __init__(self):
        this = _cffirmware.new_piecewise_traj()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_piecewise_traj
    __del__ = lambda self: None
piecewise_traj_swigregister = _cffirmware.piecewise_traj_swigregister
piecewise_traj_swigregister(piecewise_traj)


def piecewise_duration(pp):
    return _cffirmware.piecewise_duration(pp)
piecewise_duration = _cffirmware.piecewise_duration

def piecewise_plan_5th_order(p, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1):
    return _cffirmware.piecewise_plan_5th_order(p, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1)
piecewise_plan_5th_order = _cffirmware.piecewise_plan_5th_order

def piecewise_plan_7th_order_no_jerk(p, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1):
    return _cffirmware.piecewise_plan_7th_order_no_jerk(p, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1)
piecewise_plan_7th_order_no_jerk = _cffirmware.piecewise_plan_7th_order_no_jerk

def piecewise_eval(traj, t):
    return _cffirmware.piecewise_eval(traj, t)
piecewise_eval = _cffirmware.piecewise_eval

def piecewise_eval_reversed(traj, t):
    return _cffirmware.piecewise_eval_reversed(traj, t)
piecewise_eval_reversed = _cffirmware.piecewise_eval_reversed

def piecewise_is_finished(traj, t):
    return _cffirmware.piecewise_is_finished(traj, t)
piecewise_is_finished = _cffirmware.piecewise_is_finished

_cffirmware.TRAJECTORY_STATE_IDLE_swigconstant(_cffirmware)
TRAJECTORY_STATE_IDLE = _cffirmware.TRAJECTORY_STATE_IDLE

_cffirmware.TRAJECTORY_STATE_FLYING_swigconstant(_cffirmware)
TRAJECTORY_STATE_FLYING = _cffirmware.TRAJECTORY_STATE_FLYING

_cffirmware.TRAJECTORY_STATE_LANDING_swigconstant(_cffirmware)
TRAJECTORY_STATE_LANDING = _cffirmware.TRAJECTORY_STATE_LANDING
class planner(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, planner, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, planner, name)
    __repr__ = _swig_repr
    __swig_setmethods__["state"] = _cffirmware.planner_state_set
    __swig_getmethods__["state"] = _cffirmware.planner_state_get
    if _newclass:
        state = _swig_property(_cffirmware.planner_state_get, _cffirmware.planner_state_set)
    __swig_setmethods__["reversed"] = _cffirmware.planner_reversed_set
    __swig_getmethods__["reversed"] = _cffirmware.planner_reversed_get
    if _newclass:
        reversed = _swig_property(_cffirmware.planner_reversed_get, _cffirmware.planner_reversed_set)
    __swig_setmethods__["trajectory"] = _cffirmware.planner_trajectory_set
    __swig_getmethods__["trajectory"] = _cffirmware.planner_trajectory_get
    if _newclass:
        trajectory = _swig_property(_cffirmware.planner_trajectory_get, _cffirmware.planner_trajectory_set)
    __swig_setmethods__["planned_trajectory"] = _cffirmware.planner_planned_trajectory_set
    __swig_getmethods__["planned_trajectory"] = _cffirmware.planner_planned_trajectory_get
    if _newclass:
        planned_trajectory = _swig_property(_cffirmware.planner_planned_trajectory_get, _cffirmware.planner_planned_trajectory_set)
    __swig_setmethods__["pieces"] = _cffirmware.planner_pieces_set
    __swig_getmethods__["pieces"] = _cffirmware.planner_pieces_get
    if _newclass:
        pieces = _swig_property(_cffirmware.planner_pieces_get, _cffirmware.planner_pieces_set)

    def __init__(self):
        this = _cffirmware.new_planner()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cffirmware.delete_planner
    __del__ = lambda self: None
planner_swigregister = _cffirmware.planner_swigregister
planner_swigregister(planner)


def plan_init(p):
    return _cffirmware.plan_init(p)
plan_init = _cffirmware.plan_init

def plan_stop(p):
    return _cffirmware.plan_stop(p)
plan_stop = _cffirmware.plan_stop

def plan_is_stopped(p):
    return _cffirmware.plan_is_stopped(p)
plan_is_stopped = _cffirmware.plan_is_stopped

def plan_current_goal(p, t):
    return _cffirmware.plan_current_goal(p, t)
plan_current_goal = _cffirmware.plan_current_goal

def plan_takeoff(p, pos, yaw, height, duration, t):
    return _cffirmware.plan_takeoff(p, pos, yaw, height, duration, t)
plan_takeoff = _cffirmware.plan_takeoff

def plan_land(p, pos, yaw, height, duration, t):
    return _cffirmware.plan_land(p, pos, yaw, height, duration, t)
plan_land = _cffirmware.plan_land

def plan_go_to(p, relative, hover_pos, hover_yaw, duration, t):
    return _cffirmware.plan_go_to(p, relative, hover_pos, hover_yaw, duration, t)
plan_go_to = _cffirmware.plan_go_to

def plan_start_trajectory(p, trajectory, reversed):
    return _cffirmware.plan_start_trajectory(p, trajectory, reversed)
plan_start_trajectory = _cffirmware.plan_start_trajectory

_cffirmware.TRAJECTORY_FIGURE8_swigconstant(_cffirmware)
TRAJECTORY_FIGURE8 = _cffirmware.TRAJECTORY_FIGURE8

def poly4d_set(poly, dim, coef, val):
    return _cffirmware.poly4d_set(poly, dim, coef, val)
poly4d_set = _cffirmware.poly4d_set

def poly4d_get(poly, dim, coef):
    return _cffirmware.poly4d_get(poly, dim, coef)
poly4d_get = _cffirmware.poly4d_get

def pp_get_piece(pp, i):
    return _cffirmware.pp_get_piece(pp, i)
pp_get_piece = _cffirmware.pp_get_piece

def malloc_poly4d(size):
    return _cffirmware.malloc_poly4d(size)
malloc_poly4d = _cffirmware.malloc_poly4d
# This file is compatible with both classic and new-style classes.

