"""Pure quaternion/vector transforms shared by geometry and ROS pose adapters."""
import math

def rotate(q, v):
    w, x, y, z = q
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if norm < 1e-10:
        raise ValueError('Invalid pose quaternion')
    w, x, y, z = (t/norm for t in q)
    uv = (y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0])
    uuv = (y*uv[2]-z*uv[1], z*uv[0]-x*uv[2], x*uv[1]-y*uv[0])
    return [v[i]+2*(w*uv[i]+uuv[i]) for i in range(3)]


def multiply(a, b):
    w,x,y,z = a; v,i,j,k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]

