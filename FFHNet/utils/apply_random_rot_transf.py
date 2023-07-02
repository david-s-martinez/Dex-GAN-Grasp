import numpy as np
import time

def __R_x(angle):
    """Generate random rotational matrix with angle (0,57) deg
    """
    return np.array([[1,0,0],
                    [0,np.cos(angle),-np.sin(angle)],
                    [0,np.sin(angle),np.cos(angle)]])
    
def __R_y(angle):
    return np.array([[np.cos(angle),0,np.sin(angle)],
                    [0,1,0],
                    [-np.sin(angle),0,np.cos(angle)]])

def __R_z(angle):
    return np.array([[np.cos(angle),-np.sin(angle),0],
                    [np.sin(angle),np.cos(angle),0],
                    [0,0,1]]) 
    
def generate_random_rot_max():
    a = time.time()
    alpha, beta, gamma = np.random.rand(3)
    rot_x = __R_x(alpha)
    rot_y = __R_y(beta)
    rot_z = __R_z(gamma)
    R = np.matmul(rot_z,rot_y)
    R = np.matmul(R,rot_x)
    print(time.time()-a, ' s')
    return R

if __name__ == "__main__":
    R = generate_random_rot_max()
    print(R)