'''
Class to work with coordinate transforms that come off of fiducial markers, for example Aruco Markers:
    # detects marker and gets cornors
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frameRight, aruco_dict, parameters=parameters)
    # how you get rotation and translation vectors
    rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker, mtx, distortion)
The functions are losely built around inputing 3x3 Rotation matrix and 3x1 translation vector into kinematic
transform chains

'''
import numpy as np
import cv2


class Xfm:
    '''
    This class will create and handle request about 4x4 projective transform in a list of transforms
    from fiducial a1 to aN as seen from the camera: Tcam_a1, Tcam_a2, ..., Tcam_aN
    Usages:
    "cam" means camera; Tx_y means projecting x to y (y in frame x); inverse(Tx_y) = Ty_x
    Xfm is formed from the "R" 3x3 rotation matrix and the "t" 3x1 translation matrix
    R3x3 = [[x0, x1, x2]   t3x1 = [X,Y,Z]^T (transpose)
            [y0, y1, y2]
            [z0, z1, z2]]

    T4x4 = [R | t]  = [ [ x0, x1, x2, X]
           [0 | 1]      [ y0, y1, y2, Y]
                        [ z0, z1, z2, Z]
                        [  0,  0,  0, 1]
    USAGE:
    This class will handle a chain of 1 or more transforms. If there are 4 Aruco tags with IDs 1,2,3,4
    Then inputing The camera to tag transforms in a list of
    Camera to ArucoID transforms tcs = [Tc_0, Tc_1,Tc_2,Tc_3,Tc_4] and list of IDs tids = [0,1,2,3,4]:
       Xfm(tcs,tids)
    will call functions you can call separately Xfm.make_T_chains(tcs) and Xfm.make_T_ids(tids) which
    will create a chain of Transforms and inverse transforms in the class:
        Xfm.Tchain[T0_1,T1_2,T2_3,T3_4]
        Xfm.Tchain_inv[T1_0,T2_1,T3_2,T4_3]
        Xfm.Tids[[0,1],[1,2],[2,3],[3,4]] #and
        Xfm.Tids_inv[[1,0],[2,1],[3,2],[4,3]]
    You can then call
        Tx_y(start,stop) #to return any single or joint multiple transform in either direction
    '''

    ################### CLASS INIT AND SETTING OF MEMBERS ###################

    '''
     You can pass a vector of 4x4 Transforms from camera to each seen fiducial/aruco target
     and/or a vector of aruco IDs. If not, they default to zero lists [].
     Args:
         T_cams (): 4x4 Transform from camera to Aruco fiducial (or None Default)
         T_ids (): Id of Aruco fiducial (or None)
     '''
    def __init__(self, rvecs = None, tvecs = None, ids = None, ids_to_choose=None, T_cams = None, T_ids = None):
        '''
        Constructor to set up transform chains
        Args:
            rvecs ():
            tvecs ():
            ids ():
            ids_to_choose ():
            T_cams ():
            T_ids ():
        '''

        if rvecs is not None and tvecs is not None and ids is not None and ids_to_choose is not None:
            # Squeeze everything just in case there are undeeded dimensions
            rvecs = np.squeeze(rvecs)
            tvecs = np.squeeze(tvecs)
            ids = np.squeeze(ids)
            T_cams = [] #Fill these
            T_ids = []
            for idx in ids_to_choose:
                i = ids_to_choose.index(idx)
                T = self.compose_Rt(rvecs[i],tvecs[i]) #This will auto-convet a (3,1) vect into 3x3 if necessary
                T_cams.append(T)
                T_ids.append(idx)
        if T_cams:
            self.make_T_chain(T_cams)
        else:
            self.Tchain = []
            self.Tchain_inv = []
        if T_ids:
            self.make_T_IDs(T_ids)
        else:
            self.Tids = []
            self.Tids_inv = []  # inverted order of the list

    def make_T_chain(self,T_cams):
        '''
        Take one or more Tcam_a1,Tcam_a2, Tcam_aN and turns it into a chain of Ta1_a2,Ta2_a3,...TaN-1_Tan
        The len(T_chain) that results from above is len(T_cams) - 1 because we go in pairs
           Also, remember that inverse(Tcam_a1) = Ta1_cam, or: Tcam_a1^-1 = Ta1_cam
        Effects:
           given Tcams = [tcam_a, tcam_b, tcam_c, tcam_d]
           self.Tchain = [Ta_b,Tb_c,Tc_d] and self.Tchain_inv = [Tb_a,Tc_b,Td_c]
        Args:
            T_cams (): list of numpy 4x4 T, transform vectors T_from_to of T_cam_aruco
        Returns: Sets self.Tchain and self.Tchain_inv internally, returns len(Tchain) externally
        '''
        self.Tchain = []
        self.Tchain_inv = []
        for i in range(len(T_cams)-1):
            self.Tchain.append(self.invertT(T_cams[i]).dot(T_cams[i+1]))
            self.Tchain_inv.append(self.invertT(self.Tchain[i]))
        return len(self.Tchain)

    def make_T_IDs(self, T_ids):
        '''
        Append in list of aruco IDs that are in the same order as the arucos seen by the camera in make_T_chain
        It will form a list of pairs. If ids are a1, a2, a3, a4, the list will be
        Tids = [[a1,a2],[a2,a3],[a3,a4]]; These pairs are in the order of frame [from, to]
        Args:
            T_ids: list of aruco numerical IDs as seen from camera
        Returns: len(self.Tids)
           sets self.Tids as above and self.Tids_inv = [[a2,a1],[a3,a2],[a4,a3]]
        '''
        self.Tids = []
        self.Tids_inv = [] #inverted order of the list
        for i in range(len(T_ids) - 1):
            self.Tids.append([T_ids[i], T_ids[i+1]])
            self.Tids_inv.append([T_ids[i+1],T_ids[i]])
        return len(self.Tids)

    ################### GETTING TRANSFORMS ###################
    def Tidx_idy(self,from_,to):
        '''
        Give a frame Aruco ID "from_" and an Acruco ID "to", return the associated transform between those frames
        Note: This just calls self.find_start_stop and then self.Tx_y()
        Args:
            from_ (): Aruco ID of the reference frame (Aruco ID of coordinate system in transform chain
            to ():    Aruco ID of where you want the transform to
        Returns: The 4x4 Transform from reference frame to the desired frame  Tfrom_to
        '''
        start,stop,dir = self.find_start_stop(from_,to)
        return self.Tx_y(start,stop,dir)
    def Tx_y(self, start, stop, inclusive=True):
        '''
        Will return the computed transform from start to stop in in step=1 or step=-1 as appropriate
        as one "do it all" 4x4 Transform.
        Since we are usually doing the entire transform by which we mean start 2 to 3 means Tchain[2].dot(Tchain[3])
        so by default inclusive = True in order to do that (otherwise it would be just Tchain[2]. So:
        * If you want to go from index 3 to 1, then it will produce the transforms (note it "hits" 1):
            Tchain_inv[3].dot(Tchain_inv[2]).dot(Tchain_inv[1])
        * If you want 2 to 3, then Tx_y(2,3) will yield
            Tchain[2].dot(Tchain[3]) #Note it "hits" 3
        * if you set inclusive=False, start and stop will behave exactly as in the python range() function
        * As a convienance:
        * TO RETRIEVE A SINGLE TRANSFORM TA_B:
            * set start=stop, then inclusive=True => Tchain[start] and inclusive=False => Tchain_inv[start] e.g.:
               * T from 0 to 1: Tx_y(0,0) will yield T0_1 = Tchain[0]
               * T from 1 to 0: T1_0(0,0,False) will yield T1_0 = Tchain_inv[0]
               * T3_2: Tx_y(2,2,False) will yield T3_2 = Tchain_inv[2]
               * T2_3: Tx_y(2,2) will yield T2_3 = Tchain[2]
        Args:
            start (): Start index such as in the range() function (must be in range of Tchain)
            stop (): Stop index such as in the range() function (must be in range of Tchain)
            inclusive (): We generally want to include the stop index. By default (True) it does include it
                          For example by default (inclusive=True) start = 1, stop = 2 produces range(1,3,1)
                              if inclusive is False, start = 3, stop = 2 produces range(3,2,-1)
                              inclusive adds one to stop if range goes up subtracts one if range goes down
        Returns:
            T (the joint transform from start to stop, usually inclusive)
        '''
        # PROTECT AGAINST NO TRANSORM and BAD INPUTS
        if len(self.Tchain) == 0:
            return np.identity(4, dtype=float)
        # PROTECT AGAINST BAD INPUTS
        len_Tchain = len(self.Tchain)
        # make positive
        if start < 0:
            start = len_Tchain + start
        if stop < 0:
            stop = len_Tchain + stop
        if start >= len_Tchain or stop >= len_Tchain or start < 0 or stop < 0:
            raise ValueError("Start {} or Stop {} is < 0 or > Tchain sie {}".format(start,stop,len_Tchain))
        # handle return of single elements
        if start == stop:
            if inclusive: #Return Tcurrent_next
                return self.Tchain[start]
            else:         #Return Tnext_current
                return self.Tchain_inv[start]
        # SET STEP:
        step = 1
        if start > stop: #Then got to step down
            step = -1
        # set for inclusive
        inc = step   # if inclusive, we need to expand the range by one step
        if not inclusive:
            inc = 0
        # PROCESS
        T = np.identity(4,dtype=float)
        if step < 0:
            for i in range(start,stop+inc,step):
                T = T.dot(self.Tchain_inv[i])
        else:
            for i in range(start,stop+inc,step):
                T = T.dot(self.Tchain[i])
        return T

    def find_start_stop(self,from_,to):
        '''
        Once you have set up Tchain and Tids, AND assuming Aruco IDs are unique in the scene
        you may use this to find the start, stop indicies and direction,
        Theindicies in
            Tchain (if start ends up < stop) or
            Tchain_inv (if start > stop)
        Args:
            from_ (): Aruco ID of the reference frame (Aruco ID of coordinate system in transform chain
            to ():    Aruco ID of where you want the transform to
        Returns: start, stop indexes and direction (True => forward or Tchain, False back or Tchain_inv
        '''
        start = -1
        stop = -1
        forward = True
        if len(self.Tids) == 0 or len(self.Tchain) == 0:
            return start, stop, forward #nothing to find
        #Find matching ids
        for i, id in zip(range(len(self.Tids)), self.Tids):
            if start < 0 and from_ == id[0] or from_ == id[1]:
                start = i
            if stop < 0 and to == id[0] or to == id[1]:
                stop = i
            if start == stop: #This is a single transform, look to the ID order
                if from_ == id[1]: #Reverse order
                    forward = False
            if start >= 0 and stop >= 0:
                break
        if start < 0 or stop < 0:
            return -1,-1,forward #Didn't find both fiducials
        # Adjust for forward direction (start < stop) or backwards direction (start > stop)
        if start < stop: #Forward direction
            start += 1
        elif start > stop: #Back
            stop += 1
            forward = False
        return start, stop, forward

    ################### UTILITY FUNCTIONS ###################
    def type_np(self,v):
        '''
        Test if numpy array
        Args:
            v: array
        Returns:
            True if numpy array, else False
        '''
        if type(v) != np.ndarray:
            return False
        return True

    def shape_check(self,v,size):
        '''
        Test if v is of size "size"
        Args:
            v: np array
            size (): (3,), (3,3) whatever size v should be
        Returns:
            True if v.shape is size, else False
        '''
        if v.shape != size:
            return False
        return True

    def invertT(self,T):
        '''
        Invert 4x4 numpy transform matrix T
        Args:
            T (): 4x4 float numpy Transform matrix
        Returns: T^-1
        '''
        if not self.type_np(T) or not self.shape_check(T,(4,4)):
            raise ValueError("T {} {} not numpy array or 4x4".format(type(T),T.shape))
        return np.linalg.inv(T)

    def compose_Rt(self,R, t):
        '''
        Compose a (3x1) or (3,) or 3x3 R rotation, and 3x1 t translation into a 4x4 transfrom T
        Args:
            R (): Rotation 3x3 numpy array, or if it's a (3,1) or (3,) rotation vector, autoconvert into 3x3 numpy_array
            t (): translation 3x1 numpy array
        Returns:
            4x4 T(R,t); if you passed in non-numpy array or wrong sizes, then returns
        '''
        # Safety check
        if type(R) != np.ndarray or type(t) != np.ndarray:
            raise ValueError("R and/or t was not an np.array: type(R):{}, type(t){}".format(type(R),type(t)))
        R = np.squeeze(R)
        #If R is a vector (3,1) => (3,) after squeeze, turn it int a 3x3
        if R.shape == (3,):
            R = cv2.Rodrigues(R)[0] #Convert rotation vector into 3x3 rotation matrix
        t = np.squeeze(t)
        if R.shape != (3,3) or t.shape != (3,):
            raise ValueError("R and/or t wrong size R_3x3:{}, t_3:{}".format(R.shape,t.shape))
        # Compose R,t to T
        T = np.zeros((4, 4), dtype=float)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        T[3, 3] = 1.0
        return T
    def p3_to_4(self,p):
        '''
        convert 3x1 point (3,) to 4x1 (4,) homogenous point
        Args:
            p (): (3,) point
        Returns:  p(x,y,z) -> p(x,y,z,1)
        '''
        four1s = np.ones(4, dtype=float)
        four1s[0:3] = p
        return four1s


########################################################################
# For bring up tests
def main():
    Tc_x = np.array([[0,1,0,0],[-1,0,0,2],[0,0,1,4],[0,0,0,1]],dtype=float)
    Tc_y = np.array([[0, 0, -1, -1],[-1, 0, 0, 3],[0, 1, 0, 3],[0, 0, 0, 1]],dtype=float)
    Tc_z = np.array([[0, 0, -1, -2],[0 ,-1, 0, 2],[-1, 0, 0, 2],[0, 0, 0 ,1]],dtype=float)
    Tc_a = np.array([[0, 0, -1, 4],[0, -1, 0, 3],[-1, 0, 0, 2],[0, 0, 0, 1]],dtype=float)
    Tc_b = np.array([[1, 0, 0, -5],[0, 0, 1, 0],[0, -1, 0, 2],[0, 0, 0 ,1]],dtype=float)
    Tc_g = np.array([[0, -1, 0, -4],[0, 0, -1, 1],[1, 0, 0, 4],[0, 0, 0, 1]],dtype=float)
    Tcamera = [Tc_x,Tc_y,Tc_z,Tc_a,Tc_b,Tc_g]
    Tids = ['x','y','z','a','b','g']
    T4x4 = Xfm(None,None,None,None,Tcamera, Tids)
    print("T(2,2):")
    print(T4x4.Tx_y(2,2))
    print("T(2,2,False)")
    print(T4x4.Tx_y(2,2,False))
    start,stop,dir = T4x4.find_start_stop('y' ,'a')
    print("y_a start {}, stop {}, forward {}".format(start,stop,dir))
    start,stop,dir = T4x4.find_start_stop('a', 'y')
    print("a_y start {}, stop {}, forward {}".format(start,stop,dir))
    start,stop,dir = T4x4.find_start_stop('z' ,'a')
    print("z_a start {}, stop {}, forward {}".format(start,stop,dir))
    start,stop,dir = T4x4.find_start_stop('a', 'z')
    print("a_z start {}, stop {}, forward {}".format(start,stop,dir))

    for id,t in zip(T4x4.Tids,T4x4.Tchain):
        print("\nid:{}:\n{}".format(id,t))

    start,stop,dir = T4x4.find_start_stop('a','g')
    print("a_g -- start {}, stop {},  dir {}".format(start,stop,dir))
    Tabc = T4x4.Tx_y(start,stop)
    print("Tabc\n{}\n".format(Tabc))

    start,stop,dir = T4x4.find_start_stop('b','y')
    print("b_y -- start {}, stop {},  dir {}".format(start,stop,dir))
    Tb_y = T4x4.Tx_y(start,stop)
    print("Tb_y\n{}\n".format(Tb_y))

if __name__ == "__main__":
    main()
