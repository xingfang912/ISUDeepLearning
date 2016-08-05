#!/usr/bin/env python

import time, copy, threading
import pickle
import rospy
import roslib
from std_msgs.msg import String
#roslib.load_manifest('mirage')
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
import tf
import cv2

class NeuroViz(object):

    def __init__(self):
        rospy.init_node('neuroviz', anonymous=False)
        while (rospy.get_rostime() == 0.0):
            pass
        # =========================
        self.lock = threading.Lock()
        self.tfli = tf.TransformListener()
        self.tfbr = tf.TransformBroadcaster()
        self.markerArray = MarkerArray()
        #self.markerArray.header.frameid = '/my_frame'
        self.markerNames = [] # used to prevent duplicate markers.
        self.next_marker_id = 0
        self.pub_markers = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1)
        self.refresh_thread = None
        self.lock2 = threading.Lock()
        self.get_transform_threads = {} # threads working on getting a transform.
        self.lock3 = threading.Lock()
        self.transforms = {} # dictionary of requested transforms. A buffer.
        self.run()

    def run(self):
        self.refresh_thread = threading.Thread(target=self.markerRefreshThread)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()
        # ============================
        img = cv2.imread('/home/ryan/Desktop/Poster/ncat_symbol_16.bmp')
        #for i in range(0, 10):
        #    for j in range(0, 10):
        #        for p in range(0, 10):
        #            self.add_ball(i,3+p,j)
        self.remove_all_markers()
        pts = self.add_picture(img, 0, 0, 1)
        pts2 = self.add_tensor(0,10,0, 16,2,16)
        pts3 = self.add_tensor(0,15,0, 16,2,16)
        pts4 = self.add_tensor(4,20,4, 8,2,8, pooling=True)
        pts5 = self.add_tensor(4,25,4, 8,2,8)
        pts6 = self.add_tensor(4,30,4, 8,2,8)
        pts7 = self.add_tensor(6,35,6, 4,2,4, pooling=True)
        pts8 = self.add_tensor(6,40,6, 4,2,4)
        pts9 = self.add_tensor(6,45,6, 4,2,4)
        pts10 = self.add_tensor(6,50,6, 4,2,4) # fully connected layer
        pts11 = self.add_tensor(3,55,8, 10,1,1) # softmax layer
        self.max_pool(pts3, pts4, 0,15,0, 16,2,16, 4,20,4, 8,2,8)
        self.max_pool(pts6, pts7, 4,30,4, 8,2,8, 6,35,6, 4,2,4)
        #self.convolve(pts, pts2)
        #self.convolve(pts2, pts3)
        #self.convolve(pts4, pts5)
        #self.convolve(pts5, pts6)
        #self.convolve(pts7, pts8)
        #self.convolve(pts8, pts9)
        #self.fully_connect(pts9, pts10)
        #self.fully_connect(pts10, pts11)
        self.final_output(pts11)
        time.sleep(10.0)

    def markerRefreshThread(self):
        while (True):
            self.refresh_markers()
            time.sleep(0.5) # 2 Hz

    def refresh_markers(self):
        with self.lock:
            self.pub_markers.publish(self.markerArray)
            
    def join(self):
        self.refresh_thread.join(1.0)
    
    def shutdown(self):
        self.join()

    # x,y,z position. w,h,d = width, height, depth
    def add_tensor(self, x, y, z, x_, y_, z_, pooling=False):
        pts = []
        for x__ in range(0, x_):
            for y__ in range(0, y_):
                for z__ in range(0, z_):
                    if pooling:
                        pts.append(self.add_cube(x+x__,y+y__,z+z__))
                    else:
                        pts.append(self.add_ball(x+x__,y+y__,z+z__))
        if not pooling:
            self.add_cube(x+(x_/2.0)-0.5,y+(y_/2.0)-0.5,z+(z_/2.0)-0.5, x_,y_,z_, 0.0, 1.0, 0.0, 0.3)
        return pts

    def convolve(self, pts, pts2):
        colors = [(0.0,0.0,1.0), (0.0,1.0,1.0), (1.0,0.0,0.0), (1.0,0.0,1.0), (1.0,1.0,0.0), (1.0,1.0,1.0), (0.5,1.0,0.5), (0.0,0.5,1.0), (0.0,0.8,0.0)]
        color = 0
        for pt in pts:
            for pt2 in pts2:
                if abs(pt[0] - pt2[0]) <= 1.0 and abs(pt[2] - pt2[2]) <= 1.0:
                    self.add_arrow(pt, pt2, colors[color][0], colors[color][1], colors[color][2])
                    color = (color + 1) % len(colors)

    def max_pool(self, pts, pts2, x,y,z, x_,y_,z_, x2,y2,z2, x2_,y2_,z2_):
        for pt in pts:
            xr, yr, zr = pt[0]-x, pt[1]-y, pt[2]-z
            xt, zt = x+xr/2+x_/4, z+zr/2+z_/4
            self.add_arrow(pt, (xt, y+5, zt))

    def fully_connect(self, pts, pts2):
        for pt in pts:
            for pt2 in pts2:
                self.add_arrow(pt, pt2)

    def final_output(self, pts):
        for pt in pts:
            self.add_arrow(pt, (pt[0], pt[1]+2, pt[2]), fat=True)

    def add_picture(self, img, x, y, z, splay=True):
        pts = []
        x_, y_, z_ = 0, 0, 0
        if len(img.shape) == 3:
            (x_, y_, z_) = img.shape
        else:
            (x_, y_) = img.shape
        for i in range(0, x_):
            for j in range(0, y_):
                self.add_cube(x+i,z,y+j, 0.9,0.9,0.9, img[-j,i,2]/256.0,img[-j,i,1]/256.0,img[-j,i,0]/256.0,1.0)
                self.add_cube(x+i,z+2,y+j, 0.9,0.9,0.9, img[-j,i,0]/256.0,img[-j,i,0]/256.0,img[-j,i,0]/256.0,0.7)
                self.add_cube(x+i,z+3,y+j, 0.9,0.9,0.9, img[-j,i,1]/256.0,img[-j,i,1]/256.0,img[-j,i,1]/256.0,0.7)
                pts.append(self.add_cube(x+i,z+4,y+j, 0.9,0.9,0.9, img[-j,i,2]/256.0,img[-j,i,2]/256.0,img[-j,i,2]/256.0,0.7))
        return pts

    def add_cube(self, x, y, z, sx=0.9,sy=0.9,sz=0.9, r=0.65, g=0.65, b=0.65, a=1.0):
        m = Marker()
        m.id = self.next_marker_id
        self.next_marker_id += 1
        m.header.frame_id = '/map'
        m.action = Marker.ADD
        m.type = Marker.CUBE
        m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        #m.pose.position.x, m.pose.position.y, m.pose.position.z = -0.01, -0.04, -0.125/2.0
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        with self.lock:
            self.markerArray.markers.append(m)
        return (x,y,z)

    def add_ball(self, x, y, z, r=0.0, g=0.0, b=1.0, a=1.0):
        m = Marker()
        m.id = self.next_marker_id
        self.next_marker_id += 1
        m.header.frame_id = '/map'
        m.action = Marker.ADD
        m.type = Marker.SPHERE
        m.scale.x, m.scale.y, m.scale.z = 0.8, 0.8, 0.8
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        #m.pose.position.x, m.pose.position.y, m.pose.position.z = -0.01, -0.04, -0.125/2.0
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        with self.lock:
            self.markerArray.markers.append(m)
        return (x,y,z)

    def add_arrow(self, (x,y,z), (x2,y2,z2), r=0.3, g=0.7, b=0.3, a=1.0, fat=False):
        m = Marker()
        m.id = self.next_marker_id
        self.next_marker_id += 1
        m.header.frame_id = '/map'
        m.action = Marker.ADD
        m.type = Marker.ARROW
        scale = 0.05
        if fat:
            scale = 0.15
        m.scale.x, m.scale.y, m.scale.z = scale, scale, scale
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        #m.pose.position.x, m.pose.position.y, m.pose.position.z = -0.01, -0.04, -0.125/2.0
        m.points.append(Point(x,y,z))
        m.points.append(Point(x2,y2,z2))
        with self.lock:
            self.markerArray.markers.append(m)
        return ((x,y,z),(x2,y2,z2))

    def remove_all_markers(self):
        with self.lock:
            for m in self.markerArray.markers:
                m.action = Marker.DELETE
            self.pub_markers.publish(self.markerArray)
            self.markerArray.markers = []
            self.markerNames = []

if __name__ == '__main__':
    try:
        NeuroViz()
    except rospy.ROSInterruptException:
        pass
