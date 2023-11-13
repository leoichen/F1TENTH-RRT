#!/usr/bin/env python3

"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math
from math import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion

# TODO: import as you need
from math import *

# class def for tree nodes
# It's up to you if you want to use this
class TreeNode(object):
    def __init__(self, x, y, node_id):
        self.x = x
        self.y = y
        self.node_id = node_id
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')

        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "ego_racecar/odom"
        scan_topic = "/scan"
        vis_one_topic = "/vis"
        vis_topic = "/vis_array"
        drive_topic = "/drive"
        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)
        self.pose_sub_

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        self.scan_sub_

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.vis_pub_ = self.create_publisher(MarkerArray, vis_topic, 10)
        self.vis_one_pub_ = self.create_publisher(Marker, vis_one_topic, 10)

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.grid_width = 6 # y
        self.grid_height = 4 # x
        self.num_cells = 8 # keep this even because +1 in occ_grid)
        self.occ_grid = np.zeros((self.num_cells+1, self.num_cells+1))
        self.occ_xGrid = np.linspace(0, self.grid_height, self.num_cells+1)[::-1]
        self.occ_yGrid = np.linspace(-self.grid_width/2, self.grid_width/2, self.num_cells+1)

        # self.occ_grid = [0] * (self.grid_width * self.grid_height)
        self.tree = []
        self.marker_id = 0
        self.curr_x = 0
        self.curr_y = 0
        self.curr_theta = 0
        self.marker_array = MarkerArray()
        self.collision_points = 100
        self.min_angle = -30*np.pi/180
        self.max_angle = -self.min_angle
        self.poseNotCollected = True
        self.step_size = 1.
        self.goal_x = 0
        self.goal_y = 0
        self.goal_theta = 0
        self.cwd = '/sim_ws/src/pure_pursuit/scripts/'
        wps = np.genfromtxt(self.cwd+'levine_blocked_waypoints.csv', delimiter = ',')
        uniques = np.unique(wps, axis = 0, return_index = True)
        idxs = np.sort(uniques[1])
        self.unique_wps = wps[idxs]
        self.unique_xys = self.unique_wps[:,:2]
        self.width_leftBound = -0.75
        self.width_rightBound = 0.75
        self.dist_threshold = .6
        self.node_id = 0
        self.wp_x = 0
        self.wp_y = 0

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:
        """


        # ensure pose has been collected first --> do I really need this?
        if self.poseNotCollected:
            return

        temp_xGrid = np.linspace(0, self.grid_height, self.num_cells+1)[::-1]
        temp_yGrid = np.linspace(-self.grid_width/2,\
             self.grid_width/2, self.num_cells+1)


        #self.occ_xGrid = temp_xGrid*cos(self.curr_theta) - temp_yGrid*sin(self.curr_theta)
        #self.occ_yGrid = temp_xGrid*sin(self.curr_theta) + temp_yGrid*cos(self.curr_theta)

        self.occ_xGrid = temp_xGrid
        self.occ_yGrid = temp_yGrid

        #self.occ_xGrid = temp_xGrid*cos(self.curr_theta)
        #self.occ_yGrid = temp_yGrid*sin(self.curr_theta)

        # reset occupancy grid
        self.occ_grid = np.zeros((self.num_cells+1, self.num_cells+1)) # do i need this?

        # get range and angle data
        range_data = np.array(scan_msg.ranges)
        angles_arr = np.linspace(scan_msg.angle_min , \
            scan_msg.angle_max, len(range_data)) # [rad]

        # update occupancy grid
        for i in range(len(range_data)):
            
            # check if angle is in range
            # if (angles_arr[i] < self.min_angle or angles_arr[i] > self.max_angle):
            #     continue

            # calculate x, y (in car frame)
            scanCar_x = range_data[i]*cos(angles_arr[i])
            scanCar_y = range_data[i]*sin(angles_arr[i])

            # check if cell is within in occupancy grid
            if ((scanCar_x <= self.grid_height) and (np.abs(scanCar_y) <= self.grid_width/2)):
                # convert to occ_grid frame
                pass

            mapped_x_idx = np.argmin(np.abs(list(map(lambda x: scanCar_x - x, self.occ_xGrid))))
            mapped_y_idx = np.argmin(np.abs(list(map(lambda y: scanCar_y - y, self.occ_yGrid))))

            # set occ_grid to occupied
            self.occ_grid[mapped_x_idx][mapped_y_idx] = 1

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        
        max_iterations = 10
        self.curr_x = pose_msg.pose.pose.position.x
        self.curr_y = pose_msg.pose.pose.position.y
        self.curr_theta = self.get_yaw(pose_msg)
        # use poseNotCollected to ensure pose_callback runs 
        # before scan_callback
        self.poseNotCollected = False

        # get current odometry
        for i in range(max_iterations):
            self.marker_id = 0
            self.marker_array.markers = []

            self.curr_x = pose_msg.pose.pose.position.x
            self.curr_y = pose_msg.pose.pose.position.y
            self.curr_theta = self.get_yaw(pose_msg)

            # define goal
            self.goal_x, self.goal_y, self.goal_theta = self.get_goal()
            self.marker_array.markers.append(self.create_marker(self.goal_x, \
                    self.goal_y, 4))
            #print('current goal', self.goal_x, self.goal_y)

            # draw grid and wps
            # self.draw_grid()
            # self.draw_waypoints()
            # self.draw_tree()

            # set root node on first_callback (i.e., when tree is empty)
            if not self.tree:
                curr_node = TreeNode(self.curr_x, self.curr_y, self.node_id)
                self.node_id += 1
                curr_node.is_root = True
                self.tree.append(curr_node)
                #print('Root node added into self.tree', curr_node.x, curr_node.y)


            # sample random point
            sampled_point = self.sample()
            #print('sampled point', sampled_point[0], sampled_point[1])

            # find nearest node in tree to sampled_point
            nearest_node_on_tree = self.nearest(sampled_point)
            #print('nearest_node_on_tree', nearest_node_on_tree.x,\
               # nearest_node_on_tree.y)
            new_node = self.steer(nearest_node_on_tree, sampled_point)
            #print('new node', new_node.x, new_node.y)

            # see if path to nearest_node is collison free
            if self.check_collision(nearest_node_on_tree, new_node):
                
                # set new_node's parent as nearest_node_on_tree
                new_node.parent = nearest_node_on_tree
                new_node.x = self.wp_x
                new_node.y = self.wp_y
                #print('new node id', new_node.node_id, new_node.parent.node_id)
                self.node_id += 1
                self.tree.append(new_node)
                
                #print('No collision')
                #print('node added to tree', new_node.x, new_node.y)
                #print('Add', self.tree)
                print('wp', self.wp_x, self.wp_y)
                self.drive_to_node(new_node)
                break
                if self.is_goal(new_node, self.goal_x, self.goal_y):
                    # print('Found path')
                    #path = self.find_path(self.tree, new_node)
                    # self.drive_to_node(new_node)
                    # break
                    pass
        

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """

        # need to make sure sampled point is not occupied

        # sample in car frame within occ grid
        self.width_leftBound = -self.grid_width/2 - self.curr_y
        self.width_rightBound = self.grid_width/2 - self.curr_y
        keepSampling = True
        while(keepSampling):
            # print('Sampling')
            
            x = np.random.uniform(low=0*self.grid_height, high= self.grid_height)
            y = np.random.uniform(low=-self.grid_width/2, high=self.grid_width/2)

            # convert goal to car frame
            goal_car_x = self.goal_x - self.curr_x
            goal_car_y = self.goal_y - self.curr_y

            #x = np.random.uniform(low = goal_car_x - self.dist_threshold, high = goal_car_x + 2*self.dist_threshold)
            #y = np.random.uniform(low = goal_car_y - self.dist_threshold, high = goal_car_y + 2*self.dist_threshold)

            # create a marker for this
            # self.vis_one_pub_.publish(self.create_marker(x+self.curr_x\
            #     , y+self.curr_y, 2))
            
            # if point is occupied, sample again
            if self.isPointOccupied(x, y):
                continue

            # point not occupied, keep this sampled point
            else:
                keepSampling = False
                break
            
        # convert to map frame
        temp_x = x*cos(self.curr_theta)-y*sin(self.curr_theta)
        temp_y = x*sin(self.curr_theta)+y*sin(self.curr_theta)

        final_x = self.curr_x + x
        final_y = self.curr_y + y

        # print('final', final_x, final_y)
        return (final_x, final_y)

    def nearest(self, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """

        # calc euclidian distance to get nearest node
        sampled_point = np.array(sampled_point)
        euc_dist_arr = []

        euc_dists = list(map(lambda node: np.sqrt(np.sum(np.square([node.x, node.y]\
             - sampled_point))), self.tree))

        # find and return nearest node
        nearest_node_on_tree = np.argmin(euc_dists)
        return self.tree[nearest_node_on_tree]


    def steer(self, nearest_node_on_tree, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """

        # find the distance between the nearest_node_on_tree and sampled_point
        diff_x = sampled_point[0] - nearest_node_on_tree.x
        diff_y = sampled_point[1] - nearest_node_on_tree.y
        distance = np.sqrt(diff_x**2 + diff_y**2)
        #print('distance', distance)

        # check if sampled point is within self.step_size 
        if distance <= self.step_size:
            new_node = TreeNode(sampled_point[0], sampled_point[1], self.node_id)
        else:
            steering_angle = atan2(diff_y, diff_x) - self.curr_theta
            new_dist_x = nearest_node_on_tree.x + self.step_size*cos(steering_angle)
            new_dist_y = nearest_node_on_tree.y + self.step_size*sin(steering_angle)
            new_node = TreeNode(new_dist_x, new_dist_y, self.node_id)
        #print('new node in steer', new_node.x, new_node.y)
        
        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """

        # convert both new_node and nearest_node to car frame
        # return True
        new_node_x_car = new_node.x - self.curr_x
        new_node_y_car = new_node.y - self.curr_y
        nearest_node_x_car = nearest_node.x - self.curr_x
        nearest_node_y_car = nearest_node.y - self.curr_y

        # calculate the increment in path
        x_inc = (new_node_x_car - nearest_node_x_car) / self.collision_points
        y_inc = (new_node_y_car - nearest_node_y_car) / self.collision_points

        # set up curr variables
        cur_x = nearest_node_x_car
        cur_y = nearest_node_y_car

        for i in range(self.collision_points):
            #print(i)
            # increment step
            cur_x += x_inc
            cur_y += y_inc
            #print('cur', cur_x, cur_y)

            if (cur_x > self.grid_height or np.abs(cur_y) > self.grid_width):
                continue
            
            # convert to occ_grid frame
            mapped_x_idx = np.argmin(np.abs(list(map(lambda x: cur_x - x, self.occ_xGrid))))
            mapped_y_idx = np.argmin(np.abs(list(map(lambda y: cur_y - y, self.occ_yGrid))))
            #print('x grid', self.occ_xGrid)
            #print('y grid', self.occ_yGrid)
            #print('mapped', mapped_x_idx, mapped_y_idx)
            #print(self.occ_grid)
            #print('result', self.occ_grid[mapped_x_idx][mapped_y_idx])
            
            # check if curr position is occupied and return False if it is
            if self.occ_grid[mapped_x_idx][mapped_y_idx] == 1:
                # print('Collision Exists')
                return False
            
        # return True if path is not occupied
        return True

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enough to the goal
        """
        #print('goals', goal_x, goal_y)
        dist = np.sqrt((goal_x - latest_added_node.x) ** 2 + \
            (goal_y - latest_added_node.y) ** 2)

        #print('is_goal dist', dist)

        return dist < self.dist_threshold

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        current_node = latest_added_node
        # keep adding nodes to path until I get to the root node
        foundRoot = False
        while current_node.parent is not None:
            #print('id', current_node.node_id, current_node.parent.node_id)
            # if current_node.is_root:
            #     foundRoot = True
            #     path.append(current_node)
            #     break
            path.append(current_node)
            current_node = current_node.parent
            
        # return path in reverse so it starts at root node
        return path



    # ===============================================================
     # my own functions

    def get_goal(self):
        
        # find current goal point 10 m away from current position
        goal_x = self.curr_x + (self.step_size-0.2)*cos(self.curr_theta)
        goal_y = self.curr_y + (self.step_size-0.2)*sin(self.curr_theta)#0#.np.random.uniform(self.width_leftBound, 0.3*self.width_rightBound)

        # find the closest waypoint
        # calculate euclidian distance of each wp to the cars current position + lookahead distance
        euc_dists = np.sqrt(np.sum((self.unique_xys - \
            [goal_x, goal_y]) ** 2, axis = 1))
        
        # find argmin of euc_dists
        min_dist_idx = np.argmin(euc_dists)

        # find wp closest to odom_l_x and odom_l_y and record it
        wp_x = self.unique_xys[min_dist_idx][0]
        wp_y = self.unique_xys[min_dist_idx][1]
        wp_theta = atan2(wp_y, wp_x)

        self.wp_x = wp_x
        self.wp_y= wp_y
        # self.goal_theta = wp_theta

        return wp_x, wp_y, wp_theta

    def drive_to_node(self, node):
        
        diff_x = node.x - self.curr_x
        diff_y = node.y - self.curr_y

        print('diff', diff_x, diff_y)
        # calculate steering angle
        transformed_x = diff_x*cos(self.curr_theta)+diff_y*sin(self.curr_theta)
        transformed_y = -diff_x*sin(self.curr_theta)+diff_y*cos(self.curr_theta)
        steering_angle = 2*transformed_y / (np.hypot(transformed_x, transformed_y) ** 2)
       # steering_angle = atan2(transformed_y, transformed_x) #- self.curr_theta

        #steering_angle = 0.5 * (2 * node.y) / L ** 2
        #steering_angle = atan2(diff_y, diff_x) - self.curr_theta
        # print('steering', atan2(diff_y, diff_x) * 180/np.pi)
        print('steering_angle', steering_angle * 180/np.pi)
        # clip steering_angle   
        steering_angle = np.clip(steering_angle, -20*np.pi/180, 20*np.pi/180)
        print('steering_angle clipped', steering_angle*180/np.pi)

        # publish to /drive
        drive = AckermannDrive(speed = .4, \
                steering_angle = steering_angle)
        drive_msg = AckermannDriveStamped(drive = drive)
        self.drive_pub_.publish(drive_msg)

    def isPointOccupied(self, sampled_x, sampled_y):

        mapped_sampled_x_idx = np.argmin(np.abs(list(map(lambda x: sampled_x - x, self.occ_xGrid))))
        mapped_sampled_y_idx = np.argmin(np.abs(list(map(lambda y: sampled_y - y, self.occ_yGrid))))

        return self.occ_grid[mapped_sampled_x_idx][mapped_sampled_y_idx]


    def draw_tree(self):
        tree_draw_arr = []

        for i in range(len(self.tree)):
            curr_node = self.tree[i]
            if i == len(self.tree) - 1:
                self.marker_array.markers.append(self.create_marker(curr_node.x, \
                    curr_node.y, 3))
            else:
                self.marker_array.markers.append(self.create_marker(curr_node.x, \
                curr_node.y, 4))

        self.vis_pub_.publish(self.marker_array)

    def draw_waypoints(self):
        wp_marker_array = MarkerArray()

        # draw waypoints
        for [x,y] in self.unique_xys:
            wp_marker_array.markers.append(self.create_marker(x, y, 3))
        
        # draw all waypoints
        self.vis_pub_.publish(wp_marker_array)

    def draw_grid(self):
       
        # empty markers array
        # self.marker_id = 0

        grid_shape = np.shape(self.occ_grid)
        # for each point on the occupany grid create a marker with the correct color
        # blue -> not occupied; red -> occupied
        for i in range(np.shape(self.occ_grid)[0]):
            for j in range(np.shape(self.occ_grid)[1]):
                marker_x = self.curr_x + self.occ_xGrid[i]
                marker_y = self.curr_y + self.occ_yGrid[j]

                # create marker and add to marker_array
                self.marker_array.markers.append(self.create_marker(marker_x, \
                    marker_y, self.occ_grid[i][j]))

        # draw the entire grid
        self.vis_pub_.publish(self.marker_array)

    def create_marker(self, x, y, occupied = 0):
        self.marker_id += 1
        # set default colors as if unoccupied
        red = 0.
        blue = 1.
        green = 0.
        marker_type = Marker.CUBE
        y_scale = .1
        if occupied == 1:
            red = 1.
            blue = 0.
        elif occupied == 2: # sampled point
            red = 0.
            green = 1.
            blue = 0.
        elif occupied == 3: # unoccupied waypoint
            red = 1.
            green = 0.
            blue = 0.
            marker_type = Marker.SPHERE
        elif occupied == 4: # goal waypoints
            red = 0.
            green = 1.0
            blue = 0.
            marker_type = Marker.SPHERE
            y_scale = .1

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.id = self.marker_id
        marker.type = marker_type
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.scale.x = y_scale
        marker.scale.y = y_scale
        marker.scale.z = .1
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.
        marker.lifetime.sec = 1000
        
        self.marker_id += 1
        
        return marker
        

    def get_yaw(self, pose_msg):

        # for [x,y] in self.unique_xys:
        #     self.draw_marker(x, y, self.id, 'r')
        #     self.id += 1

        quaternion = np.array([pose_msg.pose.pose.orientation.x, 
                        pose_msg.pose.pose.orientation.y, 
                        pose_msg.pose.pose.orientation.z, 
                        pose_msg.pose.pose.orientation.w])

        euler = euler_from_quaternion(quaternion)

        return euler[2]




    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
