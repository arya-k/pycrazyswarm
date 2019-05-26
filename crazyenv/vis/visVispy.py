"""
Visualizer used by all envs to see the environment.
Uses the high performance, GPU accelerated library Vispy
to allow for interactive viewing in all 3 dimensions.

Supports multiple drones, moving obstacles, labelled points
and a breakcrumbs trail behind a single drone to visualize the
path taken.
"""

import os
import math
import numpy as np
import random

from vispy import scene, app, io
from vispy.color import Color
from vispy.visuals import transforms
from vispy.scene.cameras import TurntableCamera


def get_color_names():  # limit the number of colors for obstacles to ones that can easily be seen
    return ['crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue']


class VisVispy:
    def __init__(self, hidden=False):  # hidden=true useful for speed when creating video
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1900, 1145), show=not hidden, config=dict(
            samples=4), resizable=True, always_on_top=True, bgcolor='white', vsync=True)

        # Set up a viewbox to display the cube with interactive arcball
        self.view = self.canvas.central_widget.add_view()
        self.view.bgcolor = '#efefef'
        self.view.camera = TurntableCamera(
            fov=60.0, elevation=30.0, azimuth=280.0)

        # add a colored 3D axis for orientation
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        self.cfs = []
        self.spheres = []
        self.crumbs = []
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.transform = transforms.STTransform()
        self.obstacles = []
        # ground = scene.visuals.Plane(6.0, 6.0, direction="+z", color=(0.3, 0.3, 0.3, 1.0), parent=self.view.scene)

    def update(self, t, crazyflies, spheres=[], obstacles=[], crumb=None):
        if len(self.cfs) == 0:  # add the crazyflies if they don't exist already
            verts, faces, normals, nothin = io.read_mesh(
                os.path.join(os.path.dirname(__file__), "crazyflie2.obj.gz"))
            for i in range(0, len(crazyflies)):
                mesh = scene.visuals.Mesh(
                    vertices=verts, shading='smooth', faces=faces, parent=self.view.scene)
                mesh.transform = transforms.MatrixTransform()
                self.cfs.append(mesh)

        if crumb is not None:  # add breadcrumb trails behind drones if requested.
            if len(self.crumbs) == 0 or np.linalg.norm(self.crumbs[-1]-crumb) > 0.05:
                # Only add them if drone is sufficiently far from the last breakcrumb.
                self.crumbs.append(crumb)
                self.markers.set_data(
                    np.array(self.crumbs), size=5, face_color='black', edge_color='black')
        elif self.crumbs:
            self.markers.set_data(np.array([[1e10, 1e10]]))
            self.crumbs = []

        if spheres:  # sphere: (position, color)
            if not self.spheres:
                for pos, color in spheres:
                    self.spheres.append(scene.visuals.Sphere(
                        radius=.02, color=color, parent=self.view.scene))
                    self.spheres[-1].transform = transforms.STTransform(
                        translate=pos)
            for i, (pos, color) in enumerate(spheres):
                self.spheres[i].transform.translate = pos

        if obstacles:  # cube : (position, size)
            if not self.obstacles:
                for pos, size, _ in obstacles:
                    self.obstacles.append(scene.visuals.Cube(
                        size=size, color=random.choice(get_color_names()), parent=self.view.scene))
                    self.obstacles[-1].transform = transforms.STTransform(
                        translate=pos)
            for i, (pos, size, _) in enumerate(obstacles):
                self.obstacles[i].transform.translate = pos

        # update the location of the crazyflies if they have changed
        for i in range(0, len(self.cfs)):
            x, y, z = crazyflies[i].position(t)
            roll, pitch, yaw = crazyflies[i].rpy(t)
            self.cfs[i].transform.reset()
            self.cfs[i].transform.rotate(90, (1, 0, 0))
            self.cfs[i].transform.rotate(math.degrees(roll), (1, 0, 0))
            self.cfs[i].transform.rotate(math.degrees(pitch), (0, 1, 0))
            self.cfs[i].transform.rotate(math.degrees(yaw), (0, 0, 1))
            self.cfs[i].transform.scale((0.001, 0.001, 0.001))
            self.cfs[i].transform.translate((x, y, z))

        self.canvas.app.process_events()  # redraw the canvas

    def close(self):  # we propogate close functions up to this point.
        self.canvas.close()
