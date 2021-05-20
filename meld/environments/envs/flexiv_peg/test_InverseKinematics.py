import pdb

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
import numpy as np
import pdb
import random

mjlib = mjbindings.mjlib

FlexivPeg_XML = assets.get_contents(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/assets/graphics/robot_chain.xml')

_SITE_NAME = 'ee_site'
_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
_TOL = 1e-14
_MAX_STEPS = 100
_MAX_RESETS = 10


_TARGETS = [
    # target_pos              # target_quat
    (np.array([0., 0., 0.3]), np.array([0., 1., 0., 1.])),
    (np.array([-0.5, 0., 0.5]), None),
    (np.array([0., 0., 0.8]), np.array([0., 1., 0., 1.])),
    (np.array([0., 0., 0.8]), None),
    (np.array([0., -0.1, 0.5]), None),
    (np.array([0., -0.1, 0.5]), np.array([1., 1., 0., 0.])),
    (np.array([0.5, 0., 0.5]), None),
    (np.array([0.4, 0.1, 0.5]), None),
    (np.array([0.4, 0.1, 0.5]), np.array([1., 0., 0., 0.])),
    (np.array([0., 0., 0.3]), None),
    (np.array([0., 0.5, -0.2]), None),
    (np.array([0.5, 0., 0.3]), np.array([1., 0., 0., 1.])),
    (None, np.array([1., 0., 0., 1.])),
    (None, np.array([0., 1., 0., 1.])),
]
_INPLACE = [False, True]



class _ResetArm:

  def __init__(self, seed=None):
    self._rng = np.random.RandomState(seed)
    self._lower = None
    self._upper = None

  def _cache_bounds(self, physics):
    self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
    limited = physics.named.model.jnt_limited[_JOINTS].astype(np.bool)
    # Positions for hinge joints without limits are sampled between 0 and 2pi
    self._lower[~limited] = 0
    self._upper[~limited] = 2 * np.pi

  def __call__(self, physics):
    if self._lower is None:
      self._cache_bounds(physics)
    # NB: This won't work for joints with > 1 DOF
    new_qpos = self._rng.uniform(self._lower, self._upper)
    physics.named.data.qpos[_JOINTS] = new_qpos





class my_env(parameterized.TestCase):
    def __init__(self):
        # super(lab_env, self).__init__(env)
        # 导入xml文档
        self.model = load_model_from_path("assets/simpleEE_4box.xml")
        # 调用MjSim构建一个basic simulation
        self.sim = MjSim(model=self.model)
        self.sim = MjSim(self.model)

        self.viewer = MjViewer(self.sim)
        self.viewer._run_speed = 0.001
        self.timestep = 0
        # Sawyer Peg
        #self.init_qpos = np.array([-0.305, -0.83, 0.06086, 1.70464, -0.02976, 0.62496, -0.04712])
        # Flexiv Peg
        self.init_qpos = np.array([-0.22, -0.43, 0.449, -2, -0.25, 0.799, 0.99])

        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = self.init_qpos[i]
        self.testQposFromSitePose(
            (np.array([0.57, 0.075, 0.08]), np.array([0.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17])),
            _INPLACE, True)

        print(self.sim.data.ctrl)
        print(self.sim.data.qpos)

    def get_state(self):
        self.sim.get_state()
        # 如果定义了相机
        # self.sim.data.get_camera_xpos('[camera name]')

    def reset(self):
        self.sim.reset()
        self.timestep = 0

    def step(self):
        # self.testQposFromSitePose((np.array([0.605, 0.075, 0.03]), np.array([0.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17])), _INPLACE)
        x=random.uniform(0.415, 0.635)
        y=random.uniform(-0.105, 0.105)

        self.testQposFromSitePose(
            (np.array([x, y, 0.045]), np.array([0.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17])),
            _INPLACE)
        # self.testQposFromSitePose(
        #     (None, np.array([0.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17])),
        #     _INPLACE, True)
        self.sim.step()
        # self.sim.data.ctrl[0] += 0.01
        # print(self.sim.data.ctrl)
        # pdb.set_trace()
        # print(self.sim.data.qpos)
        print("sensordata", self.sim.data.sensordata)
        # self.viewer.add_overlay(const.GRID_TOPRIGHT, " ", SESSION_NAME)
        self.viewer.render()
        self.timestep += 1

    def create_viewer(self, run_speed=0.0005):
        self.viewer = MjViewer(self.sim)
        self.viewer._run_speed = run_speed
        # self.viewer._hide_overlay = HIDE_OVERLAY
        # self.viewer.vopt.frame = DISPLAY_FRAME
        # self.viewer.cam.azimuth = CAM_AZIMUTH
        # self.viewer.cam.distance = CAM_DISTANCE
        # self.viewer.cam.elevation = CAM_ELEVATION

    def testQposFromSitePose(self, target, inplace, qpos_flag=False):

        physics = mujoco.Physics.from_xml_string(FlexivPeg_XML)
        target_pos, target_quat = target
        count = 0
        physics2 = physics.copy(share_model=True)
        resetter = _ResetArm(seed=0)
        while True:
          result = ik.qpos_from_site_pose(
              physics=physics2,
              site_name=_SITE_NAME,
              target_pos=target_pos,
              target_quat=target_quat,
              joint_names=_JOINTS,
              tol=_TOL,
              max_steps=_MAX_STEPS,
              inplace=inplace,
          )

          if result.success:
            break
          elif count < _MAX_RESETS:
            resetter(physics2)
            count += 1
          else:
            raise RuntimeError(
                'Failed to find a solution within %i attempts.' % _MAX_RESETS)

        self.assertLessEqual(result.steps, _MAX_STEPS)
        self.assertLessEqual(result.err_norm, _TOL)
        # pdb.set_trace()
        physics.data.qpos[:] = result.qpos
        for i in range(len(self.sim.data.qpos)):
            if qpos_flag:
                self.sim.data.qpos[i]=physics.data.qpos[i]
            else:
                self.sim.data.ctrl[i] = physics.data.qpos[i]
        # print(physics.data.qpos)
        mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
        if target_pos is not None:
          pos = physics.named.data.site_xpos[_SITE_NAME]
          np.testing.assert_array_almost_equal(pos, target_pos)
        if target_quat is not None:
          xmat = physics.named.data.site_xmat[_SITE_NAME]
          quat = np.empty_like(target_quat)
          mjlib.mju_mat2Quat(quat, xmat)
          quat /= quat.ptp()  # Normalize xquat so that its max-min range is 1
          # np.testing.assert_array_almost_equal(quat, target_quat)

if __name__ == "__main__":
    env = my_env()

    env.get_state()

    while True:
      if env.timestep == 1000:
        env.reset()
      #   input()
      env.step()
      # print(env.sim.data.sensordata)



