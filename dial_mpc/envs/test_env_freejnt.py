from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
import jax.numpy as jp
import pickle
from functools import partial

from brax import math
import brax.base as base
from brax.base import Motion, Transform
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path
import dial_mpc.envs as dial_envs
import dial_mpc.utils.io_dict_to_hdf5 as ioh5
from dial_mpc.utils import quaternions
from dial_mpc.utils.preprocess import ReferenceClip

@dataclass
class FlyConfig(BaseEnvConfig):
    task_name: str = "fly"
    randomize_tasks: bool = False  # Whether to randomize the task.
    # P gain, or a list of P gains for each joint.
    kp: float = 30.0
    # D gain, or a list of D gains for each joint.
    kd: float = 1.0
    debug: bool = True
    # dt of the environment step, not the underlying simulator step.
    dt: float = 0.002
    # timestep of the underlying simulator step. user is responsible for making sure it matches their model.
    timestep: float = 0.002
    backend: str = "mjx"  # backend of the environment.
    # control method for the joints, either "torque" or "position"
    leg_control: str = "torque"
    clip_length: int = 601
    obs_noise: float = 0.05
    physics_steps_per_control_step: int = 10
    action_scale: float = 1.0  # scale of the action space.
    free_jnt: bool = True  # whether the first joint is free.
    mocap_hz: int = 200  # frequency of the mocap data.
    too_far_dist: float = 1000.0
    bad_pose_dist: float = 20 # 60.0
    bad_quat_dist: float = 1000.0 #1.25
    ctrl_cost_weight: float = 0.01
    pos_reward_weight: float = 0.0
    quat_reward_weight: float = 0.0
    joint_reward_weight: float = 1.0
    angvel_reward_weight: float = 1.0
    bodypos_reward_weight: float = 0.0
    endeff_reward_weight: float = 0.0
    healthy_reward: float = 0.25
    healthy_z_range: tuple = (-0.05, 0.1)
    terminate_when_unhealthy: bool = True
    ref_len: int = 5
    reset_noise_scale=1e-3
    solver: str ="cg"
    iterations: int = 6
    ls_iterations: int = 6
    free_jnt: bool = False
    inference_mode: bool = False
    end_eff_names: tuple = ('claw_T1_left', 'claw_T1_right', 'claw_T2_left', 'claw_T2_right', 'claw_T3_left', 'claw_T3_right')
    appendage_names: tuple = ('tarsus_T1_left', 'tarsus_T1_right', 'tarsus_T2_left', 'tarsus_T2_right', 'tarsus_T3_left', 'tarsus_T3_right')
    body_names: tuple = (
    'thorax',
    'head',
    'rostrum',
    'haustellum',
    'labrum_left',
    'labrum_right',
    'antenna_left',
    'antenna_right',
    'wing_left',
    'wing_right',
    'abdomen',
    'abdomen_2',
    'abdomen_3',
    'abdomen_4',
    'abdomen_5',
    'abdomen_6',
    'abdomen_7',
    'haltere_left',
    'haltere_right',
    'coxa_T1_left',
    'femur_T1_left',
    'tibia_T1_left',
    'tarsus_T1_left',
    'tarsus2_T1_left',
    'tarsus3_T1_left',
    'tarsus4_T1_left',
    'claw_T1_left',
    'coxa_T1_right',
    'femur_T1_right',
    'tibia_T1_right',
    'tarsus_T1_right',
    'tarsus2_T1_right',
    'tarsus3_T1_right',
    'tarsus4_T1_right',
    'claw_T1_right',
    'coxa_T2_left',
    'femur_T2_left',
    'tibia_T2_left',
    'tarsus_T2_left',
    'tarsus2_T2_left',
    'tarsus3_T2_left',
    'tarsus4_T2_left',
    'claw_T2_left',
    'coxa_T2_right',
    'femur_T2_right',
    'tibia_T2_right',
    'tarsus_T2_right',
    'tarsus2_T2_right',
    'tarsus3_T2_right',
    'tarsus4_T2_right',
    'claw_T2_right',
    'coxa_T3_left',
    'femur_T3_left',
    'tibia_T3_left',
    'tarsus_T3_left',
    'tarsus2_T3_left',
    'tarsus3_T3_left',
    'tarsus4_T3_left',
    'claw_T3_left',
    'coxa_T3_right',
    'femur_T3_right',
    'tibia_T3_right',
    'tarsus_T3_right',
    'tarsus2_T3_right',
    'tarsus3_T3_right',
    'tarsus4_T3_right',
    'claw_T3_right',)
    
    site_names: tuple = (
        'tracking[coxa_T1_left]',
        'tracking[femur_T1_left]',
        'tracking[tibia_T1_left]',
        'tracking[tarsus_T1_left]',
        'tracking[claw_T1_left]',
        'tracking[coxa_T1_right]',
        'tracking[femur_T1_right]',
        'tracking[tibia_T1_right]',
        'tracking[tarsus_T1_right]',
        'tracking[claw_T1_right]',
        'tracking[coxa_T2_left]',
        'tracking[femur_T2_left]',
        'tracking[tibia_T2_left]',
        'tracking[tarsus_T2_left]',
        'tracking[claw_T2_left]',
        'tracking[coxa_T2_right]',
        'tracking[femur_T2_right]',
        'tracking[tibia_T2_right]',
        'tracking[tarsus_T2_right]',
        'tracking[claw_T2_right]',
        'tracking[coxa_T3_left]',
        'tracking[femur_T3_left]',
        'tracking[tibia_T3_left]',
        'tracking[tarsus_T3_left]',
        'tracking[claw_T3_left]',
        'tracking[coxa_T3_right]',
        'tracking[femur_T3_right]',
        'tracking[tibia_T3_right]',
        'tracking[tarsus_T3_right]',
        'tracking[claw_T3_right]',)
    
    joint_names: tuple = (
    'coxa_flexion_T1_left',
    'coxa_twist_T1_left',
    'femur_T1_left',
    'femur_twist_T1_left',
    'tibia_T1_left',
    'tarsus_T1_left',
    'coxa_flexion_T1_right',
    'oxa_twist_T1_right',
    'emur_T1_right',
    'emur_twist_T1_right',
    'tibia_T1_right',
    'tarsus_T1_right',
    'coxa_flexion_T2_left',
    'coxa_twist_T2_left',
    'femur_T2_left',
    'femur_twist_T2_left',
    'tibia_T2_left',
    'tarsus_T2_left',
    'coxa_flexion_T2_right',
    'oxa_twist_T2_right',
    'emur_T2_right',
    'emur_twist_T2_right',
    'tibia_T2_right',
    'tarsus_T2_righ',
    'coxa_flexion_T3_left',
    'coxa_twist_T3_left',
    'femur_T3_left',
    'femur_twist_T3_left',
    'tibia_T3_left',
    'tarsus_T3_left',
    'coxa_flexion_T3_right',
    'oxa_twist_T3_right',
    'emur_T3_right',
    'emur_twist_T3_right',
    'tibia_T3_right',
    'tarsus_T3_rig',)
    center_of_mass: str = 'thorax'
    


class Flybody(BaseEnv):
    def __init__(self, config: FlyConfig):
        super().__init__(config)
        
        
        mj_model = self.sys.mj_model
        # dpath = '/mmfs1/home/eabe/Research/MyRepos/dial-mpc/flybody/0.h5'
        dpath = '/home/eabe/Research/MyRepos/dial-mpc/flybody/0.h5'
        ref_clip = ioh5.load(dpath)
        clip = ReferenceClip()
        for key, val in ref_clip.items():
            ref_clip[key] = jp.array(val)
        clip = clip.replace(
            position=ref_clip['position'],
            quaternion=ref_clip['quaternion'],
            joints=ref_clip['joints'],
            body_positions=ref_clip['body_positions'],
            velocity=ref_clip['velocity'],
            joints_velocity=ref_clip['joints_velocity'],
            angular_velocity=ref_clip['angular_velocity'],
            body_quaternions=ref_clip['body_quaternions'],
        )

        self._thorax_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), config.center_of_mass
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in config.joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in config.body_names
            ]
        )
        
        self._site_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("site"), site)
                for site in config.site_names
            ]
        )
        # using this for appendage for now bc im to lazy to rename
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in config.end_eff_names
            ]
        )
        self._sim_timestep = config.timestep
        self._free_jnt = config.free_jnt
        self._inference_mode = config.inference_mode
        self._mocap_hz = config.mocap_hz
        self._bad_pose_dist = config.bad_pose_dist
        self._too_far_dist = config.too_far_dist
        self._bad_quat_dist = config.bad_quat_dist
        self._ref_traj = clip
        self._ref_len = config.ref_len
        self._clip_len = config.clip_length
        self._pos_reward_weight = config.pos_reward_weight
        self._quat_reward_weight = config.quat_reward_weight
        self._joint_reward_weight = config.joint_reward_weight
        self._angvel_reward_weight = config.angvel_reward_weight
        self._bodypos_reward_weight = config.bodypos_reward_weight
        self._endeff_reward_weight = config.endeff_reward_weight
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._healthy_reward = config.healthy_reward
        self._healthy_z_range = config.healthy_z_range
        self._reset_noise_scale = config.reset_noise_scale
        self._terminate_when_unhealthy = config.terminate_when_unhealthy

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

        start_frame = jax.random.randint(rng, (), 0, 44)

        info = {
            "start_frame": start_frame,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        new_qpos = jp.array(self.sys.qpos0)

        # Add quat
        # new_qpos = qpos_with_pos.at[3:7].set(self._track_quat[start_frame])

        # Add noise
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, start_frame)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
        }
        return State(data, obs, reward, done, metrics, info)
    def make_system(self, config: FlyConfig) -> System:
        model_path = ("/home/eabe/Research/MyRepos/dial-mpc/dial_mpc/models/fruitfly/fruitfly_force_fast.xml")
        # model_path = ("/mmfs1/home/eabe/Research/MyRepos/dial-mpc/dial_mpc/models/fruitfly/fruitfly_force_fast.xml")
        # sys = mjcf.load(model_path)
        # sys = sys.tree_replace({"opt.timestep": config.timestep})
        spec = mujoco.MjSpec()
        spec.from_file(model_path)
        thorax = spec.find_body("thorax")
        first_joint = thorax.first_joint()
        if (config.free_jnt == False) & (first_joint.name == "free"):
            first_joint.delete()
        mj_model = spec.compile()

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }['cg']
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.timestep = config.timestep
        mj_model.opt.jacobian = 0

        sys = mjcf.load_model(mj_model)

        return sys
    
    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()

        # Logic for getting current frame aligned with simulation time
        cur_frame = (
            info["start_frame"] + jp.floor(data.time * self._mocap_hz).astype(jp.int32)
        ) % self._clip_len

        if self._ref_traj.position is not None:
            track_pos = self._ref_traj.position
            pos_distance = data.qpos[:3] - track_pos[cur_frame]
            pos_reward = self._pos_reward_weight * jp.exp(
                -400 * jp.sum(pos_distance**2)
            )
            track_quat = self._ref_traj.quaternion
            quat_distance = jp.sum(
                self._bounded_quat_dist(data.qpos[3:7], track_quat[cur_frame]) ** 2
            )
            quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)
        else:
            pos_distance = 0.0
            quat_distance = 0.0
            pos_reward = 0.0
            quat_reward = 0.0

        track_joints = self._ref_traj.joints
        joint_distance = jp.sum((data.qpos[self._joint_idxs] - track_joints[cur_frame])** 2) 
        # joint_reward = self._joint_reward_weight * jp.exp(-0.1 * joint_distance)
        joint_reward = self._joint_reward_weight* jp.exp(-0.5/.8**2  * joint_distance)
        info["joint_distance"] = joint_distance

        track_angvel = self._ref_traj.angular_velocity
        angvel_distance = jp.sum((data.qvel[3:6] - track_angvel[cur_frame])** 2)
        # angvel_reward = self._angvel_reward_weight * jp.exp(-0.01 * angvel_distance)
        # angvel_reward = self._angvel_reward_weight * jp.exp(-0.5/53.7801**2 * angvel_distance)
        angvel_reward = self._angvel_reward_weight* jp.exp(-20 * angvel_distance)
        info["angvel_distance"]
        
        track_bodypos = self._ref_traj.body_positions
        bodypos_distance = jp.sum((data.xpos[self._body_idxs] - track_bodypos[cur_frame][self._body_idxs]).flatten()** 2)
        bodypos_reward = self._bodypos_reward_weight* jp.exp(-50 * bodypos_distance)
        # bodypos_reward = self._bodypos_reward_weight * jp.exp(-0.1* bodypos_distance)
        info["bodypos_distance"] = bodypos_distance
        
        endeff_distance = jp.sum((data.xpos[self._endeff_idxs] - track_bodypos[cur_frame][self._endeff_idxs]).flatten()** 2)
        endeff_reward = self._endeff_reward_weight* jp.exp(-700 * endeff_distance)
        # endeff_reward = self._endeff_reward_weight * jp.exp(-0.5 * endeff_distance)
        info["endeff_distance"] = endeff_distance

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, cur_frame)
        rewards_temp = self.get_reward_factors(data)
        pos_reward = rewards_temp[0]
        # joint_reward = rewards_temp[1]
        quat_reward = rewards_temp[2]
        # rewards = {
        #     'pos_reward': rewards_temp[0],
        #     'joint_reward': rewards_temp[1],
        #     'quat_reward': rewards_temp[2],
        # }
        # reward = sum(rewards.values())

        reward = (
            pos_reward
            + joint_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            + healthy_reward
            - ctrl_cost
        )
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        done = jp.max(jp.array([done, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=1 - is_healthy,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_obs(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        # Get the relevant slice of the ref_traj
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    cur_frame + 1,
                    self._ref_len,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, self._ref_traj)

        # track_pos_local = jax.vmap(
        #     lambda a, b: math.rotate(a, b), in_axes=(0, None)
        # )(
        #     ref_traj.position - data.qpos[:3],
        #     data.qpos[3:7],
        # ).flatten()

        # quat_dist = jax.vmap(
        #     lambda a, b: math.relative_quat(a, b), in_axes=(None, 0)
        # )(
        #     data.qpos[3:7],
        #     ref_traj.quaternion,
        # ).flatten()

        joint_dist = (ref_traj.joints - data.qpos[self._joint_idxs]).flatten()

        # TODO test if this works
        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj.body_positions[:,self._body_idxs] - data.xpos[self._body_idxs]),
            data.qpos[3:7],
        ).flatten()

        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                # data.cinert[1:].ravel(),
                # data.cvel[1:].ravel(),
                # data.qfrc_actuator,
                # track_pos_local,
                # quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes a quaternion distance limiting the difference to a max of pi/2.

        This function supports an arbitrary number of batch dimensions, B.

        Args:
          source: a quaternion, shape (B, 4).
          target: another quaternion, shape (B, 4).

        Returns:
          Quaternion distance, shape (B, 1).
        """
        source /= jp.linalg.norm(source, axis=-1, keepdims=True)
        target /= jp.linalg.norm(target, axis=-1, keepdims=True)
        # "Distance" in interval [-1, 1].
        dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
        # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
        dist = jp.minimum(1.0, dist)
        # Divide by 2 and add an axis to ensure consistency with expected return
        # shape and magnitude.
        return 0.5 * jp.arccos(dist)[..., np.newaxis]
    # ------------ reward functions----------------
    def get_reward_factors(self, data):
        """Returns factorized reward terms."""
        if self._inference_mode:
            return (1,)
        step = round(data.time / self.dt)
        walker_ft = self._get_walker_features(data, self._joint_idxs,
                                        self._site_idxs)
        reference_ft = self._get_reference_features(self._ref_traj, step)
        reward_factors = self._reward_factors_deep_mimic(
            walker_features=walker_ft,
            reference_features=reference_ft,
            weights=(0, 1, 1))
        return reward_factors
    
    def _compute_diffs(self, walker_features: Dict[str, jp.ndarray],
                    reference_features: Dict[str, jp.ndarray],
                    n: int = 2) -> Dict[str, float]:
        """Computes sums of absolute values of differences between components of
        model and reference features.

        Args:
            model_features, reference_features: Dictionaries of features to compute
                differences of.
            n: Exponent for differences. E.g., for squared differences use n = 2.

        Returns:
            Dictionary of differences, one value for each entry of input dictionary.
        """
        diffs = {}
        for k in walker_features:
            if 'quat' not in k:
                # Regular vector differences.
                diffs[k] = jp.sum(
                    jp.abs(walker_features[k] - reference_features[k])**n)
            else:
                # Quaternion differences (always positive, no need to use jp.abs).
                diffs[k] = jp.sum(
                    quaternions.quat_dist_short_arc(walker_features[k],
                                                    reference_features[k])**n)
        return diffs


    def _get_walker_features(self, data, mocap_joints, mocap_sites):
        """Returns model pose features."""

        qpos = data.qpos[mocap_joints]
        qvel = data.qvel[mocap_joints]
        sites =data.xpos[mocap_sites]
        # root2site = quaternions.get_egocentric_vec(qpos[:3], sites, qpos[3:7])

        # Joint quaternions in local egocentric reference frame,
        # (except root quaternion, which is in world reference frame).
        root_quat = data.qpos[3:7]
        
        xaxis1 = data.xaxis[mocap_joints]
        xaxis1 = quaternions.rotate_vec_with_quat(
            xaxis1, quaternions.reciprocal_quat(root_quat))
        
        joint_quat = quaternions.joint_orientation_quat(xaxis1, qpos)
        joint_quat = jp.vstack((root_quat, joint_quat))

        model_features = {
            'com': qpos[:3],
            'qvel': qvel,
            # 'root2site': root2site,
            'joint_quat': joint_quat,
        }

        return model_features


    def _get_reference_features(self,reference_clip, step):
        """Returns reference pose features."""
        qpos_ref = reference_clip.joints[step, :]
        qvel_ref = reference_clip.joints_velocity[step, :]
        # root2site_ref = reference_clip['root2site'][step, :, :]
        joint_quat_ref = reference_clip.body_quaternions[step, self._joint_idxs, :]
        joint_quat_ref = jp.vstack((reference_clip.quaternion[step], joint_quat_ref))

        reference_features = {
            'com': reference_clip.position[step, :],
            'qvel': qvel_ref,
            # 'root2site': root2site_ref,
            'joint_quat': joint_quat_ref,
        }
        return reference_features


    def _reward_factors_deep_mimic(self, walker_features,
                                reference_features,
                                std=None,
                                weights=(1, 1, 1)):
        """Returns four reward factors, each of which is a product of individual
        (unnormalized) Gaussian distributions evaluated for the four model
        and reference data features:
            1. Cartesian center-of-mass position, qpos[:3].
            2. qvel for all joints, including the root joint.
            3. Egocentric end-effector vectors. Deleted for now.
            4. All joint orientation quaternions (in egocentric local reference
            frame), and the root quaternion.

        The reward factors are equivalent to the ones in the DeepMimic:
        https://arxiv.org/abs/1804.02717
        """
        if std is None:
            # Default values for fruitfly walking imitation task.
            std = {
                'com': 0.078487,
                'qvel': 53.7801,
                # 'root2site': 0.0735,
                'joint_quat': 1.2247
            }

        diffs = self._compute_diffs(walker_features, reference_features, n=2)
        reward_factors = []
        for k in walker_features.keys():
            reward_factors.append(jp.exp(-0.5 / std[k]**2 * diffs[k]))
        reward_factors = jp.array(reward_factors)
        reward_factors *= jp.asarray(weights)

        return reward_factors

brax_envs.register_environment("Flybody", Flybody)
# dial_envs.register_config("FlyConfig", FlyConfig)