"""Jacobian-based damped least-squares IK solver for MuJoCo robots.

Works with any robot described by a ``RobotSpec`` from ``robot_registry``.
Uses ``mujoco.mj_jac`` to compute the geometric Jacobian of the end-effector
site, then iteratively solves for joint angles that reach a target pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import numpy.typing as npt

from .robot_registry import RobotSpec


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IKResult:
    """Result of a single IK solve."""
    success: bool
    qpos: npt.NDArray[np.float64]
    position_error: float
    orientation_error: float
    iterations: int


# ---------------------------------------------------------------------------
# IK Solver
# ---------------------------------------------------------------------------

class JacobianIKSolver:
    """Damped least-squares IK solver using MuJoCo Jacobians.

    Parameters
    ----------
    model : mujoco.MjModel
        The loaded MuJoCo model.
    data : mujoco.MjData
        The MuJoCo data instance (will be modified during solving).
    robot_spec : RobotSpec
        Specification of the robot being solved for.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_spec: RobotSpec):
        self.model = model
        self.data = data
        self.spec = robot_spec

        # Resolve joint and site IDs
        self._joint_ids: List[int] = []
        for jname in robot_spec.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found in model")
            self._joint_ids.append(jid)

        # End-effector site
        ee_site = robot_spec.end_effector_site
        if ee_site:
            self._ee_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, ee_site
            )
        else:
            self._ee_site_id = -1

        if self._ee_site_id < 0:
            # Fallback: use the end-effector body's child site if any
            ee_body = robot_spec.end_effector_body
            if ee_body:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, ee_body
                )
                if body_id >= 0:
                    # Find first site attached to this body
                    for i in range(model.nsite):
                        if model.site_bodyid[i] == body_id:
                            self._ee_site_id = i
                            break

        self._use_body_fallback = False
        self._ee_body_id = -1

        if self._ee_site_id < 0:
            # Fallback: use body position (e.g. Panda with no site)
            ee_body = robot_spec.end_effector_body
            if ee_body:
                self._ee_body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, ee_body
                )
            if self._ee_body_id >= 0:
                self._use_body_fallback = True
            else:
                raise ValueError(
                    f"Cannot find end-effector site or body for robot '{robot_spec.robot_id}'. "
                    f"EE site='{robot_spec.end_effector_site}', body='{robot_spec.end_effector_body}'"
                )

        # Pre-compute qpos and nv (velocity) indices for arm joints
        self._qpos_indices: List[int] = []
        self._nv_indices: List[int] = []
        for jid in self._joint_ids:
            self._qpos_indices.append(model.jnt_qposadr[jid])
            self._nv_indices.append(model.jnt_dofadr[jid])

        self._dof = len(self._joint_ids)

        # Temp arrays for Jacobian
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))

    def solve(
        self,
        target_pos: Sequence[float],
        target_quat: Optional[Sequence[float]] = None,
        damping: float = 0.05,
        max_iters: int = 100,
        pos_tol: float = 1e-3,
        ori_tol: float = 1e-2,
    ) -> IKResult:
        """Solve IK for a single target pose.

        Parameters
        ----------
        target_pos : array-like (3,)
            Target world position [x, y, z].
        target_quat : array-like (4,), optional
            Target orientation as MuJoCo quaternion [w, x, y, z].
            If None, only position is solved (orientation unconstrained).
        damping : float
            Damping factor for damped least-squares.
        max_iters : int
            Maximum solver iterations.
        pos_tol : float
            Position convergence tolerance (meters).
        ori_tol : float
            Orientation convergence tolerance (radians).

        Returns
        -------
        IKResult
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        use_ori = target_quat is not None
        if use_ori:
            target_quat = np.asarray(target_quat, dtype=np.float64)

        data = self.data
        model = self.model
        site_id = self._ee_site_id
        dof = self._dof

        for iteration in range(max_iters):
            # Update kinematics to get current EE position
            mujoco.mj_forward(model, data)

            # Current EE pose (site or body fallback)
            if self._use_body_fallback:
                site_xpos = data.xpos[self._ee_body_id].copy()
                site_xmat = data.xmat[self._ee_body_id].reshape(3, 3).copy()
            else:
                site_xpos = data.site_xpos[site_id].copy()
                site_xmat = data.site_xmat[site_id].reshape(3, 3).copy()

            # Position error
            pos_err = target_pos - site_xpos
            pos_err_norm = np.linalg.norm(pos_err)

            # Orientation error (if requested)
            ori_err_norm = 0.0
            if use_ori:
                # Current site quaternion
                site_quat = np.zeros(4)
                mujoco.mju_mat2Quat(site_quat, site_xmat)
                # Error quaternion: target * current^-1
                err_quat = np.zeros(4)
                mujoco.mju_negQuat(err_quat, site_quat)
                err_quat_final = np.zeros(4)
                mujoco.mju_mulQuat(err_quat_final, target_quat, err_quat)
                # Convert to rotation vector (axis-angle)
                ori_err = np.zeros(3)
                mujoco.mju_quat2Vel(ori_err, err_quat_final, 1.0)
                ori_err_norm = np.linalg.norm(ori_err)

            # Check convergence
            if pos_err_norm < pos_tol and (not use_ori or ori_err_norm < ori_tol):
                return IKResult(
                    success=True,
                    qpos=self._get_arm_qpos(),
                    position_error=pos_err_norm,
                    orientation_error=ori_err_norm,
                    iterations=iteration,
                )

            # Compute Jacobian (site or body fallback)
            if self._use_body_fallback:
                mujoco.mj_jacBody(model, data, self._jac_pos, self._jac_rot, self._ee_body_id)
            else:
                mujoco.mj_jacSite(model, data, self._jac_pos, self._jac_rot, site_id)

            # Extract arm-joint columns (using velocity-space indices)
            J_pos = self._jac_pos[:, self._nv_indices]  # (3, dof)
            J_rot = self._jac_rot[:, self._nv_indices]  # (3, dof)

            if use_ori:
                J = np.vstack([J_pos, J_rot])  # (6, dof)
                error = np.concatenate([pos_err, ori_err])  # (6,)
            else:
                J = J_pos  # (3, dof)
                error = pos_err  # (3,)

            # Damped least-squares: dq = J^T @ inv(J @ J^T + λ²I) @ error
            JJT = J @ J.T
            lam_sq = damping ** 2
            A = JJT + lam_sq * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(A, error)

            # Apply joint update with limits
            self._apply_dq(dq)

        # Did not converge
        return IKResult(
            success=False,
            qpos=self._get_arm_qpos(),
            position_error=float(pos_err_norm),
            orientation_error=float(ori_err_norm) if use_ori else 0.0,
            iterations=max_iters,
        )

    def solve_trajectory(
        self,
        waypoints: Sequence[Sequence[float]],
        target_quat: Optional[Sequence[float]] = None,
        damping: float = 0.05,
        max_iters: int = 100,
        pos_tol: float = 1e-3,
    ) -> List[IKResult]:
        """Solve IK for a sequence of waypoints, seeding each solve with
        the previous solution for smoothness."""
        results: List[IKResult] = []
        for wp in waypoints:
            result = self.solve(
                target_pos=wp,
                target_quat=target_quat,
                damping=damping,
                max_iters=max_iters,
                pos_tol=pos_tol,
            )
            results.append(result)
            if not result.success:
                # Continue with best-effort qpos
                pass
        return results

    def get_ee_pos(self) -> npt.NDArray[np.float64]:
        """Get current end-effector position after mj_forward."""
        mujoco.mj_forward(self.model, self.data)
        if self._use_body_fallback:
            return self.data.xpos[self._ee_body_id].copy()
        return self.data.site_xpos[self._ee_site_id].copy()

    def get_ee_quat(self) -> npt.NDArray[np.float64]:
        """Get current end-effector orientation as quaternion [w,x,y,z]."""
        mujoco.mj_forward(self.model, self.data)
        if self._use_body_fallback:
            xmat = self.data.xmat[self._ee_body_id].reshape(3, 3)
        else:
            xmat = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat)
        return quat

    def set_arm_qpos(self, qpos: Sequence[float]) -> None:
        """Set arm joint positions in data.qpos."""
        for i, qi in enumerate(qpos):
            self.data.qpos[self._qpos_indices[i]] = qi

    def _get_arm_qpos(self) -> npt.NDArray[np.float64]:
        """Read current arm joint positions from data.qpos."""
        return np.array(
            [self.data.qpos[idx] for idx in self._qpos_indices],
            dtype=np.float64,
        )

    def _apply_dq(self, dq: npt.NDArray[np.float64]) -> None:
        """Apply joint delta with joint-limit clamping."""
        model = self.model
        data = self.data
        for i, (jid, qidx) in enumerate(zip(self._joint_ids, self._qpos_indices)):
            new_q = data.qpos[qidx] + dq[i]
            # Clamp to joint range
            jrange = model.jnt_range[jid]
            if jrange[0] < jrange[1]:  # has limits
                new_q = np.clip(new_q, jrange[0], jrange[1])
            data.qpos[qidx] = new_q
