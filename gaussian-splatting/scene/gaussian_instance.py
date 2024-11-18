import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_raw_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    random_quaternions,
    random_rotation,
    random_rotations,
    rotation_6d_to_matrix,
    standardize_quaternion,
)
from scene.gaussian_model import GaussianModel
def quat_mult(q1, q2):
    # NOTE:
    # Q1 is the quaternion that rotates the vector from the original position to the final position
    # Q2 is the quaternion that been rotated
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T
class GaussianInstance:
    def __init__(self, model):
        self._model = model
        self._location = torch.zeros(3, device="cuda")
        self._rotation = torch.zeros(3, device="cuda")
   
    def add_to_world(self, gs):
        if self._model:
        #   quat = axis_angle_to_quaternion(self._rotation)
            rot_mat = euler_angles_to_matrix(self._rotation, "XYZ")
            quat = matrix_to_quaternion(rot_mat)
            quat = quat / quat.norm(dim=-1, keepdim=True)
            quats = quat.expand(self._model._rotation.shape[0],4)
            means = torch.matmul(self._model.get_xyz, rot_mat) + self._location
            rotations = quat_mult(self._model._rotation, quats)
            rotations2 = quaternion_multiply(self._model._rotation, quats)
        #   points_hom = torch.cat([points, ones], dim=1)

            gs._xyz = torch.cat([gs._xyz, means], 0)
            gs._rotation = torch.cat([gs._rotation, rotations], 0)  
            gs._features_dc = torch.cat([gs._features_dc, self._model._features_dc], 0)
            gs._features_rest = torch.cat([gs._features_rest, self._model._features_rest], 0)
            gs._opacity = torch.cat([gs._opacity, self._model._opacity], 0)
            gs._scaling = torch.cat([gs._scaling, self._model._scaling], 0)
     