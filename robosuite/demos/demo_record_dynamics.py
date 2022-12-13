"""
Record video and dynamics of agent episodes.
This script uses offscreen rendering.

"""

from squaternion import Quaternion

import argparse
import pdb
import imageio
import numpy as np
import pickle
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.camera_utils import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


import robosuite.utils.macros as macros
from robosuite import make

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"


def make_robosuite_env():
    environment = "Push"
    robot = "Floating"

    # initialize an environment with offscreen renderer
    env = make(
        environment,
        robot,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=['tableview'],
        camera_heights=64,
        camera_widths=64,
        camera_segmentations=["instance"]
    )

    return env


def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays.
    i, j = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = np.stack([transformed_i, -transformed_j, -np.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = np.sum(camera_dirs, axis=-1)
    ray_origins = np.broadcast_to(height_width_focal, np.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)


if __name__ == "__main__":
    # pxls  = pickle.load(open('pixel_test.pkl', 'rb'))
    # plt.imshow(pxls)
    # plt.show()

    # pxls  = pickle.load(open('gt.pkl', 'rb'))
    # imageio.imsave('./test_image_1.jpg', pxls)
    # # plt.imshow(pxls)
    # # plt.show()
    # exit(0)


    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="birdview", help="Name of camera to render")
    parser.add_argument("--video_path", type=str, default="./video_data/")
    parser.add_argument("--data_path", type=str, default="./gt/")
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=10)
    args = parser.parse_args()

    env = make_robosuite_env()

    #TODO: Randomize robot start position

    for rollout in range(args.num_rollouts):
        obs = env.reset()

        #load an extra object and try and add it to the environment
        #mujoco_object = MujocoXMLObject("./robosuite-2951/robosuite/demos/shapes/shape0.xml", name="shape0", joints=[dict(type="free", damping="0.0005")],obj_type="all",duplicate_collision_geoms=True) 
        #mujoco_object.set("pos", "0 0 1.52")


        #env.model.mujoco_objects.append(mujoco_object)

        ndim = env.action_dim

        # create a video writer with imageio
        #writer = imageio.get_writer(args.video_path + f'video_{rollout}.mp4', fps=20)
        frames = []
        dynamics_data = []

        object_index_map = {1:'cubeA', 2:'cubeB', 3:'cubeC', 4:'cubeD', 7:'roboFinger'}
        for i in range(args.timesteps):

            # run a uniformly random agent
            action = 0.5 * np.random.randn(2)
            #action[2] = 0.0 #keeps the z-joint from moving
            obs, reward, done, info = env.step([0, action[0], action[1], 0])

            #env.render()
            state = {}

            #Save ground truth dynamics info
            for obj in env.model.mujoco_objects: 

                obj_pos = env.sim.data.get_body_xpos(obj._name + "_main")
                obj_vel = env.sim.data.get_body_xvelp(obj._name + "_main")
                obj_quat = env.sim.data.get_body_xquat(obj._name + "_main")
                obj_qvel = env.sim.data.get_body_xvelr(obj._name + "_main")
                state[obj._name] = {}
                state[obj._name]['x'] = obj_pos[0]
                state[obj._name]['y'] = obj_pos[1]
                state[obj._name]['th'] = Quaternion(*obj_quat).to_euler()[2]
                state[obj._name]['dx'] = obj_vel[0]
                state[obj._name]['dy'] = obj_vel[1]
                state[obj._name]['dth'] = obj_qvel[2]

            robot_eef_name = 'eef'
            state[robot_eef_name] = {}
            eef_pos = env.sim.data.get_body_xpos("robot0_right_hand")
            eef_vel = env.sim.data.get_body_xvelr("robot0_right_hand")
            state[robot_eef_name]['x'] = eef_pos[0]
            state[robot_eef_name]['y'] = eef_pos[1]
            state[robot_eef_name]['dx'] = eef_vel[0]
            state[robot_eef_name]['dy'] = eef_vel[1]
            #print(env.sim.data.get_body_xpos("robot0_right_hand"))

            frames.append(state)

            # dump a frame from every K frames
            if i % args.skip_frame == 0: 
                # fovy = env.sim.model.cam_fovy[4]

                # f = 0.25 * args.height / np.tan(fovy * np.pi / 360)

                # #K = np.array(((-f, 0, args.width / 2), (0, f, args.height / 2), (0, 0, 1)))
                # # print(K.shape)
                # # print(env.sim.model.cam_pos)
                # pos = env.sim.model.cam_pos[4]
                # quat = env.sim.model.cam_quat[4]
                # r = R.from_quat([quat[3], quat[0], quat[1], quat[2]])
                # rot = r.as_matrix()
                # transform = np.zeros((4, 4))
                # transform[:3, :3] = env.sim.model.cam_mat0[4].reshape(3,3)
                # transform[0, 3]  = pos[0]
                # transform[1, 3]  = pos[1]

                # transform[2, 3]  = pos[2]


                # transform[3, 3]  = 1
                # #transform = get_camera_extrinsic_matrix(env.sim, 'tableview')#, 64, 64)

                # print(transform)
                # print(np.linalg.inv(transform))
                # #extrinsic = np.matmul(env.sim.model.cam_quat, env.sim.model.cam_pos)
                # #print(env.sim.model.cam_mat0[4].reshape(3,3))
                # # print(env.sim.model.cam_mat0[1].reshape(3,3))
                # # mat = env.sim.model.cam_mat0[1].reshape(3,3)
                # # mat = mat.T
                # rays = get_rays(64, 64, f, transform)
                # # print(rays[0][0])
                # # print(rays[1][0])
                
                # pixels = np.zeros((64, 64, 1))
                # for i in range(64):
                #     for j in range(64):
                #         pixels[i,j] = env.sim.ray(np.ascontiguousarray(rays[0][i,j].reshape(3,)), np.ascontiguousarray(rays[1][i,j].reshape(3,)))[1]
                # pickle.dump(pixels, open('pixel_test.pkl', 'wb'))
                frame = obs["tableview_image"]
                mask = obs["tableview_segmentation_instance"]

                # imageio.imsave('./test_image_1.jpg', frame)
                # np.save('./test_mask_1', mask)
                # #imageio.imsave('./test_mask_1.jpg', frame)
                # exit(0)

                # objs = np.unique(frame)
                # for obj in objs: 
                #     if obj not in object_index_map.keys():
                #         continue
                #     name = object_index_map[obj]
                #     folder = f'./{name}/'
                #     masked_frame = np.where(frame == obj, 1, 0)
                imageio.imsave('./images/'+ f'img_{rollout}_{i}.jpg', frame)
                np.save('./masks/' + f'img_{rollout}_{i}', mask)
                #writer.append_data(frame)
                print("Saving frame #{}".format(i))

            if done:
                break

        pickle.dump(frames, open(args.data_path + f'rollout_{rollout}.pkl', 'wb'))
        #writer.close()

