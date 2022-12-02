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
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        use_object_obs=False,
        camera_names='birdview',
        camera_heights=512,
        camera_widths=512,
    )

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="birdview", help="Name of camera to render")
    parser.add_argument("--video_path", type=str, default="./video_data/")
    parser.add_argument("--data_path", type=str, default="./dynamics_data/")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=5)
    args = parser.parse_args()

    env = make_robosuite_env()

    #TODO: Randomize robot start position

    for rollout in range(args.num_rollouts):
        obs = env.reset()
        ndim = env.action_dim

        # create a video writer with imageio
        writer = imageio.get_writer(args.video_path + f'video_{rollout}.mp4', fps=20)
        frames = []
        dynamics_data = []

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
            eef_vel = env.sim.data.get_body_xpos("robot0_right_hand")
            state[robot_eef_name]['x'] = eef_pos[0]
            state[robot_eef_name]['y'] = eef_pos[1]
            state[robot_eef_name]['dx'] = eef_vel[0]
            state[robot_eef_name]['dy'] = eef_vel[1]
            print(env.sim.data.get_body_xpos("robot0_right_hand"))

            frames.append(state)

            # dump a frame from every K frames
            if i % args.skip_frame == 0:
                frame = obs[args.camera + "_image"]
                writer.append_data(frame)
                print("Saving frame #{}".format(i))

            if done:
                break

        pickle.dump(frames, open(args.data_path + f'rollout_{rollout}.pkl', 'wb'))
        writer.close()

