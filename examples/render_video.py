import genesis as gs
import numpy as np
import sys
# from transforms3d import euler

def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer = True,
        viewer_options = gs.options.ViewerOptions(
            res           = (2048, 960),
            camera_pos    = (0.0, 8, 0.5),
            camera_lookat = (0.0, 0.0, 3),
            camera_fov    = 60,
            max_FPS       = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True,
            world_frame_size = 1.0,
            show_link_frame  = False,
            show_cameras     = False,
            plane_reflection = True,
            ambient_light    = (0.1, 0.1, 0.1),
        ),
        renderer=gs.renderers.Rasterizer(),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -10.0),
        ),
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cartpole = scene.add_entity(
        # gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        gs.morphs.URDF(file='assets/urdf/cartpole.urdf', fixed=True),
    )

    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (0.0, 3.5, 2.5),
        lookat = (0, 0, 0.5),
        fov    = 45,
        GUI    = False,
    )

    scene.build()
    
    if not sys.platform == "linux":
        if sys.platform == "darwin" and scene._visualizer._viewer is not None:
            scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
        gs.tools.run_in_another_thread(fn=run_sim, args=(scene, cartpole, cam))
    else:
        run_sim(scene, cartpole, cam)
    scene.viewer.start()

def rotate_cam_pose(i, distance=3, angular_velocity_scaler=1, fps=60, flip=False):
    angle = i / fps * angular_velocity_scaler
    angle = -angle if flip else angle
    x = distance * np.sin(angle)
    y = distance * np.cos(angle)
    return (x, y, 2)


def run_sim(scene, cartpole, cam):
    cam.start_recording()
    
    # jnt_names = [j.name for j in cartpole.joints]
    # jnt_names = ['slider_to_cart', 'cart_to_pole']
    jnt_names = ['slider_to_cart', 'cart_to_pole']
    dofs_idx = [cartpole.get_joint(name).dof_idx_local for name in jnt_names]
    # cartpole.set_pos(np.array([0, 0, 10]))
    cartpole.control_dofs_force(np.array([-25, 0]), dofs_idx)
    
    for i in range(300):
        scene.step()
        if i > 75:
            cartpole.control_dofs_force(np.array([0, 0]), dofs_idx)
        cam.set_pose(
            pos    = rotate_cam_pose(i, distance=10, angular_velocity_scaler=0.5),
            lookat = (0, 0, 0.5),
        )
        cam.render()
        if i % 50 == 0:
            slider_to_cart_position = cartpole.get_dofs_position(dofs_idx)[0]
            cart_position_l = cartpole.get_link("cart").get_pos()[0] # x ~ xyz
            
            cart_velocity = cartpole.get_dofs_velocity(dofs_idx)[0]
            cart_velocity_l = cartpole.get_link("cart").get_vel() # xyz
            
            cart_angle_velocity = cartpole.get_link("cart").get_ang() # always 0
            cart_to_pole_position = cartpole.get_dofs_position(dofs_idx)[1]
            # pole_angle = cartpole.joints[dofs_idx[1]].dofs_motion_ang
            # pole_angle_velocity = cartpole.joints[dofs_idx[1]].dofs_motion_vel
            pole_angle_velocity = cartpole.get_link("pole").get_ang()
            # pole_angle = euler.quat2euler(cartpole.get_link("pole").get_quat(), axes='sxyz')
            pole_position_l = cartpole.get_link("pole").get_pos()
            # pole_angle_j = cartpole.get_joint('cart_to_pole').get_quat() # always [1, 0, 0, 0]
            # pole_angle_j = cartpole.get_joint('cart_to_pole').get_pos()
            
            pole_height = cartpole.links[dofs_idx[1]].get_AABB()
            pole_height_l = cartpole.get_link("pole").get_AABB()[1,2] - cartpole.get_joint('cart_to_pole').get_pos()[2]
            
    cam.stop_recording(save_to_filename='video.mp4', fps=60)
    scene.viewer.stop()


if __name__ == "__main__":
    main()