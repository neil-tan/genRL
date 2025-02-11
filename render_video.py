import genesis as gs
import numpy as np
import sys

def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer = True,
        viewer_options = gs.options.ViewerOptions(
            # res           = (1280, 960),
            camera_pos    = (3.5, 0.0, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 40,
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
        gs.morphs.URDF(file='assets/urdf/cartpole.urdf'),
    )

    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (3.5, 0.0, 2.5),
        lookat = (0, 0, 0.5),
        fov    = 45,
        GUI    = False,
    )

    scene.build()
    
    # jnt_names = [j.name for j in cartpole.joints]
    jnt_names = ['slider_to_cart', 'cart_to_pole']
    dofs_idx = [cartpole.get_joint(name).dof_idx_local for name in jnt_names]
    # cartpole.set_pos(np.array([0, 0, 10]))
    cartpole.control_dofs_force(np.array([10,0]), dofs_idx)
    
    if not sys.platform == "linux":
        if sys.platform == "darwin" and scene._visualizer._viewer is not None:
            scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
        gs.tools.run_in_another_thread(fn=run_sim, args=(scene, cam))
    else:
        run_sim(scene, cam)
    scene.viewer.start()


def run_sim(scene, cam):
    cam.start_recording()
    angular_velocity_scaler = 0.5
    for i in range(300):
        scene.step()
        cam.set_pose(
            pos    = (3.0 * np.sin(i / 60 * angular_velocity_scaler), 3.0 * np.cos(i / 60 * angular_velocity_scaler), 2.5),
            lookat = (0, 0, 0.5),
        )
        cam.render()
    cam.stop_recording(save_to_filename='video.mp4', fps=60)
    scene.viewer.stop()


if __name__ == "__main__":
    main()