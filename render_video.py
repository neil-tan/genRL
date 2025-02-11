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
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        # gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        gs.morphs.URDF(file='assets/urdf/cartpole.urdf'),
    )

    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (3.5, 0.0, 2.5),
        lookat = (0, 0, 0.5),
        fov    = 30,
        GUI    = False,
    )

    scene.build()
    if not sys.platform == "linux":
        if sys.platform == "darwin" and scene._visualizer._viewer is not None:
            scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
        gs.tools.run_in_another_thread(fn=run_sim, args=(scene, cam))
    else:
        run_sim(scene, cam)
    scene.viewer.start()


def run_sim(scene, cam):
    cam.start_recording()
    for i in range(300):
        scene.step()
        cam.set_pose(
            pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
            lookat = (0, 0, 0.5),
        )
        cam.render()
    cam.stop_recording(save_to_filename='video.mp4', fps=60)
    scene.viewer.stop()


if __name__ == "__main__":
    main()