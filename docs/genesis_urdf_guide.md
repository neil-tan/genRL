# Implementing Genesis Environments with Complex URDFs (e.g., Go2)

This guide summarizes key steps and potential pitfalls when implementing Gymnasium environments using Genesis for robots defined by complex URDFs, particularly those with FREE joints (like floating bases).

## Key Steps

1.  **URDF Loading:**
    *   Use `gs.morphs.URDF(file=path_to_urdf)` to load the robot model.
    *   Ensure the URDF path is correct and accessible.
    *   **Crucially:** Make sure all mesh files (`.dae`, `.obj`, `.stl`, etc.) referenced within the URDF are present at the correct relative paths to the URDF file. Missing assets will cause loading errors.

2.  **Joint and DOF Identification:**
    *   After adding the URDF entity to the scene (`scene.add_entity(...)`), access joint information via `entity.joints`.
    *   Identify joint names (`j.name`), types (`j.type`), and DOF indices (`j.dof_idx_local`).
    *   **FREE Joints:** A `gs.JOINT_TYPE.FREE` joint represents a floating base and typically has 6 DOFs (3 translation, 3 rotation). Its `dof_idx_local` will be a *list* of 6 indices (e.g., `[0, 1, 2, 3, 4, 5]`).
    *   **Revolute/Prismatic Joints:** These usually have 1 DOF, and `dof_idx_local` will be a single integer index.

3.  **Handling DOF Indices for Genesis API:**
    *   **Internal State:** Genesis manages a flat, contiguous list of all DOFs for the entity (e.g., 6 for FREE joint + 1 for each subsequent revolute joint). The indices usually start from 0.
    *   **API Calls:** Most Genesis API functions (`set_dofs_position`, `get_dofs_position`, `control_dofs_velocity`, etc.) expect:
        *   `dofs_idx_local`: A **Python list** of *all* DOF indices you want to interact with (e.g., `[0, 1, 2, 3, 4, 5, 6, ..., 17]` for Go2).
        *   `position`/`velocity`/etc.: A **NumPy array** of shape `(num_envs, num_dofs)` with `dtype=float32`, where `num_dofs` matches the length of the `dofs_idx_local` list. The order of values in the last dimension must correspond exactly to the order of indices in `dofs_idx_local`.
    *   **Best Practice:** Define `self.dof_indices` in your `__init__` by iterating through `entity.joints` and collecting all `dof_idx_local` values into a flat list. Use this list consistently for all Genesis DOF API calls.

4.  **Mapping RL Actions/Observations to Genesis DOFs:**
    *   Your RL agent likely only controls *actuated* joints (e.g., the 12 revolute joints of Go2), not the FREE joint.
    *   Define your `action_space` based on the number of *actuated* DOFs (e.g., 12).
    *   Define `self.actuated_dof_indices` (e.g., `self.dof_indices[6:]` for Go2).
    *   **Setting DOFs:** When calling `set_dofs_position` or `control_dofs_velocity`:
        *   Create a full NumPy array of shape `(num_envs, total_num_dofs)` (e.g., 18 for Go2).
        *   Fill the DOFs corresponding to the FREE joint (e.g., indices 0-5) with appropriate values (e.g., current state or zeros).
        *   Fill the DOFs corresponding to actuated joints (e.g., indices 6-17) with values derived from the RL action.
        *   Pass the full array and the full `dofs_idx_local` list to the Genesis API.
    *   **Getting DOFs:** When calling `get_dofs_position`/`velocity`:
        *   Call the Genesis API with the full `dofs_idx_local` list.
        *   Extract only the columns corresponding to the *actuated* DOFs (e.g., `[:, 6:]`) for use in your observation vector.

5.  **Initialization Order:**
    *   Initialize Genesis (`gs.init`).
    *   Create the scene (`gs.Scene`).
    *   Add entities (plane, URDF).
    *   **Build the scene (`scene.build(...)`).** This is crucial for finalizing the internal state.
    *   *After* building, get default poses/velocities and set the initial state using `set_dofs_position`/`velocity` with the full DOF list and correctly shaped NumPy arrays.
    *   Define action/observation spaces based on actuated DOFs.
    *   Call `self.reset()` at the end of `__init__` to ensure a clean starting state.

6.  **Reset Logic:**
    *   In `reset()`, use `set_dofs_position`/`velocity` with the full DOF list and correctly shaped NumPy arrays to set the desired initial state (e.g., default pose + noise).
    *   Ensure tensors are converted to NumPy float32 arrays on the CPU before passing to Genesis.

## Pitfalls and Debugging

*   **`TaichiIndexError: Field with dim X accessed with indices of dim Y`:** This is the most common error. It almost always means a mismatch between the `dofs_idx_local` list and the shape/dtype/order of the `position`/`velocity` array passed to the Genesis API.
    *   **Check:** Ensure `dofs_idx_local` is a Python list of *all* relevant DOFs.
    *   **Check:** Ensure the data array is a NumPy float32 array of shape `(num_envs, len(dofs_idx_local))`.
    *   **Check:** Ensure the order of values in the array matches the order of indices.
*   **Asset Not Found Errors:** Double- and triple-check all relative paths to mesh files within your URDF.
*   **Genesis State Issues:** Genesis can sometimes retain state unexpectedly.
    *   Use `pytest` fixtures with `scope="function"` for tests.
    *   Consider adding an `autouse=True` fixture to call `gs.destroy()` and `gs.init()` before/after each test function for maximum isolation.
*   **NumPy vs. PyTorch:** While Genesis *might* accept PyTorch tensors sometimes, the most reliable method (based on debugging and examples) is to pass **NumPy float32 arrays on the CPU**. Use `.cpu().numpy().astype('float32')`.
*   **Headless Mode:** When running without a display (e.g., server, CI), ensure `show_viewer=False` is passed to the `gs.Scene` constructor.
*   **Debugging:** Use `debug_print` (or similar) extensively to check shapes, dtypes, devices, and index lists right before calling Genesis API functions. Compare against minimal working examples.

By following these steps and being mindful of the pitfalls, you can successfully implement complex URDF-based environments in Genesis.
