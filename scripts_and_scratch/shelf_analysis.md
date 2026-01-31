# Shelf Structure Analysis

## SDF File Structure

From `shelves.sdf`:
- **Main link**: `shelves_body` (origin of the shelves model)
- **Secondary link**: `top_and_bottom` (connected to `shelves_body` via fixed joint, shares same origin)
- **Lower shelf** (`shelf_lower`):
  - Pose relative to `top_and_bottom`: `0 0 -0.13115 0 0 0`
  - Size: `0.3 0.6 0.016` (width x depth x height in meters)
  - Center of shelf is at z = -0.13115 relative to `shelves_body`
  - Top surface of shelf is at z = -0.13115 + 0.016/2 = -0.13115 + 0.008 = **-0.12315** relative to `shelves_body`
- **Upper shelf** (`shelf_upper`):
  - Pose relative to `top_and_bottom`: `0 0 0.13115 0 0 0`
  - Size: `0.3 0.6 0.016` (width x depth x height in meters)
  - Center of shelf is at z = 0.13115 relative to `shelves_body`
  - Top surface of shelf is at z = 0.13115 + 0.016/2 = 0.13115 + 0.008 = **0.13915** relative to `shelves_body`

## World Frame Transformation

From `obs_modeling_scenario.yaml`:
- `shelves_body` is welded to world at:
  - Translation: `[0.5, 0, 0.3995]`
  - Rotation: `[0, 0, 180]` degrees (rotation around z-axis)
  
A 180° rotation around z-axis transforms coordinates as:
- x → -x
- y → -y  
- z → z (unchanged)

## Lower Shelf Position in World Frame

The lower shelf (`shelf_lower`) center is at `[0, 0, -0.13115]` in `shelves_body` frame.

After 180° rotation around z:
- Since x=0 and y=0, they remain 0
- z remains -0.13115

In world frame:
- x = 0.5 (shelves_body x position)
- y = 0 (shelves_body y position)
- z = 0.3995 + (-0.12315) = **0.27635** (top surface of shelf)

## Upper Shelf Position in World Frame

The upper shelf (`shelf_upper`) center is at `[0, 0, 0.13115]` in `shelves_body` frame.

After 180° rotation around z:
- Since x=0 and y=0, they remain 0
- z remains 0.13115

In world frame:
- x = 0.5 (shelves_body x position)
- y = 0 (shelves_body y position)
- z = 0.3995 + 0.13915 = **0.53865** (top surface of shelf)

## Recommended VOXEL_GRID_SHELF_LOWER_ORIGIN

To place the origin exactly on top of the lower shelf:
```python
VOXEL_GRID_SHELF_LOWER_ORIGIN = np.array([0.5, 0.0, 0.27635])
```

This places the origin at:
- x = 0.5 m (center of shelf in x-direction)
- y = 0.0 m (center of shelf in y-direction)  
- z = 0.27635 m (top surface of lower shelf)
