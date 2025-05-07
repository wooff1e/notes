import bpy
import numpy as np

# Set up the random curve for x-axis movement
np.random.seed(42)  # For reproducibility
frames = 100
x_positions = np.cumsum(np.random.uniform(-0.05, 0.05, frames))  # Random walk
x_positions = (x_positions - x_positions.min()) / (x_positions.max() - x_positions.min())  # Normalize to 0-1
x_positions *= 1  # Scale to 1 meter

# Create a cube
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0, 0, 0))
cube = bpy.context.object

# Get the animation data and create an F-Curve for the x-axis location
if not cube.animation_data:
    cube.animation_data_create()
if not cube.animation_data.action:
    cube.animation_data.action = bpy.data.actions.new(name="CubeAction")

action = cube.animation_data.action
fcurve = action.fcurves.new(data_path="location", index=0)

# Add keyframe points to the F-Curve
fcurve.keyframe_points.add(count=frames)
for frame, x in enumerate(x_positions):
    fcurve.keyframe_points[frame].co = (frame + 1, x)  # Frame and value
    fcurve.keyframe_points[frame].interpolation = 'LINEAR'  # Set interpolation type

# Set the frame range for the animation
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = frames
