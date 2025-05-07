import bpy

# Create a new cube if it doesn't exist
if "Cube" not in bpy.data.objects:
    bpy.ops.mesh.primitive_cube_add()

cube = bpy.data.objects["Cube"]

# Set up animation
cube.animation_data_clear()  # Clear existing animation data

# Insert keyframes for rotation
cube.rotation_euler = (0, 0, 0)
cube.keyframe_insert(data_path="rotation_euler", frame=1)

cube.rotation_euler = (0, 0, 1000 * (3.14159 / 180))  # Convert degrees to radians
cube.keyframe_insert(data_path="rotation_euler", frame=100)

# Set interpolation to linear
for fcurve in cube.animation_data.action.fcurves:
    for keyframe in fcurve.keyframe_points:
        keyframe.interpolation = 'LINEAR'


# move the cube to in x axis 1 m over 100 frames
