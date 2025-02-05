import bmesh
import bpy
import math
import mathutils
import numpy as np

# ------------------------
# Parameters
# ------------------------
span = 1200.0       # mm (wing span)
root_chord = 230.0  # mm (root chord)
tip_chord = 90.7    # mm (tip chord)
naca_thickness = 0.12  # Thickness ratio for NACA 4412
num_points = 100    # Number of points per airfoil section

# Dihedral parameters along the wing
bend_start = 350    # Start of dihedral change (mm)
bend_end = 750      # End of dihedral change (mm)
root_dihedral = 0   # Degrees at root
mid_dihedral = 10   # Maximum dihedral angle (degrees)
tip_dihedral = 5    # Dihedral at tip

# Sharklet parameters
sharklet_height = 150         # Extension height (mm)
sharklet_tip_chord = tip_chord * 0.6  # Chord of the sharklet tip
# For softer edges, we will apply a bevel modifier later.
bevel_width = 8.0

# ------------------------
# Functions
# ------------------------

def create_airfoil_points(chord, thickness, y_pos, z_offset):
    """
    Generate a closed airfoil outline based on a NACA 4412 formula.

    Returns a list of (x, y, z) tuples.
    """
    pts = []
    m = 0.04  # Maximum camber (4% for NACA 4412)
    p = 0.4   # Location of maximum camber (40% for NACA 4412)

    def naca_4digit(x, m, p, t):
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        if x < p:
            yc = (m / (p**2)) * (2 * p * x - x**2)
        else:
            yc = (m / ((1 - p)**2)) * (1 - 2 * p + 2 * p * x - x**2)
        dyc_dx = (2 * m / (p**2)) * (p - x) if x < p else (2 * m / ((1 - p)**2)) * (p - x)
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        return xu, yu, xl, yl

    # Upper surface (from leading edge to trailing edge)
    for x_norm in np.linspace(0.0, 1.0, num_points):
        xu, yu, xl, yl = naca_4digit(x_norm, m, p, thickness)
        pts.append((xu * chord, y_pos, yu * chord + z_offset))

    # Lower surface (from trailing edge back to leading edge)
    for x_norm in np.linspace(1.0, 0.0, num_points)[1:]:
        xu, yu, xl, yl = naca_4digit(x_norm, m, p, thickness)
        pts.append((xl * chord, y_pos, yl * chord + z_offset))

    # Close the loop by repeating the first point
    pts.append(pts[0])
    return pts

def build_loft_mesh(sections):
    """
    Given a list of sections (each a list of (x, y, z) tuples),
    build a mesh by bridging corresponding vertices between adjacent sections.
    Assumes each section has the same number of vertices.
    """
    mesh = bpy.data.meshes.new("WingMesh")
    bm = bmesh.new()

    num_sections = len(sections)
    num_verts = len(sections[0])  # Number of vertices per section

    # Create vertices for all sections
    vertices = []
    for section in sections:
        vs = [bm.verts.new(mathutils.Vector(co)) for co in section]
        vertices.append(vs)
    bm.verts.ensure_lookup_table()

    # Create faces bridging each adjacent pair of sections
    for i in range(num_sections - 1):
        for j in range(num_verts - 1):
            v1 = vertices[i][j]
            v2 = vertices[i][j+1]
            v3 = vertices[i+1][j+1]
            v4 = vertices[i+1][j]
            try:
                face = bm.faces.new([v1, v2, v3, v4])
            except Exception as e:
                # Face exists or other issues; skip
                pass

    bm.faces.ensure_lookup_table()

    # Write bmesh data into a new mesh
    bm.to_mesh(mesh)
    bm.free()
    return mesh

# ------------------------
# Create Wing Sections
# ------------------------
sections = []     # List to hold cross-section points
z_offsets = []    # To maintain vertical offsets for each section

# We'll use 15 sections along the span
y_positions = np.linspace(0, span, 15)
prev_z = 0
prev_y = 0

for y in y_positions:
    # Compute dihedral angle
    if y <= bend_start:
        angle = root_dihedral
    elif y < bend_end:
        t = (y - bend_start) / (bend_end - bend_start)
        angle = root_dihedral + t * (mid_dihedral - root_dihedral)
    else:
        t = (y - bend_end) / (span - bend_end)
        angle = mid_dihedral + t * (tip_dihedral - mid_dihedral)

    delta_y = y - prev_y
    z_offset = prev_z + delta_y * math.tan(math.radians(angle))
    # Linearly interpolate chord from root to tip
    chord = root_chord + (tip_chord - root_chord) * (y / span)

    pts = create_airfoil_points(chord, naca_thickness, y, z_offset)
    sections.append(pts)
    z_offsets.append(z_offset)

    prev_z = z_offset
    prev_y = y

# ------------------------
# Add Sharklet Sections to Wingtip
# ------------------------
# Sharklet base uses the same chord as the wing tip
sharklet_base = create_airfoil_points(tip_chord, naca_thickness, span, z_offsets[-1])
# Intermediate section for smoother transition in the sharklet
mid_chord = (tip_chord + sharklet_tip_chord) / 2.0
sharklet_mid = create_airfoil_points(mid_chord, naca_thickness, span + sharklet_height / 2, z_offsets[-1] + sharklet_height / 2)
# Tip section for the sharklet
sharklet_tip = create_airfoil_points(sharklet_tip_chord, naca_thickness, span + sharklet_height, z_offsets[-1] + sharklet_height)

# Append sharklet sections to the wing sections
sections.extend([sharklet_base, sharklet_mid, sharklet_tip])

# ------------------------
# Build Mesh and Create Object
# ------------------------
wing_mesh = build_loft_mesh(sections)
wing_obj = bpy.data.objects.new("GullWing", wing_mesh)
bpy.context.collection.objects.link(wing_obj)

# Set object mode so modifiers can be applied
bpy.context.view_layer.objects.active = wing_obj
bpy.ops.object.select_all(action='DESELECT')
wing_obj.select_set(True)

# ------------------------
# Add Bevel Modifier for Soft, Rounded Edges
# ------------------------
bevel = wing_obj.modifiers.new(name="Bevel", type='BEVEL')
bevel.width = bevel_width
bevel.segments = 25  # Increase segments for smoother rounding
bevel.profile = 0.5
bevel.limit_method = 'NONE'  # Apply bevel broadly

# Optionally, apply the modifier to make changes permanent:
# bpy.ops.object.modifier_apply(modifier=bevel.name)

# ------------------------
# Recalculate Mesh & Update Scene
# ------------------------
wing_obj.data.update()
bpy.context.view_layer.update()

print("Gull-wing with sharklet and soft edges created in Blender using NACA 4412 profile.")
