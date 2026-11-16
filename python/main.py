#!/usr/bin/env python

"""
Create 3D model file of PET scanner defined in PETSIRD list mode file.
Known format that work:
    - ply, stl, obj
With color? 
    - ply

3D viewer tried:
    meshlab: Work with color 
    blender: Voodoo magic for color?
    paraview: Work with color sometime?

Potential issues:
    - If the number of detectors is not the same for each type of detectors, color stuff will break.

"""


#########################################################################################
# Import
#########################################################################################
import sys
import os
import numpy as np
import numpy.typing as npt
import petsird
import petsird.helpers.geometry
import trimesh
import argparse


#########################################################################################
# Constants
#########################################################################################
crystal_color = np.array([255, 40, 40], dtype=np.uint8)

# Face colors for --color-faces
front_face_color = np.array([255, 255, 255, 255], dtype=np.uint8)  # white
back_face_color = np.array([0, 0, 0, 255], dtype=np.uint8)  # black
# side faces, named by the edge of the front face they are attached to.
# For the PETSIRD generator geometry, edges 1-2 and 3-0 lie at z max / z min
# (axial faces, red tones), edges 0-1 and 2-3 face the neighbouring crystals
# in the ring (tangential faces, blue tones).
side_face_colors = {
    (1, 2): np.array([220, 0, 0, 255], dtype=np.uint8),  # red: +z (z max)
    (3, 0): np.array([255, 100, 100, 255], dtype=np.uint8),  # light red: -z (z min)
    (0, 1): np.array([0, 60, 220, 255], dtype=np.uint8),  # blue
    (2, 3): np.array([130, 190, 255, 255], dtype=np.uint8),  # light blue
}


#########################################################################################
# Methods
#########################################################################################

def create_box_from_vertices(vertices):
    # Define faces using the indices of vertices that make up each face.
    # PETSIRD BoxShape: corners 0-3 = front face (towards scanner axis),
    # corners 4-7 = back face; side faces are named by the front-face edge they
    # are attached to. The +z/-z and ring-neighbour directions hold for the
    # PETSIRD generator geometry (see side_face_colors).
    faces = [
        [0, 1, 2],
        [0, 2, 3],  # front face (corners 0-3, towards scanner axis)
        [4, 6, 5],
        [4, 7, 6],  # back face (corners 4-7, away from scanner axis)
        [0, 3, 7],
        [0, 7, 4],  # side at edge 3-0 (axial, -z) (bottom)
        [1, 6, 2],
        [1, 5, 6],  # side at edge 1-2 (axial, +z) (top)
        [0, 5, 1],
        [0, 4, 5],  # side at edge 0-1 (towards ring neighbour)
        [3, 2, 6],
        [3, 6, 7],  # side at edge 2-3 (towards other ring neighbour)
    ]

    # Create and return a Trimesh object
    box = trimesh.Trimesh(vertices=vertices, faces=faces)

    return box


def set_face_colors(det_mesh):
    """
    Color the faces of a detecting element created by create_box_from_vertices():
    PETSIRD front face (corners 0-3, the first 2 triangles) white,
    PETSIRD back face (corners 4-7, the next 2 triangles) black,
    side faces by the front-face edge they are attached to:
    1-2 red (+z), 3-0 light red (-z), 0-1 blue, 2-3 light blue.
    """
    colors = np.zeros((len(det_mesh.faces), 4), dtype=np.uint8)
    colors[0:2] = front_face_color  # front face (corners 0, 1, 2, 3)
    colors[2:4] = back_face_color  # back face  (corners 4, 5, 6, 7)
    colors[4:6] = side_face_colors[(3, 0)]  # side at edge 3-0 (corners 0, 3, 7, 4), -z
    colors[6:8] = side_face_colors[(1, 2)]  # side at edge 1-2 (corners 1, 2, 6, 5), +z
    colors[8:10] = side_face_colors[(0, 1)]  # side at edge 0-1 (corners 0, 1, 5, 4)
    colors[10:12] = side_face_colors[(2, 3)]  # side at edge 2-3 (corners 3, 2, 6, 7)
    det_mesh.visual.face_colors = colors
    return det_mesh


def extract_detector_eff(show_det_eff, header):
    if not show_det_eff:
        return None

    eff = header.scanner.detection_efficiencies.detection_bin_efficiencies
    if eff is None:
        raise ValueError(
            "The scanner detection efficiencies are not defined. Correct this or remove the show_det_eff flag."
        )
    return eff

    
def set_detector_color(det_mesh, detector_efficiencies, type_of_module, num_modules, mod_i, num_det_in_module, det_i, random_color):
    if random_color == True:
        color = np.random.randint(0, 255, size=3)
    elif detector_efficiencies is not None:
        eff = np.reshape(detector_efficiencies[type_of_module], shape=(num_modules, num_det_in_module, -1))
        color = (crystal_color * np.mean(eff[mod_i, det_i, :]))
    else:
        color = crystal_color
    
    f_color = np.array([color[0], color[1], color[2], 50]).astype(
        np.uint8
    )
    det_mesh.visual.face_colors = f_color

    return det_mesh



def set_module_color(
        module_mesh, detector_efficiencies, type_of_module, num_modules, mod_i, num_det_in_module, random_color
):
    if random_color == True:
        color = np.random.randint(0, 255, size=3)
    elif detector_efficiencies is not None:
        # Mean of the detector efficiencies in the current module
        eff = np.reshape(detector_efficiencies[type_of_module], shape=(num_modules, num_det_in_module, -1))
        color = crystal_color * np.mean(
            eff[mod_i, :, :])
    else:
        color = crystal_color

    f_color = np.array([color[0], color[1], color[2], 50]).astype(np.uint8)

    module_mesh.visual.face_colors = f_color

    return module_mesh


def create_mesh(header, modules_only=False, show_det_eff=False, random_color=False, fov=None,
                modules_indices=None, color_faces=False):
    """
    Create 3D model of PET scanner defined in PETSIRD list mode file.
    """
    detector_efficiencies = extract_detector_eff(show_det_eff, header)

    shapes = []
    # draw all crystals
    module_count = 0
    for type_of_module in range(len(header.scanner.scanner_geometry.replicated_modules)):
        rep_module = header.scanner.scanner_geometry.replicated_modules[type_of_module]
        num_modules = len(rep_module.transforms)
        det_els = (
            rep_module.object.detecting_elements
        )  # Get all the detecting elements modules
        for mod_i in range(len(rep_module.transforms)):
            module_count += 1
            if modules_indices is not None and (module_count - 1) not in modules_indices:
                continue

            vertices = []  # If showing modules only
            mod_transform = rep_module.transforms[mod_i]
            num_det_in_module = len(det_els.transforms)
            for det_i in range(num_det_in_module):
                transform = det_els.transforms[det_i]
                box: petsird.BoxShape = petsird.helpers.geometry.transform_BoxShape(
                    petsird.helpers.geometry.mult_transforms(
                        [mod_transform, transform]),
                    det_els.object.shape,
                )
                corners = []
                for boxcorner in box.corners:
                    corners.append(boxcorner.c)

                if not modules_only:
                    det_mesh = create_box_from_vertices(corners)
                            
                    if color_faces:
                        det_mesh = set_face_colors(det_mesh)
                    else:
                        det_mesh = set_detector_color(det_mesh, detector_efficiencies, type_of_module, num_modules, mod_i, num_det_in_module, det_i, random_color)

                    shapes.append(det_mesh)
                else:
                    vertices.append(corners)
            if modules_only:
                vertices_reshaped = np.array(vertices).reshape((-1, 3))
                module_mesh = trimesh.convex.convex_hull(vertices_reshaped)

                module_mesh = set_module_color(
                    module_mesh,
                    detector_efficiencies,
                    type_of_module,
                    num_modules,
                    mod_i,
                    num_det_in_module,
                    random_color
                )

                shapes.append(module_mesh)

    if fov is not None:
        shapes.append(
            trimesh.creation.cylinder(radius=fov[0], height=fov[1])
        )
    return trimesh.util.concatenate(shapes)


#########################################################################################
# Main
#########################################################################################
def parserCreator():
    parser = argparse.ArgumentParser(description="Create 3D model of PET scanner defined in PETSIRD list mode file.")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="File to read from, or stdin if omitted",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="File to write (with the appropriate file format extension)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        nargs=2,
        default=None,
        required=False,
        help="Add a cylindrical FOV, as radius and height",
    )
    parser.add_argument(
        "--modules-only",
        action="store_true",
        dest="modules_only",
        default=False,
        required=False,
        help="Generate shapes for modules only",
    )
    parser.add_argument(
        "--show-det-eff",
        action="store_true",
        dest="show_det_eff",
        default=False,
        required=False,
        help="Change color following detector effeciency",
    )
    parser.add_argument(
        "--random-color",
        action="store_true",
        dest="random_color",
        default=False,
        required=False,
        help="Force random color for easier debug position",
    )
    parser.add_argument(
        "--num-modules",
        type=int,
        dest="num_modules",
        default=0,
        required=False,
        help="Max number of modules to convert",
    )
    parser.add_argument(
        "--skip-modules",
        type=int,
        dest="skip_modules",
        default=0,
        required=False,
        help="Skip every xx modules",
    )
    parser.add_argument(
        "--color-faces",
        action="store_true",
        dest="color_faces",
        default=False,
        required=False,
        help="Color the faces of each detecting element: front face (corners 0-3) white, "
             "back face (corners 4-7) black, side faces by front-face edge: "
             "red (edge 1-2, +z), light red (3-0, -z), blue (0-1), light blue (2-3)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parserCreator()
    if args.skip_modules < 0:
        sys.exit('skip_modules should be >= 0')

    file = None
    if args.input is None:
        file = sys.stdin.buffer
    else:
        file = open(args.input, "rb")
    output_fname = args.output

    with petsird.BinaryPETSIRDReader(file) as reader:
        header = reader.read_header()
        module_indices = range(0, args.num_modules * (args.skip_modules + 1), args.skip_modules + 1) if args.num_modules > 0 else None
        mesh = create_mesh(header, args.modules_only, args.show_det_eff, args.random_color, args.fov,
                          module_indices, args.color_faces
        )
        mesh.export(output_fname)

        # Forced to do this
        for time_block in reader.read_time_blocks():
            pass
