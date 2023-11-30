import pathlib
import argparse
import os
import xml.etree.ElementTree as ET
import relax


def main():
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_pascal_voc_xml_files",
        nargs="*",
        help="Comma-separated list of input Pascal POC XML files.",
    )
    parser.add_argument(
        "--output_pascal_voc_xml_dir",
        help="Where to output pascal voc XML files.",
    )
    parser.add_argument(
        "--inner_multiplier",
        type=float,
        help="Usually <= 1; inner boundary will have length width * inner_multiplier and be centered. If negative that will have the effect of marking the inner boundary as missing, unless the boundary was empty to start with, in which case the point it represents is reflected on both axes. Can also be '[+/-]inf'.",
        default=1,
    )
    parser.add_argument(
        "--outer_multiplier",
        type=float,
        help="Usually >= 1; outer boundary will have length width * outer_multiplier and be centered.  If negative that will have the effect of marking the inner boundary as missing, unless the boundary was empty to start with, in which case the point it represents is reflected on both axes. Can also be '[+/-]inf'.",
        default=1,
    )
    parser.add_argument(
        "--print0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, print a null character after processing each file.",
    )
    parser.add_argument(
        "--clamp_outer_boundary_to_image_size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to clamp to image size.",
    )
    parser.add_argument(
        "--clamp_inner_boundary_to_outer_boundary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to clamp inner boundary to (streched and clamped) outer boundary.",
    )
    parser.add_argument(
        "--swap_inner_boundary_corners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to swap inner boundary [:2] with [2:]. Runs after clamp_inner_boundary_to_outer_boundary.",
    )
    args = parser.parse_args()

    print("Images: {}".format(len(args.input_pascal_voc_xml_files)), flush=True)
    for input_pascal_voc_xml_file in args.input_pascal_voc_xml_files:
        os.makedirs(args.output_pascal_voc_xml_dir, exist_ok=True)
        pure_posix_path = pathlib.PurePosixPath(input_pascal_voc_xml_file)
        tree = ET.parse(input_pascal_voc_xml_file)
        modify_in_place(
            args.outer_multiplier,
            args.inner_multiplier,
            args.clamp_outer_boundary_to_image_size,
            args.clamp_inner_boundary_to_outer_boundary,
            args.swap_inner_boundary_corners,
            tree,
        )
        output_file_path = os.path.join(
            args.output_pascal_voc_xml_dir, pure_posix_path.name
        )
        print(output_file_path)
        tree.write(output_file_path)
        if args.print0:
            print("\x00", flush=True)


def modify_in_place(
    outer_multiplier,
    inner_multiplier,
    clamp_outer_boundary_to_image_size,
    clamp_inner_boundary_to_outer_boundary,
    swap_inner_boundary_corners,
    tree
):
    root = tree.getroot()
    for index, obj in enumerate(root.findall("object")):
        relax.modify_object_in_place(
            root, index, obj,
            outer_multiplier,
            clamp_outer_boundary_to_image_size,
            inner_multiplier,
            clamp_inner_boundary_to_outer_boundary,
            swap_inner_boundary_corners
        )


if __name__ == "__main__":
    main()
