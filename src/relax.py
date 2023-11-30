import copy


def multiply_box_edges(box, factor):
    return tuple(
        map(
            lambda x: x / 2,
            (
                (box[0] + box[2]) - (box[2] - box[0]) * factor,
                (box[1] + box[3]) - (box[3] - box[1]) * factor,
                (box[0] + box[2]) + (box[2] - box[0]) * factor,
                (box[1] + box[3]) + (box[3] - box[1]) * factor,
            ),
        )
    )


def clamp_to_bounding_box(box, bounding_box):
    return tuple((max if i < 2 else min)(box[i], bounding_box[i]) for i in range(4))


def modify_object_in_place(
    root,
    tag,
    obj,
    outer_multiplier,
    clamp_outer_boundary_to_image_size,
    inner_multiplier,
    clamp_inner_boundary_to_outer_boundary,
    swap_inner_boundary_corners,
):
    size_element = root.find("size")
    width = float(size_element.find("width").text)
    height = float(size_element.find("height").text)

    name_element = obj.find("name")
    name = name_element.text

    box_element = obj.find("bndbox")
    x_min_element = box_element.find("xmin")
    y_min_element = box_element.find("ymin")
    x_max_element = box_element.find("xmax")
    y_max_element = box_element.find("ymax")

    x_min = float(x_min_element.text)
    y_min = float(y_min_element.text)
    x_max = float(x_max_element.text)
    y_max = float(y_max_element.text)

    assert x_min < x_max
    assert y_min < y_max

    obj2 = copy.deepcopy(obj)

    outer_boundary = multiply_box_edges((x_min, y_min, x_max, y_max), outer_multiplier)
    final_outer_boundary = (
        clamp_to_bounding_box(
            outer_boundary,
            (0, 0, width, height),
        )
        if clamp_outer_boundary_to_image_size
        else outer_boundary
    )
    # print("final_outer_boundary: {}".format(final_outer_boundary))
    inner_boundary = multiply_box_edges((x_min, y_min, x_max, y_max), inner_multiplier)
    # print("inner_boundary: {}".format(inner_boundary))
    semifinal_inner_boundary = (
        clamp_to_bounding_box(
            inner_boundary,
            final_outer_boundary,
        )
        if clamp_inner_boundary_to_outer_boundary
        else inner_boundary
    )
    # print("semifinal_inner_boundary: {}".format(semifinal_inner_boundary))
    final_inner_boundary = (
        (semifinal_inner_boundary[2:] + semifinal_inner_boundary[:2])
        if swap_inner_boundary_corners
        else semifinal_inner_boundary
    )
    # print("final_inner_boundary: {}".format(final_inner_boundary))

    name_element.text = "{} {} i".format(name, tag)
    (
        x_min_element.text,
        y_min_element.text,
        x_max_element.text,
        y_max_element.text,
    ) = map(
        str,
        final_inner_boundary,
    )

    obj2.find("name").text = "{} {} o".format(name, tag)
    box_element2 = obj2.find("bndbox")
    (
        box_element2.find("xmin").text,
        box_element2.find("ymin").text,
        box_element2.find("xmax").text,
        box_element2.find("ymax").text,
    ) = map(
        str,
        final_outer_boundary,
    )
    root.append(obj2)
