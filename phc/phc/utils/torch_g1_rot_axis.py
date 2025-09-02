import xml.etree.ElementTree as ET
import torch
import os

def extract_joint_axes(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    axes = []
    joint_names = []

    for joint in root.findall("joint"):
        jtype = joint.attrib.get("type")
        if jtype == "revolute" or jtype == "continuous":  # skip fixed
            name = joint.attrib["name"]
            axis_tag = joint.find("axis")
            if axis_tag is not None:
                xyz = [float(v) for v in axis_tag.attrib["xyz"].split()]
            else:
                xyz = [0, 0, 1]  # default if no axis specified
            axes.append(xyz)
            joint_names.append(name)
    
    axes_tensor = torch.tensor([axes], dtype=torch.float32)  # shape (1, N, 3)
    return joint_names, axes_tensor

if __name__ == "__main__":

    # Example usage:
    urdf_path = "resources/robots/g1/urdf/g1_23dof.urdf"   # <-- replace with your actual file path
    joint_names, G1_ROTATION_AXIS = extract_joint_axes(urdf_path)

    print("Found", len(joint_names), "joints")
    print("Joint names:", joint_names)
    print("Rotation axes tensor shape:", G1_ROTATION_AXIS.shape)
    # print("Rotation axes tensor:", G1_ROTATION_AXIS)

    tree = ET.parse("resources/robots/g1/g1.xml")
    root = tree.getroot()

    body_list = root.findall(".//body")
    print("Found", len(body_list), "bodies in MJCF file")
    body_names = [body.attrib.get("name") for body in body_list]
    print("Body names:", body_names)