import numpy as np

max_id = -1
floor_id = 0

table_id = 2
mug_id = 3
rack_id = 16777218

base_id = 1
link_0_id = 16777217
link_1_id = 33554433
link_2_id = 50331649
link_3_id = 83886081
link_4_id = 100663297

gripper_0_id = 117440513
gripper_1_id = 150994945
finger_0_id = 167772161
finger_1_id = 184549377
gripper_ids = [gripper_0_id, gripper_1_id, finger_0_id, finger_1_id]


def get_clouds(cams, occlusion=False):
    cloud_classes = []
    cloud_points = []
    cloud_colors = []
    if occlusion:
        cam_indices = list(np.random.choice(len(cams.cams), 3, replace=False))
    else:
        cam_indices = list(np.arange(len(cams.cams)))
    for i, cam in enumerate(cams.cams):
        if i in cam_indices:
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            points, colors = cam.get_pcd(
                in_world=True,
                rgb_image=rgb,
                depth_image=depth,
                depth_min=0.0,
                depth_max=np.inf,
            )

            cloud_points.append(points)
            cloud_colors.append(colors)
            cloud_classes.append(seg.flatten())

    return cloud_points, cloud_colors, cloud_classes


def get_object_clouds(cams, ids=[mug_id, rack_id, gripper_ids], occlusion=False):
    cloud_classes = []
    cloud_points = []
    cloud_colors = []
    if occlusion:
        cam_indices = list(np.random.choice(len(cams.cams), 3, replace=False))
    else:
        cam_indices = list(np.arange(len(cams.cams)))
    for i, cam in enumerate(cams.cams):
        if i in cam_indices:
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            points, colors = cam.get_pcd(
                in_world=True,
                rgb_image=rgb,
                depth_image=depth,
                depth_min=0.0,
                depth_max=np.inf,
            )
            classes = seg.flatten()  # flat_seg
            for j, obj_ids in enumerate(ids):
                mask = np.isin(classes, obj_ids)
                cloud_points.append(points[mask])
                cloud_colors.append(colors[mask])
                cloud_classes.append(j * np.ones(mask.sum()))

    cloud_points = np.concatenate(cloud_points, axis=0)
    cloud_colors = np.concatenate(cloud_colors, axis=0)
    cloud_classes = np.concatenate(cloud_classes, axis=0)
    return cloud_points, cloud_colors, cloud_classes


def get_object_clouds_from_demo_npz(grasp_data, ids=[mug_id, rack_id, gripper_ids]):
    print("-------------inside get_object_clouds_from_demo_npz()-------------")
    grasp_data_dict = dict(grasp_data)
    cloud_points = grasp_data_dict["object_pointcloud"]
    seg = grasp_data_dict["seg"]  # 4,1
    obj_pose_world = grasp_data_dict["obj_pose_world"]
    l_seg = []
    for i in range(4):
        l_seg.append(seg[i, 0])
    l_seg_cat = np.concatenate(l_seg)
    print("cloud_points.shape:{}".format(cloud_points.shape))
    print("l_seg_cat.shape:{}".format(l_seg_cat.shape))
    print("obj_pose_world.shape:{}".format(obj_pose_world.shape))

    cloud_classes = []
    cloud_points = []
    cloud_colors = []

    cloud_points = np.concatenate(cloud_points, axis=0)
    cloud_colors = np.concatenate(cloud_colors, axis=0)
    cloud_classes = np.concatenate(cloud_classes, axis=0)

    return cloud_points, cloud_colors, cloud_classes
