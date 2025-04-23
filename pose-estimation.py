## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
import sys
from datetime import datetime
import os
import open3d as o3d
import matplotlib.pyplot as plt
import re


sys.path.append('/third-party/lang-segment-anything/lang_sam')
from lang_sam import LangSAM
from lang_sam.utils import draw_image

 
PROMPT = "extrusion."


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("No Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale 
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

print("Press SPACE to segment the image, and ESC with focus on the image window to quit the programm")

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (aligned_frames.get_depth_frame())  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        

        # Get instrinsics from aligned_depth_frame
        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # We use openCV for realtime streaming 
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
        cv2.imshow("Pose Estimation", color_image)
        key = cv2.waitKey(1)



        # Intrinsics ===========================================================
        # print(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
        # print(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
        # print(f"{0.0} {0.0} {1.0}\n")
        # ======================================================================

        
        if key & 0xFF == ord(" "):  

            print("Segmenting image")
        

            # Segmentation =========================================================
            model = LangSAM()
            image_pil = Image.fromarray(np.uint8(color_image)) 
            text_prompt = PROMPT
            results = model.predict([image_pil], [text_prompt])[0]
            labels = results['labels']
            print("Found " + str(len(labels)) + " objects: " + str(results["scores"]) )
            # ==================================================================

            if len(labels) != 0 :

                annoted_image = draw_image(
                    color_image,
                    results["masks"],
                    results["boxes"],
                    results["scores"],
                    results["labels"],
                )

                cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                cv2.setWindowTitle("Pose Estimation", "Annoted Image")
                cv2.imshow("Pose Estimation", annoted_image)
                key = cv2.waitKey(0)



                highest_score_index = results["scores"].argmax()  # index of the highest scoring mask
                print("index " + str(highest_score_index) + " of score " + str(results["scores"]))
                best_mask = results["masks"][highest_score_index]*(2**16-1)
                best_mask  = best_mask.astype('uint16')

                # plt.imshow(best_mask)
                # plt.show()

                cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                cv2.setWindowTitle("Pose Estimation", "Best Mask")
                cv2.imshow("Pose Estimation", best_mask)
                key = cv2.waitKey(0)

                # best_mask = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2RGB)
                
                print("color image type : " + str(type(color_image)) + str(color_image.dtype) + ","+ str(color_image.shape))
                print("mask  image type : " + str(type(best_mask)) + str(best_mask.dtype) + ","+ str(best_mask.shape))
                print("depth image:" + str(depth_image.dtype))

               

                # depth = cv2.applyColorMap(depth_image, cv2.COLOR_GRAY2RGB)
                # masked_depth_viewer = cv2.applyColorMap(masked_depth.astype('uint8'),cv2.COLORMAP_JET)

                print("bitwise operation types:" + str(depth_image.dtype) + " and with " + str(best_mask.dtype))

                masked_depth = np.bitwise_and(depth_image,best_mask)

                print("Masked depth image type:" + str(masked_depth.dtype))

                plt.title("depth")
                plt.imshow(depth_image)
                plt.show()

                plt.title("masked depth")
                plt.imshow(masked_depth)
                plt.show()


                
                # masked_depth_viewer = cv2.applyColorMap(masked_depth.astype('uint8'),cv2.COLORMAP_JET)
                # cv2.imshow("Pose Estimation", masked_depth_viewer)
                # key = cv2.waitKey(0)


                # point cloud ==================================================
                cam = o3d.camera.PinholeCameraIntrinsic(640,480,intrinsics.fx,intrinsics.fy,intrinsics.ppx,intrinsics.ppy)
                # cam.intrinsic_matrix =  [[intrinsics.fx, 0.00, intrinsics.ppx] , [0.00, intrinsics.fy, intrinsics.ppy], [0.00, 0.00, 1.00]] #! perhaps the culprit
                print(cam)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image),
                                                                        o3d.geometry.Image(masked_depth), 
                                                                        convert_rgb_to_intensity = False
                                                                        # ,depth_scale = (1.0/depth_scale)
                                                                        )
                
                pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

                # flip the orientation, so it looks upright, not upside-down
                pointcloud = pointcloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

                o3d.visualization.draw_geometries([pointcloud]) 

                # save point cloud =============================================
                current_dir_path = os.path.dirname(os.path.realpath(__file__))
                subfolder_pcd = os.path.join(current_dir_path, "point_clouds")

                if not os.path.exists(subfolder_pcd):
                    os.makedirs(subfolder_pcd)

                # construct name of file
                date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
                label = re.sub('[.]', '', PROMPT)
                pointcld_name = label + "-"+ str(date) + ".ply"

                pointcld_path = os.path.join(subfolder_pcd, pointcld_name)
                o3d.io.write_point_cloud(pointcld_path, pointcloud, format='auto', write_ascii=False, compressed=False, print_progress=False)



        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:

            cv2.destroyAllWindows()

            break;
         
finally:
    pipeline.stop()
