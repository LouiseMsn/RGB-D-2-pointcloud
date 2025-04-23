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
import argparse

# Filepath inclusion of SAM
sys.path.append('/third-party/lang-segment-anything/lang_sam')
from lang_sam import LangSAM
from lang_sam.utils import draw_image

parser = argparse.ArgumentParser(
                    prog='RGB-D to pointcloud',
                    description='Converts an object to a pointcloud using its name, a realsense camera, lang-segement-anything and open3D.',
                    epilog='---')
parser.add_argument('-d','--debug',help='Shows the details of the image operations & segmentation results',action="store_true")
args = parser.parse_args()

if args.debug:
    print("DEBUG MODE")

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

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

print("Press SPACE to segment the image, and ESC with focus on the image window to quit the programm")

first_frame = True # to print once the intrinsics
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


        if args.debug and first_frame:
            print("Camera Intrinsics :")
            print(f"{intrinsics.fx} {0.0} {intrinsics.ppx}")
            print(f"{0.0} {intrinsics.fy} {intrinsics.ppy}")
            print(f"{0.0} {0.0} {1.0}")
            print("\n")
            first_frame = False
        
        if key & 0xFF == ord(" "):  
            # Prompt for the object searched 
            prompt = input("Enter the name/description of the object : ") + "."
            print("Segmenting image and searching for : "+ prompt)
    
            # Segmentation =========================================================
            model = LangSAM()
            image_pil = Image.fromarray(np.uint8(color_image)) 
            results = model.predict([image_pil], [prompt])[0]
            labels = results['labels']
            if args.debug:
                print("Found " + str(len(labels)) + " objects: " + str(results["scores"]) )
            # ==================================================================

            if len(labels) == 0 :
                print("No " + str(prompt) + " was found, take another picture.")

            else :

                annoted_image = draw_image(
                    color_image,
                    results["masks"],
                    results["boxes"],
                    results["scores"],
                    results["labels"],
                )

                if args.debug :
                    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                    cv2.setWindowTitle("Pose Estimation", "Annoted Image")
                    cv2.imshow("Pose Estimation", annoted_image)
                    key = cv2.waitKey(0)



                highest_score_index = results["scores"].argmax()  # index of the highest scoring mask
                print("index " + str(highest_score_index) + " of score " + str(results["scores"]))
                best_mask = results["masks"][highest_score_index]*(2**16-1)
                best_mask  = best_mask.astype('uint16')

                if args.debug :
                    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                    cv2.setWindowTitle("Pose Estimation", "Best Mask")
                    cv2.imshow("Pose Estimation", best_mask)
                    key = cv2.waitKey(0)

                    print("color image type : " + str(type(color_image)) + str(color_image.dtype) + ","+ str(color_image.shape))
                    print("depth image type : " + str(type(depth_image)) + str(depth_image.dtype) + ","+ str(depth_image.shape))
                    print("mask  image type : " + str(type(best_mask)) + str(best_mask.dtype) + ","+ str(best_mask.shape))
                    
                masked_depth = np.bitwise_and(depth_image,best_mask)


                if args.debug :
                    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                    cv2.setWindowTitle("Pose Estimation", "Depth")
                    cv2.imshow("Pose Estimation", depth_image)
                    key = cv2.waitKey(0)

                    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                    cv2.setWindowTitle("Pose Estimation", "Masked Depth")
                    cv2.imshow("Pose Estimation", masked_depth)
                    key = cv2.waitKey(0)

                # point cloud ==================================================
                cam = o3d.camera.PinholeCameraIntrinsic(640,480,intrinsics.fx,intrinsics.fy,intrinsics.ppx,intrinsics.ppy)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image),
                                                                        o3d.geometry.Image(masked_depth), 
                                                                        convert_rgb_to_intensity = False
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
                label = re.sub('[.]', '', prompt)
                pointcld_name = label + "-"+ str(date) + ".ply"

                pointcld_path = os.path.join(subfolder_pcd, pointcld_name)
                o3d.io.write_point_cloud(pointcld_path, pointcloud, format='auto', write_ascii=False, compressed=False, print_progress=False)

                if args.debug :
                    print("Poincloud written to: " + str(pointcld_path) + str(pointcld_name))



        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:

            cv2.destroyAllWindows()

            break;
         
finally:
    pipeline.stop()
