import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import cv2
from matplotlib import pyplot as plt
from calibration import draw_keypoints_and_match, drawlines, RANSAC_F_matrix, calculate_E_matrix, extract_camerapose, disambiguate_camerapose
from rectification import rectification
from correspondence import ssd_correspondence
from depth import disparity_to_depth
x = random.randint(0,50)
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# Load and preprocess left and right images



def main():
    st.title("Depth Estimation")
    st.write("Upload an image to find depth.")



    uploaded_file = st.file_uploader("Choose a right image (grayscale)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        right_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if right_img is not None:
            st.image(right_img, caption='Right Image', use_column_width=True, channels="GRAY")

            # Perform stereo matching to obtain disparity map
            # Use StereoSGBM for better accuracy
            stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
            disparity = stereo.compute(right_img, right_img)  # Pass right image as both left and right image

            # Normalize disparity map for visualization
            min_disp = disparity.min()
            max_disp = disparity.max()
            disparity = np.uint8(255 * (disparity - min_disp) / (max_disp - min_disp))

            #st.write("Disparity Map")
            #st.image(disparity, caption='Disparity Map', use_column_width=True, channels="GRAY")

            # Reconstruct left image from disparity
            h, w = right_img.shape[:2]
            left_img = np.zeros((h, w), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    disp = disparity[y, x]
                    if x - disp >= 0:
                        left_img[y, x - disp] = right_img[y, x]

            # Interpolate missing values
            left_img = cv2.inpaint(left_img, (left_img == 0).astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            cv2.imwrite('generated_left.png', left_img)
            filename = 'leftimage1.jpg'
            filename1 = 'rightimage1.jpg'
                
                # Using cv2.imwrite() method 
                # Saving the image 
            cv2.imwrite(filename, left_img) 
            cv2.imwrite(filename1, right_img) 

            st.write("Generated Left Image")
            st.image(left_img, caption='Generated Left Image', use_column_width=True, channels="GRAY")
        else:
            st.write("Error: Cannot read the image file.")
    
            # Filename 
    
        if st.button("Depth Estimation!"):
            #number = int(input("Please enter the dataset number (1/2/3) to use for calculating the depth map\n"))
            number = 1
            img1 = cv2.imread(f"leftimage1.jpg", 0)
            img2 = cv2.imread(f"rightimage1.jpg", 0)

            width = int(img1.shape[1]* 0.3) # 0.3
            height = int(img1.shape[0]* 0.3) # 0.3

            img1 = cv2.resize(img1, (width, height), interpolation = cv2.INTER_AREA)
            # img1 = cv2.GaussianBlur(img1,(5,5),0)
            img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)
            # img2 = cv2.GaussianBlur(img2,(5,5),0)
            
            #__________________Camera Parameters________________________________
            K11 = np.array([[5299.313,  0,   1263.818], 
                        [0,      5299.313, 977.763],
                        [0,          0,       1   ]])
            K12 = np.array([[5299.313,   0,    1438.004],
                        [0,      5299.313,  977.763 ],
                        [0,           0,      1     ]])

            K21 = np.array([[4396.869, 0, 1353.072],
                            [0, 4396.869, 989.702],
                            [0, 0, 1]])
            K22 = np.array([[4396.869, 0, 1538.86],
                        [0, 4396.869, 989.702],
                        [0, 0, 1]])
            
            K31 = np.array([[5806.559, 0, 1429.219],
                            [0, 5806.559, 993.403],
                            [ 0, 0, 1]])
            K32 = np.array([[5806.559, 0, 1543.51],
                            [ 0, 5806.559, 993.403],
                            [ 0, 0, 1]])
            camera_params = [(K11, K12), (K21, K22), (K31, K32)]

            while(1):
                try:
                    list_kp1, list_kp2 = draw_keypoints_and_match(img1, img2)
                    
                    #_______________________________Calibration_______________________________

                    F = RANSAC_F_matrix([list_kp1, list_kp2])
                    print("F matrix", F)
                    print("=="*20, '\n')
                    K1, K2 = camera_params[number-1]
                    E = calculate_E_matrix(F, K1, K2)
                    print("E matrix", E)
                    print("=="*20, '\n')
                    camera_poses = extract_camerapose(E)
                    best_camera_pose = disambiguate_camerapose(camera_poses, list_kp1)
                    print("Best_Camera_Pose:")
                    print("=="*20)
                    print("Roatation", best_camera_pose[0])
                    print()
                    print("Transaltion", best_camera_pose[1])
                    print("=="*20, '\n')
                    pts1 = np.int32(list_kp1)
                    pts2 = np.int32(list_kp2)

                    #____________________________Rectification________________________________
                    
                    rectified_pts1, rectified_pts2, img1_rectified, img2_rectified = rectification(img1, img2, pts1, pts2, F)
                    break
                except Exception as e:
                    # print("error", e)
                    continue
            
            # Find epilines corresponding to points in right image (second image) and drawing its lines on left image
            
            lines1 = cv2.computeCorrespondEpilines(rectified_pts2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            img5, img6 = drawlines(img1_rectified, img2_rectified, lines1, rectified_pts1, rectified_pts2)

            # Find epilines corresponding to points in left image (first image) and drawing its lines on right image

            lines2 = cv2.computeCorrespondEpilines(rectified_pts1.reshape(-1, 1, 2), 1, F)
            lines2 = lines2.reshape(-1, 3)
            img3, img4 = drawlines(img2_rectified, img1_rectified, lines2, rectified_pts2, rectified_pts1)

            cv2.imwrite("left_image.png", img5)
            cv2.imwrite("right_image.png", img3)
            st.image(img5, caption="left Image", use_column_width=True)
            st.image(img3, caption="right Image", use_column_width=True)

            left_image = Image.open('im0.png')
            right_image = Image.open('im1.png')

            preprocess = transforms.Compose([
            transforms.Resize((384, 384)),  # Resize to match MiDaS input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')

            model.eval()

            left_input_tensor = preprocess(left_image).unsqueeze(0)
            right_input_tensor = preprocess(right_image).unsqueeze(0)

            with torch.no_grad():
                left_depth_prediction = model(left_input_tensor)
                right_depth_prediction = model(right_input_tensor)

            left_depth_map = left_depth_prediction.squeeze().cpu().numpy()
            right_depth_map = right_depth_prediction.squeeze().cpu().numpy()
            
            combined_depth_map = (left_depth_map + right_depth_map) / 2.0

            depth_map_normalized = (combined_depth_map - np.min(combined_depth_map)) / (np.max(combined_depth_map) - np.min(combined_depth_map))

            # Display combined depth map
            st.image(depth_map_normalized, caption="Combined Depth Map (Normalized)", use_column_width=True)
            # Display combined depth map
            fig, ax = plt.subplots()
            im = ax.imshow(combined_depth_map, cmap='inferno')
            cbar = fig.colorbar(im)
            ax.axis('off')
            
            # Display the Matplotlib figure using Streamlit
            st.pyplot(fig)
            
            #st.image(combined_depth_map, caption="Combined Depth Map", use_column_width=True)



if __name__ == "__main__":
    main()
