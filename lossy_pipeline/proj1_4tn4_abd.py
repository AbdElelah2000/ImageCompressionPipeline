#Project 1 - upsampling and downsampling - Abd Elelah Arafah, 400197623

import cv2
import numpy as np


# Load the image
img = cv2.imread('test_image.png')


#manual implementation of splitting color channels from an image
def split_channels_manual_fxn(image):
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    return b, g, r

# manual implementation to clip values to ensure the 8-bit representation
def manual_clip_fxn(array, min_value, max_value):
    """Clip the values of an array within a specified range."""
    clipped_values = np.copy(array)
    clipped_values[clipped_values < min_value] = min_value  # clip low values
    clipped_values[clipped_values > max_value] = max_value  # clip higher values
    return clipped_values

def manual_sum_fxn(array):
    """Compute the sum of the elements of an array."""
    sum = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            sum += array[i, j]
    return sum


# Define the conversion matrices for RGB to YUV and YUV to RGB
RGB_to_YUV_conversion_matrix = np.array([[0.299, 0.587, 0.114], [0.5, -0.41869, -0.08131], [-0.16874, -0.33126, 0.5]])
YUV_to_RGB_conversion_matrix = np.array([[1.000, 0.000, 1.13983], [1.000, -0.39465, -0.58060], [1.000, 2.03211, 0.000]])
# Function which converts from RGB to YUV color space
def RGB2YUV_fxn(input_img):
    yuv_img = np.zeros_like(input_img)
    yuv_img[:,:,2] = np.dot(input_img, RGB_to_YUV_conversion_matrix[2])
    yuv_img[:,:,1] = np.dot(input_img, RGB_to_YUV_conversion_matrix[1])
    yuv_img[:,:,0] = np.dot(input_img, RGB_to_YUV_conversion_matrix[0])
    temp_var = yuv_img[:,:,1:] + 128.0
    # 8-bit representation
    yuv_img[:,:,1:] = temp_var.astype(np.uint8)
    return yuv_img


# Define the conversion matrices for YUV to BGR and BGR to YUV
YUV_to_BGR_conversion_matrix = np.array([[1.000, 0.000, 1.403], [1.000, -0.34413, -0.71414], [1.000, 1.773, 0.000]])
# Function to convert from YUV to BGR color space
RGB_to_BGR_conversion_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
# Function which converts from YUV to BGR color space
def YUV2RGB_fxn(input_img):
    # First convert from YUV to RGB.
    yuv_image = input_img.astype(np.float32)
    yuv_image[:,:,1:] = yuv_image[:,:,1:] - 128.0
    rgb_image = np.dot(yuv_image, YUV_to_RGB_conversion_matrix.T)

    # Convert from RGB to BGR color space This is because we need it in the BGR order to show the image correctly with openCV
    bgr_image = np.dot(rgb_image, RGB_to_BGR_conversion_matrix.T)
    # 8-bit representation
    bgr_image = manual_clip_fxn(bgr_image, 0, 255).astype(np.uint8)
    return bgr_image


def downsample_YUV_image_fxn(image):
    # Get the YUV channels
    Y, U, V = split_channels_manual_fxn(image)

    # Downsample the Y channel by a factor of 2
    Y_downsampled_channel = downsample_avg_fxn(Y, 2)

    # Downsample the U and V channels by a factor of 4
    U_downsampled_channel = downsample_avg_fxn(U, 4)
    V_downsampled_channel = downsample_avg_fxn(V, 4)

    return Y_downsampled_channel, U_downsampled_channel, V_downsampled_channel

# implementation of a simple average pooling kernel method
def downsample_avg_fxn(color_channel, downsampling_factor):
    # Define the downsampling kernel
    kernel = np.ones((downsampling_factor, downsampling_factor)) / (downsampling_factor * downsampling_factor)

    # We can Pad the channel
    height, width = color_channel.shape
    pad_H = height + downsampling_factor - (height % downsampling_factor)
    pad_W = width + downsampling_factor - (width % downsampling_factor)
    pad_ch = np.zeros((pad_H, pad_W), dtype=color_channel.dtype)
    pad_ch[:height, :width] = color_channel

    # Apply the downsampling kernel to the padded channel
    downsampled_ch = np.zeros((pad_H // downsampling_factor, pad_W // downsampling_factor), dtype=color_channel.dtype)
    for i in range(0, pad_H, downsampling_factor):
        for j in range(0, pad_W, downsampling_factor):
            kernel_window = pad_ch[i:i+downsampling_factor, j:j+downsampling_factor]
            downsampled_ch[i//downsampling_factor, j//downsampling_factor] = manual_sum_fxn(kernel * kernel_window)

    # Remove the padding
    downsampled_ch = downsampled_ch[:height//downsampling_factor, :width//downsampling_factor]

    return downsampled_ch

# Implementing Bilinear interpolation to upsize an image channel
def bilin_upsample_fxn(image_ch, size):
    # Determine the size of the starting image channel
    H, W = image_ch.shape

    # Determine the size of the output image from the input image size
    new_H, new_W = size

    # Create an array to store the output image channel
    output_ch = np.zeros((new_H, new_W))

    # Computation of the coordinates as image pixels
    x_px_coord = float(W - 1) / (new_W - 1)
    y_px_coord = float(H - 1) / (new_H - 1)

    # Perform bilinear interpolation to compute the output image
    for i in range(new_H):
        for j in range(new_W):
            compute_x = x_px_coord * j
            compute_y = y_px_coord * i
            x_1 = int(compute_x)
            y_1 = int(compute_y)
            x_2 = min(x_1 + 1, W - 1)
            y_2 = min(y_1 + 1, H - 1)
            dx = compute_x - x_1
            dy = compute_y - y_1
            output_ch[i, j] = (1 - dx) * (1 - dy) * image_ch[y_1, x_1] + dx * (1 - dy) * image_ch[y_1, x_2] + (1 - dx) * dy * image_ch[y_2, x_1] + dx * dy * image_ch[y_2, x_2]
    return output_ch


def n_nearest_neighbors_fxn(image_ch, up_sample_factor, size, n_neighbors=4):
    # Determine the size of the starting image channel
    H, W = image_ch.shape
    new_H, new_W = size

    # Create an empty array to hold the upscaled image
    upscale_array = np.zeros((new_H, new_W), dtype=image_ch.dtype)

    # Iterate over each pixel in the upscaled image
    for y in range(new_H):
        for x in range(new_W):
            # Compute pixel
            y_px = int(y / up_sample_factor)
            x_px = int(x / up_sample_factor)

            # Check if the original pixel location is within the bounds of the image
            if y_px >= 0 and y_px < H and x_px >= 0 and x_px < W:
                # Create a subarray of the channel array
                neighborhood = image_ch[max(0, y_px - n_neighbors//2) : min(H, y_px + n_neighbors//2 + 1),
                                       max(0, x_px - n_neighbors//2) : min(W, x_px + n_neighbors//2 + 1)]
                # Calculate the Euclidean distance between the pixel and its neighbors
                euclidean_distances = np.sqrt(np.sum((neighborhood - image_ch[y_px, x_px])**2, axis=None if neighborhood.ndim == 2 else (1, 2)))
                # Choose the pixel which is closest
                nearest_neigh = np.argmin(euclidean_distances)
                nearest_neighbor = neighborhood.flatten()[nearest_neigh]

                # value of the nearest neighbor = the current pixel
                upscale_array[y, x] = nearest_neighbor
    return upscale_array



def compute_PSNR_fxn(image1, image2):
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Error images don't have the same size dimensions")

    # compute mean squared error
    MSE_compute = np.mean((image1 - image2) ** 2)

    # compute PSNR
    if MSE_compute == 0:
        return float('inf')
    else:
        psnr = 20 * np.log10(255 / np.sqrt(MSE_compute))
        return psnr


def compute_MSE_fxn(image1, image2):
    # compare sizes
    assert image1.shape == image2.shape
    # Calculate the mean of the squared differences
    return np.mean( (image1 - image2) ** 2  )


# Convert the image to YUV color space
yuv_img = RGB2YUV_fxn(img)
# Convert the YUV image back to RGB color space
rgb_img = YUV2RGB_fxn(yuv_img)

#downsample the yuv image
Y_downsampled, U_downsampled, V_downsampled = downsample_YUV_image_fxn(yuv_img)

#upsample the yuv image
Y_upsample_img_new = bilin_upsample_fxn(Y_downsampled, img.shape[:2])
U_upsample_img_new = bilin_upsample_fxn(U_downsampled, img.shape[:2])
V_upsample_img_new = bilin_upsample_fxn(V_downsampled, img.shape[:2])
#assemble the image from YUV channels
yuv = np.dstack((Y_upsample_img_new, U_upsample_img_new, V_upsample_img_new))
#convert back to BRG so that we can view it with openCV
rgb_img = YUV2RGB_fxn(yuv)

#Another way of upsampling using n_nearest neighbors
Y_upsample_img_new_ml = n_nearest_neighbors_fxn(Y_downsampled,2, img.shape[:2])
U_upsample_img_new_ml = n_nearest_neighbors_fxn(U_downsampled,4, img.shape[:2])
V_upsample_img_new_ml = n_nearest_neighbors_fxn(V_downsampled,4, img.shape[:2])
#assemble the image from YUV channels
yuv_ml = np.dstack((Y_upsample_img_new_ml, U_upsample_img_new_ml, V_upsample_img_new_ml))
#convert back to BRG so that we can view it with openCV
rgb_img_ml = YUV2RGB_fxn(yuv_ml)

#Compute the pixel sizes for each image
height, width = img.shape[:2]
height1, width1 = rgb_img.shape[:2]
height2, width2 = rgb_img_ml.shape[:2]

#print sizes and errors
print(f"Input Image Pixel Size: {width}x{height}")
print(f"Bilinear_interpolation Output Image Pixel Size: {width1}x{height1}")
print(f"n_near_neighbors Output Image Pixel Size: {width2}x{height2}")
print(f"Bilinear_interpolation PSNR Error: {compute_PSNR_fxn(img, rgb_img)}")
print(f"n_near_neighbors PSNR Error: {compute_PSNR_fxn(img, rgb_img_ml)}")
print(f"Bilinear_interpolation MSE Error: {compute_MSE_fxn(img, rgb_img)}")
print(f"n_near_neighbors MSE Error: {compute_MSE_fxn(img, rgb_img_ml)}")

# Display the original and the converted images
cv2.imshow('Original Image', img)
cv2.imshow('Bilinear_interpolation', rgb_img)
cv2.imwrite('BilinearInterpolation.jpg', rgb_img)
cv2.imshow('n_near_neighbors', rgb_img_ml)
cv2.imwrite('nNearNeighbors.jpg', rgb_img_ml)
cv2.imshow('Input image Y channel', yuv_img[:,:,0])
cv2.imshow('Input image U channel', yuv_img[:,:,1])
cv2.imshow('Input image V channel', yuv_img[:,:,2])
cv2.imwrite('InputYchannel.jpg', yuv_img[:,:,0])
cv2.imwrite('InputUchannel.jpg', yuv_img[:,:,1])
cv2.imwrite('InputVchannel.jpg', yuv_img[:,:,2])
cv2.imshow('Y_downsampled', Y_downsampled)
cv2.imshow('U_downsampled', U_downsampled)
cv2.imshow('V_downsampled', V_downsampled)
cv2.imwrite('YDownsampled.jpg', Y_downsampled)
cv2.imwrite('UDownsampled.jpg', U_downsampled)
cv2.imwrite('VDownsampled.jpg', V_downsampled)
cv2.imshow('Yupsampled', Y_upsample_img_new)
cv2.imshow('Uupsampled', U_upsample_img_new)
cv2.imshow('Vupsampled', V_upsample_img_new)
cv2.imwrite('Yupsampled.jpg', Y_upsample_img_new)
cv2.imwrite('Uupsampled.jpg', U_upsample_img_new)
cv2.imwrite('Vupsampled.jpg', V_upsample_img_new)
cv2.imshow('YupsampledML', Y_upsample_img_new_ml)
cv2.imshow('UupsampledML', U_upsample_img_new_ml)
cv2.imshow('VupsampledML', V_upsample_img_new_ml)
cv2.imwrite('YupsampledML.jpg', Y_upsample_img_new_ml)
cv2.imwrite('UupsampledML.jpg', U_upsample_img_new_ml)
cv2.imwrite('VupsampledML.jpg', V_upsample_img_new_ml)


#terminate script
cv2.waitKey(0)
cv2.destroyAllWindows()
