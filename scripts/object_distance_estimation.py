import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch

# Function to display an image with matplotlib
def display_image(image, title="Detection Results"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# Function to run inference and visualize results
def test_model(model_path, depth_pro_model_path, test_images_path, results_path):
    # Load the YOLO model
    print("Loading the model...")
    model = YOLO(model_path)

    # Load depth model and preprocessing transform
    import depth_pro
    config = depth_pro.depth_pro.DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=depth_pro_model_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    depth_model, transform = depth_pro.create_model_and_transforms(config=config)
    depth_model.eval()

    # Create results directory if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Loop through test images
    for image_name in os.listdir(test_images_path):
        image_path = os.path.join(test_images_path, image_name)

        # Ensure the file is an image
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"Processing image: {image_name}")

        # Read the image
        image = cv2.imread(image_path)

        # Run inference on the image
        results = model.predict(source=image, save=False, conf=0.25)

        # Get results and visualize detections
        annotated_image = results[0].plot()  # Annotated image with detections

        # Save the result to the results folder
        results_images_path = os.path.join(results_path, "images")
        if not os.path.exists(results_images_path):
            os.makedirs(results_images_path)
        save_annotation_path = os.path.join(results_images_path, image_name)
        cv2.imwrite(save_annotation_path, annotated_image)

        # Display the result
        #display_image(annotated_image, title=f"Results for {image_name}")

        # Get bounding boxes
        image_depth_input = image.copy()
        object_boxes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() # Get bounding boxes
            classes = result.boxes.cls.cpu().numpy() # Get class labels
        for box, cls in zip(boxes, classes):
            #if result.names[int(cls)] == 'person': # Filter for person class
            x1, y1, x2, y2 = map(int, box[:4])
            object_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image_depth_input, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw rectangle

        # Save the result to the results folder
        results_images_path = os.path.join(results_path, "images_depth_input")
        if not os.path.exists(results_images_path):
            os.makedirs(results_images_path)
        save_path = os.path.join(results_images_path, image_name)
        cv2.imwrite(save_path, image_depth_input)

        # Prepare image for depth estimation
        image_depth, _, f_px = depth_pro.load_rgb(save_path)
        depth_input = transform(image_depth)

        # Perform depth inference
        prediction = depth_model.infer(depth_input, f_px=f_px)
        depth = prediction["depth"] # Depth in meters
        # Convert depth to numpy array
        depth_np = depth.squeeze().cpu().numpy()

        # Calculate depth for detected persons and display on image
        img_height, img_width = image_depth.shape[:2]
        for x1, y1, x2, y2 in object_boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Extract depth value at the center of the bounding box
            depth_value = depth_np[center_y, center_x]
            text = f'Depth: {depth_value:.2f}m'
            # Define font properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            # Set text position
            text_x = x1
            text_y = y1 + 15
            # Create a rectangle for text background
            rect_x1 = text_x - 10
            rect_y1 = text_y - text_size[1] - 5
            rect_x2 = text_x + text_size[0] + 5
            rect_y2 = text_y + 5
            
            # Ensure rectangle stays within image bounds
            if rect_x1 < 0:
                shift = -rect_x1
                rect_x1 = 0
                text_x += shift
                rect_x2 += shift
            if rect_y1 < 0:
                shift = -rect_y1
                rect_y1 = 0
                text_y += shift
                rect_y2 += shift
            if rect_x2 > img_width:
                shift = img_width - rect_x2
                rect_x2 = img_width
                text_x += shift
                rect_x1 += shift
            if rect_y2 > img_height:
                shift = img_height - rect_y2
                rect_y2 = img_height
                text_y += shift
                rect_y1 += shift
            
            # Draw the background rectangle and add text
            cv2.rectangle(annotated_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            cv2.putText(annotated_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # Save the result to the results folder
        results_images_path = os.path.join(results_path, "images_distance")
        if not os.path.exists(results_images_path):
            os.makedirs(results_images_path)
        save_path = os.path.join(results_images_path, image_name)
        cv2.imwrite(save_path, annotated_image)

        # Display the result
        #display_image(annotated_image, title=f"Results for {image_name}")

if __name__ == "__main__":
    # Paths
    MODEL_PATH = "models/yolo/yolo11n.pt"  # Replace with your model path
    DepthPRO_MODEL_PATH = "models/depth_pro/depth_pro.pt"  # Replace with your model path
    TEST_IMAGES_PATH = "datasets/sew-dataset-2025/images/test/05_rgb/industrial"      # Replace with the path to your test images
    TEST_VIDEOS_PATH = ""      # Replace with the path to your test images
    RESULTS_PATH = "results/model_test"        # Replace with the path to save results

    # Run the test
    test_model(MODEL_PATH, DepthPRO_MODEL_PATH, TEST_IMAGES_PATH, RESULTS_PATH)
