detection = [
  {"box_2d": [87, 360, 351, 448], "label": "person", "distance": "4.5m"},
  {"box_2d": [0, 465, 496, 640], "label": "person", "distance": "0.8m"},
  {"box_2d": [0, 112, 248, 250], "label": "poster", "distance": "1.2m"},
  {"box_2d": [0, 8, 495, 283], "label": "panel", "distance": "1.2m"},
  {"box_2d": [151, 305, 201, 353], "label": "monitor", "distance": "5.5m"},
  {"box_2d": [190, 456, 235, 493], "label": "robot arm", "distance": "7.0m"},
  {"box_2d": [222, 247, 345, 335], "label": "workbench", "distance": "5.0m"},
  {"box_2d": [454, 26, 478, 58], "label": "wheel", "distance": "1.5m"},
  {"box_2d": [465, 147, 498, 175], "label": "wheel", "distance": "2.0m"}
]

import cv2

def draw_detections(image_path, detections, output_path="output.jpg"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    img_height, img_width = image.shape[:2]

    for det in detections:
        y1, x1, y2, x2 = det["box_2d"]
        label = det["label"]
        distance = det.get("distance", "N/A")
        
        # Combine label and distance
        text = f"{label} - {distance}"

        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),   # Green
            thickness=2
        )

        # Label background
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1
        )

        # Determine text position to keep it within image boundaries
        # Check if text fits above the box
        if y1 - text_height - 4 >= 0:
            # Place text above the box
            text_y = y1 - 4
            bg_y1 = y1 - text_height - 4
            bg_y2 = y1
        else:
            # Place text inside the box at the top
            text_y = y1 + text_height + 4
            bg_y1 = y1
            bg_y2 = y1 + text_height + 4
        
        # Ensure text doesn't go beyond left boundary
        text_x = max(0, x1)
        
        # Ensure background doesn't go beyond right boundary
        bg_x2 = min(img_width, text_x + text_width)
        
        cv2.rectangle(
            image,
            (text_x, bg_y1),
            (bg_x2, bg_y2),
            (0, 255, 0),
            -1
        )

        # Label text
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # Save result
    cv2.imwrite(output_path, image)
    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    draw_detections("C:\\Users\\az04252\\Desktop\\Hussam ATOUI\\GetKerrigan\\amr-segmentation\\datasets\\sew-dataset-2025\\images\\test\\05_rgb\\industrial\\013392.jpg", detection)