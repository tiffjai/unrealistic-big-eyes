import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def enlarge_eyes(image):
    # Detect face landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)

    if not face_landmarks_list:
        raise ValueError("No faces detected in the image")

    for face_landmarks in face_landmarks_list:
        # Function to enlarge eye
        def enlarge_eye(eye_points):
            try:
                # Calculate eye center and ensure it is a tuple of floats
                eye_center = tuple(np.mean(eye_points, axis=0).astype(float))
                logger.debug(f"Eye center: {eye_center}")

                # Calculate the scaling factor
                scale_factor = 1.5  # Adjust this for more or less enlargement

                # Create a scaling matrix using the eye center
                scaling_matrix = cv2.getRotationMatrix2D(
                    eye_center, 0, scale_factor)
                logger.debug(f"Scaling matrix: {scaling_matrix}")

                # Apply the scaling to the eye region
                rows, cols = image.shape[:2]
                enlarged_eye = cv2.warpAffine(
                    image, scaling_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

                # Create a mask for seamless blending
                mask = np.zeros((rows, cols), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.array(eye_points), 255)
                mask = cv2.dilate(mask, None, iterations=2)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)

                # Blend the enlarged eye with the original image
                image_with_enlarged_eye = image.copy()
                np.copyto(image_with_enlarged_eye, enlarged_eye,
                          where=mask[:, :, None].astype(bool))

                return cv2.seamlessClone(image_with_enlarged_eye, image, mask, tuple(map(int, eye_center)), cv2.NORMAL_CLONE)
            except Exception as e:
                logger.error(f"Error in enlarge_eye: {str(e)}")
                logger.error(f"Eye points: {eye_points}")
                raise

        # Apply enlargement to both eyes
        image = enlarge_eye(face_landmarks['left_eye'])
        image = enlarge_eye(face_landmarks['right_eye'])

    return image


@app.route('/')
def home():
    return "Unrealistic Eye Enlargement API is running!"


@app.route('/enlarge-eyes', methods=['POST'])
def process_image():
    try:
        # Get image data from request
        image_data = request.json['image']

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        logger.debug(f"Image shape: {image.shape}")

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        result = enlarge_eyes(rgb_image)

        # Convert back to BGR for OpenCV encoding
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Encode the result image to send back
        _, buffer = cv2.imencode('.jpg', result_bgr)
        response = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': f'data:image/jpeg;base64,{response}'})
    except Exception as e:
        logger.exception(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
