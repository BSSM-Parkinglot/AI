import cv2
import pytesseract
import os

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'

def recognize_license_plate(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image to improve OCR accuracy
    # (You may need to fine-tune these preprocessing steps for your specific use case)
    # For example, you can apply thresholding, edge detection, and morphological operations.

    # Perform OCR on the preprocessed image
    license_plate_text = pytesseract.image_to_string(gray, lang='kor', config='--oem 3 --psm 7')

    # Print the recognized license plate text
    print("Recognized License Plate:", license_plate_text.strip())

    return license_plate_text.strip()

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Detect and recognize the license plate
        plate_text = recognize_license_plate(frame)

        # Display the frame with the license plate text
        cv2.putText(frame, plate_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("License Plate Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
