from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)


def gen_frames():
    face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
    hand_cascade = cv2.CascadeClassifier('static/closed_frontal_palm.xml')

    while True:
        success, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        hands = hand_cascade.detectMultiScale(gray, 1.1, 4)

        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font for text labels
        font_scale = 0.5  # Adjust font size as needed
        font_thickness = 2  # Adjust font thickness as needed

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x + w + 10, y + h - 5), font, font_scale, (255, 0, 0),
                        font_thickness)  # Place text next to the rectangle

        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Hand', (x + w + 10, y + h - 5), font, font_scale, (0, 255, 0),
                        font_thickness)  # Place text next to the rectangle

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Boundary for streaming

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
