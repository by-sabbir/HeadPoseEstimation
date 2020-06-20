#!/usr/bin/env python3
import os
import cv2
import dlib
import argparse
import numpy as np

# helper modules
from drawFace import draw
import reference_world as world

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--focal",
                    type=float,
                    help="Callibrated Focal Length of the camera")

args = parser.parse_args()

def main(source=0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(source)

    while True:
        GAZE = "Face Not Found"
        ret, img = cap.read()
        if not ret:
            print(f"[ERROR - System]Cannot read from source: {source}")
            break

        faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        face3Dmodel = world.ref3DModel()

        for face in faces:
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

            draw(img, shape)

            refImgPts = world.ref2dImagePoints(shape)

            rows, cols, ch = img.shape
            focalLength = args.focal * cols
            cameraMatrix = world.CameraMatrix(focalLength, (rows / 2, cols / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(img, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating angle
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            if angles[1] < -15:
                GAZE = "Looking: Left"
            elif angles[1] > 15:
                GAZE = "Looking: Right"
            else:
                GAZE = "Forward"

        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Head Pose", img)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("head_pose_poc.webm")
