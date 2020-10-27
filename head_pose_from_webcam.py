#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import argparse
import numpy as np

# helper modules
from drawFace import draw
import reference_world as world

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

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

            height, width, channels = img.shape
            focalLength = args.focal * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

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

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            # print("ThetaX: ", x)
            print("ThetaY: ", y)
            # print("ThetaZ: ", z)
            print('*' * 80)
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
    # path to your video file or camera serial
    main("test.mp4")
