"""
This script will produce shot annotation on a tennis video.
It will output a csv file containing frame id and shot name by pressing your key board keys
RIGHT_ARROW_KEY to mark a shot as FOREHAND
LEFT_ARROW_KEY to mark a shot as BACKHAND
UP_ARROW_KEY to mark a shot as SERVE
We advise you to hit the key when the player hits the ball.
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2

# OpenCV key codes differ across platforms and whether waitKey or waitKeyEx is used.
LEFT_ARROW_KEYS = {81, 2424832}
UP_ARROW_KEYS = {82, 2490368}
RIGHT_ARROW_KEYS = {83, 2555904}
EXIT_KEYS = {27, ord("q"), ord("Q")}


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate a video and write a csv file containing tennis shots"
    )
    parser.add_argument("video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    df = pd.DataFrame(columns=["Shot", "FrameId"])

    FRAME_ID = 0

    your_list = []

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)
        k = cv2.waitKeyEx(30)

        if k in RIGHT_ARROW_KEYS:  # forehand
            your_list.append({"Shot": "forehand", "FrameId": FRAME_ID})
            df = pd.DataFrame.from_records(your_list)
            print("Add forehand")
        elif k in LEFT_ARROW_KEYS:  # backhand
            your_list.append({"Shot": "backhand", "FrameId": FRAME_ID})
            df = pd.DataFrame.from_records(your_list)
            print("Add backhand")
        elif k in UP_ARROW_KEYS:  # serve
            your_list.append({"Shot": "serve", "FrameId": FRAME_ID})
            df = pd.DataFrame.from_records(your_list)
            print("Add serve")

        # Press Q or ESC on keyboard to exit
        if k in EXIT_KEYS:
            break

        FRAME_ID += 1

    out_file = f"annotation_{Path(args.video).stem}.csv"
    df.to_csv(out_file, index=False)
    print(f"Annotation file was written to {out_file}")
