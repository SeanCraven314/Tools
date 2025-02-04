# /// script
# requires-python = ">=3.12"
# dependencies = ["opencv-python", "ultralytics"]
# ///

import os
import time
import sqlite3
import sys
import cv2
from typing import Generator, Literal
import ultralytics
from PIL import Image
from dataclasses import dataclass
from ultralytics.engine.model import Results

CREATETABLE = """CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    yolo_model_name TEXT NOT NULL,
    texture TEXT NOT NULL,
    dataset TEXT NOT NULL,
    transforms TEXT NOT NULL,
    target_classes TEXT NOT NULL
);
"""
CREATETABLEDETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    prediction_id INTEGER NOT NULL,
    class INTEGER NOT NULL,
    prob FLOAT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
"""
CREATETABLEPREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    image_name TEXT NOT NULL,
    camera_height FLOAT NOT NULL,
    camera_distance FLOAT NOT NULL,
    camera_angle FLOAT NOT NULL,
    run_id INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
"""

TARGET_CLASS = 0


class DeadError(Exception):
    pass


class Exit(Exception):
    pass


def green(text: str) -> str:
    green_color = "\033[92m"
    reset_color = "\033[0m"
    return f"{green_color}{text}{reset_color}"


@dataclass
class Experiment:
    experiment_id: int
    camera_height: float
    camera_angle: int
    camera_distance: float

    @property
    def image_name(self) -> str:
        experiment_summary = f"Height:{self.camera_height}\nDistance:{self.camera_distance}\nAngle:{self.camera_angle}"
        return experiment_summary

    def summarise(
        self,
        hot_field: Literal[
            "camera_height",
            "camera_angle",
            "camera_distance",
        ],
    ) -> str:
        out = [f"Experiment: {self.experiment_id}"]
        for name, field in {
            "Height": "camera_height",
            "Distance": "camera_distance",
            "Angle": "camera_angle",
        }.items():
            str_ = f"{name}:{getattr(self, field)}"
            if field == hot_field:
                str_ = green(str_)
            out.append(str_)
        experiment_summary = "\n".join(out)
        return experiment_summary

    def _capture(self, camera: cv2.VideoCapture) -> Image.Image:
        win_name = "experiment"
        while True:
            alive, img = camera.read()
            if not alive:
                raise DeadError("Camera is very dead, try again mr bean.")
            cv2.imshow(win_name, img)
            in_ = cv2.waitKey(1)
            if in_ == ord("q"):
                raise Exit()
            if in_ == ord(" "):
                cv2.destroyWindow(win_name)
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                return pil

    def run(
        self,
        camera: cv2.VideoCapture,
        yolo: ultralytics.YOLO,
        conn: sqlite3.Connection,
    ) -> None:
        pil = self._capture(camera)
        [predictions] = yolo(pil)
        insert_detections(
            conn,
            predictions,
            self,
            {TARGET_CLASS},
            self.experiment_id,
        )


@dataclass
class Experiments:
    heights_cm: list[float]
    angles: list[int]
    distances_cm: list[float]
    id: int | None

    def generate_settings(
        self, camera: cv2.VideoCapture
    ) -> Generator[Experiment, None, None]:
        if self.id is None:
            raise ValueError(
                "Commit the settings to database before you do recoriding mr bean."
            )
        heights_count = len(self.heights_cm)
        distances_count = len(self.distances_cm)
        for i, height in enumerate(self.heights_cm):
            for j, distance in enumerate(self.distances_cm):
                for k, angle in enumerate(self.angles):
                    exp = Experiment(
                        camera_height=height,
                        camera_angle=angle,
                        experiment_id=self.id,
                        camera_distance=distance,
                    )

                    os.system("cls" if os.name == "nt" else "clear")

                    print(exp.summarise("camera_angle"))
                    print("Press space to capture")
                    yield exp

                next_distance = j + 1
                if next_distance == distances_count:
                    print("Final Distance")
                else:
                    os.system("cls" if os.name == "nt" else "clear")
                    print(
                        green(f"Change Distance to {self.distances_cm[next_distance]}")
                    )
                    print("Once alligned press space")
                    exp._capture(camera)

            next_height = i + 1
            if next_height == heights_count:
                print("End")
            else:
                os.system("cls" if os.name == "nt" else "clear")
                print(green(f"Change Height to {self.heights_cm[next_height]}"))
                print("Once alligned press space")
                exp._capture(camera)

    @property
    def name(self) -> str:
        return f"Heights:{min(self.heights_cm)}-{max(self.heights_cm)}\nAngles:{min(self.heights_cm)}-{max(self.heights_cm)}"

    def store_experiment(
        self,
        conn: sqlite3.Connection,
        yolo_model_name: str,
        texture: str,
        target_classes: list[str],
    ) -> None:
        self.name
        id = conn.execute(
            "INSERT INTO runs (yolo_model_name, texture, dataset, transforms, target_classes) VALUES (?, ?, ?, ?, ?) RETURNING id;",
            (
                str(yolo_model_name),
                str(texture),
                self.name,
                "",
                ",".join(target_classes),
            ),
        ).fetchone()
        conn.commit()
        self.id = int(id[0])


def insert_detections(
    conn: sqlite3.Connection,
    predictions: Results,
    image_metadata: Experiment,
    target_cls: set[int],
    run_id: int,
) -> None:
    boxes = predictions.boxes

    (prediction_id,) = conn.execute(
        "INSERT INTO predictions (run_id, image_name, camera_height, camera_distance, camera_angle) VALUES (?, ?, ?, ?, ?) RETURNING id;",
        (
            run_id,
            image_metadata.image_name,
            image_metadata.camera_height,
            image_metadata.camera_distance,
            image_metadata.camera_angle,
        ),
    ).fetchone()

    if boxes is None:
        return

    classes = boxes.cls.tolist()
    probs = boxes.conf.tolist()
    for cls, prob in zip(classes, probs):
        if int(cls) not in target_cls:
            continue

        conn.execute(
            "INSERT INTO detections (class, prob, prediction_id) VALUES (?, ?, ?);",
            (cls, prob, prediction_id),
        )

    conn.commit()


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while (
        len(non_working_ports) < 6
    ):  # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(
                    "Port %s is working and reads images (%s x %s)" % (dev_port, h, w)
                )
                working_ports.append(dev_port)
            else:
                print(
                    "Port %s for camera ( %s x %s) is present but does not reads."
                    % (dev_port, h, w)
                )
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


def see_stuff() -> None:
    for i in range(0, 10):
        camera = cv2.VideoCapture(i)
        while True:
            alive, img = camera.read()
            if not alive:
                continue
            cv2.imshow("Check", img)
            in_ = cv2.waitKey()
            if in_ == ord("q"):
                break
        cv2.destroyWindow("Check")
        camera.release()


def experiment() -> None:
    os.environ["YOLO_VERBOSE"] = "False"
    yolo = ultralytics.YOLO()
    webcam_port = 0
    camera = cv2.VideoCapture(webcam_port)
    conn = sqlite3.connect("results.db")
    conn.execute(CREATETABLE)
    conn.execute(CREATETABLEDETECTIONS)
    conn.execute(CREATETABLEPREDICTIONS)
    try:
        exps = Experiments([1.1], [0, 45, 90], [1.1, 1.2], id=None)
        exps.store_experiment(conn, yolo.model_name, "test", ["car"])
        for exp in exps.generate_settings(camera):
            exp.run(camera, yolo, conn)
    except Exit:
        print("Exiting")
        sys.exit(1)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    experiment()
