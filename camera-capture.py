# /// script
# requires-python = ">=3.12"
# dependencies = ["opencv-python", "ultralytics"]
# ///

import os
import time
from pathlib import Path
import sqlite3
import sys
import cv2
from typing import Generator, Literal, Protocol
import ultralytics
from PIL import Image
from dataclasses import dataclass
from ultralytics.engine.model import Results

# SET EXPERIMENT SETTINGS HERE!!!
DATABASE = "results.db"
TARGET_CLASS = "car"
ANGLES = list(range(0, 360, 45))
HEIGHTS = [0.0, 16.0, 38.0]
DISTANCES = [33.0, 50, 84, 96, 130]

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
    class TEXT NOT NULL,
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


class DeadError(Exception):
    pass


class Back(Exception):
    pass


class Exit(Exception):
    pass


def green(text: str) -> str:
    green_color = "\033[92m"
    reset_color = "\033[0m"
    return f"{green_color}{text}{reset_color}"


def camera_capture(camera: cv2.VideoCapture) -> Image.Image:
    """Camera IO, and input management.

    Show camera and wait for human input, to proccede.
    Note, this is where I would append further human interaction.
    Exceptions are used as control signals as they propagate all the way up to the
    event loop. This is not great.
    """
    win_name = "experiment"
    while True:
        alive, img = camera.read()
        if not alive:
            raise DeadError("Camera is very dead, try again mr bean.")
        cv2.imshow(win_name, img)
        in_ = cv2.waitKey(1)
        if in_ == ord("q"):
            raise Exit()
        elif in_ == ord("h"):
            cv2.destroyWindow(win_name)
            raise Back()
        elif in_ == ord(" "):
            cv2.destroyWindow(win_name)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return pil


@dataclass
class Experiment:
    experiment_id: int
    camera_height: float
    camera_angle: int
    camera_distance: float
    data_root: Path
    frame_id: int

    @property
    def image_name(self) -> str:
        return f"{self.frame_id:05d}.png"

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
        return camera_capture(camera)

    def run(
        self,
        camera: cv2.VideoCapture,
        yolo: ultralytics.YOLO,
        conn: sqlite3.Connection,
    ) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        print("Experiment")
        print(self.summarise("camera_angle"))
        print("Press space to capture, q to exit and h to go back")

        pil = self._capture(camera)
        pil.save(self.data_root / f"{self.frame_id}.png")

        [predictions] = yolo(pil)
        vk = {v: k for k, v in yolo.names.items()}[TARGET_CLASS]

        insert_detections(
            conn,
            predictions,
            self,
            {vk},
            self.experiment_id,
        )


class Runnable(Protocol):
    def run(
        self,
        camera: cv2.VideoCapture,
        yolo: ultralytics.YOLO,
        conn: sqlite3.Connection,
    ) -> None: ...


@dataclass
class Aiming:
    msg: str

    def run(
        self,
        camera: cv2.VideoCapture,
        yolo: ultralytics.YOLO,
        conn: sqlite3.Connection,
    ) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        print(self.msg)
        camera_capture(camera)


@dataclass
class Experiments:
    data_root: Path
    heights_cm: list[float]
    angles: list[int]
    distances_cm: list[float]
    id: int | None

    def generate_settings(self) -> Generator[Runnable, None, None]:
        if self.id is None:
            raise ValueError(
                "Commit the settings to database before you do recoriding mr bean."
            )
        heights_count = len(self.heights_cm)
        distances_count = len(self.distances_cm)
        ii = 0
        data_root = self.data_root.joinpath(str(self.id))
        data_root.mkdir()

        for i, height in enumerate(self.heights_cm):
            for j, distance in enumerate(self.distances_cm):
                for _, angle in enumerate(self.angles):
                    exp = Experiment(
                        data_root=data_root,
                        camera_height=height,
                        camera_angle=angle,
                        experiment_id=self.id,
                        camera_distance=distance,
                        frame_id=ii,
                    )

                    yield exp
                    ii += 1

                next_distance = j + 1
                if next_distance != distances_count:
                    msg = green(
                        f"Change Distance to {self.distances_cm[next_distance]}"
                    )
                    msg += "\nOnce alligned press space"

                    yield Aiming(msg)

            next_height = i + 1
            if next_height != heights_count:
                msg = green(f"Change Height to {self.heights_cm[next_height]}")
                msg += "\nOnce alligned press space"
                yield Aiming(msg)

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


def _insert_boxes(
    conn: sqlite3.Connection,
    predictions: Results,
    prediction_id: int,
    target_cls: set[int],
) -> None:
    boxes = predictions.boxes
    if boxes is None:
        return

    classes = boxes.cls.tolist()
    predictions.names
    probs = boxes.conf.tolist()
    for cls, prob in zip(classes, probs):
        if int(cls) not in target_cls:
            continue

        conn.execute(
            "INSERT INTO detections (class, prob, prediction_id) VALUES (?, ?, ?);",
            (predictions.names[cls], prob, prediction_id),
        )

    conn.commit()


def insert_detections(
    conn: sqlite3.Connection,
    predictions: Results,
    image_metadata: Experiment,
    target_cls: set[int],
    run_id: int,
) -> None:
    out = conn.execute(
        "SELECT id FROM predictions where image_name = ? and run_id = ?;",
        (image_metadata.image_name, run_id),
    ).fetchone()
    # If row doesn't exist make it.
    if out is None:
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
    # If row does exist delete all the detections.
    else:
        prediction_id = out[0]
        conn.execute(
            "DELETE FROM detections WHERE prediction_id = ?;", (int(prediction_id),)
        )
        conn.commit()
    _insert_boxes(
        conn,
        predictions,
        prediction_id,
        target_cls,
    )


def experiment() -> None:
    """Run experiment interaction loop"""
    data_root = os.getenv("DATA_ROOT", "/mnt/scratch/data/small_scale")
    data_root = Path(data_root)
    data_root.mkdir(exist_ok=True)

    os.environ["YOLO_VERBOSE"] = "False"
    yolo = ultralytics.YOLO()

    webcam_port = 0
    camera = cv2.VideoCapture(webcam_port)

    conn = sqlite3.connect(DATABASE)
    conn.execute(CREATETABLE)
    conn.execute(CREATETABLEDETECTIONS)
    conn.execute(CREATETABLEPREDICTIONS)
    try:
        exps = Experiments(
            data_root,
            angles=ANGLES,
            heights_cm=HEIGHTS,
            distances_cm=DISTANCES,
            id=None,
        )
        exps.store_experiment(
            conn, yolo.model_name, "green_camo_darker_plus_plus", [TARGET_CLASS]
        )

        ## Run experiments in a loop
        ## Case where user wants to re run experiments
        ## is handeled by the back error
        ## Back is raised in camera_capture
        exps = list(exps.generate_settings())
        i = 0
        while i < len(exps):
            exp = exps[i]
            try:
                exp.run(camera, yolo, conn)
            except Back:
                i = max(i - 1, 0)
                continue
            i += 1
    except Exit:
        print("Exiting")
        sys.exit(1)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def list_ports():
    for i in range(10):
        camera = cv2.VideoCapture(i)
        alive, img = camera.read()
        if not alive:
            continue
        else:
            print(camera)


if __name__ == "__main__":
    experiment()
