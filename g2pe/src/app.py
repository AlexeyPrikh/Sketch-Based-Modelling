import urllib
from pathlib import Path
from textwrap import dedent

import cv2
import PIL.Image as Image
import requests
import streamlit as st

from utils import get_final_preds, get_input, put_kps

st.set_page_config(layout="wide")

ROOT_PATH = Path(".")
ASSETS_PATH = ROOT_PATH / "assets"
DESIRED_SIZE = 512
DEFAULT_IMG_FILEPATH = ASSETS_PATH / "test.png"
DEFAULT_IMG_URLPATH = (
    "https://drawpaintacademy.com/wp-content/uploads/2018/05/Michelangelo.jpg"
)
MODEL_FILEPATH = ASSETS_PATH / "model_best.onnx"
MODEL_URLPATH = (
    f"https://github.com/kbrodt/gesture-2d-pose-estimation/releases/download/v0.1/{MODEL_FILEPATH.name}"
)


def readme():
    st.sidebar.title("2D Pose Estimation for sketches with human-like characters")

    st.sidebar.subheader("Overview")
    st.sidebar.write(dedent("""\
        Simple 2D Pose Estimation model for gesture drawings and sketches with
        human-like characters. The model trained on images with gesture drawings
        with corresponding skeletons consisting of 16 two-dimensional labels.
    """))

    st.sidebar.subheader("Usage")
    st.sidebar.write(dedent("""\
        Upload an image or put an URL and immediately get the results. You also
        can copy the key-points in JSON format, where each key is a joint name,
        and the corresponding value contains XY coordinates and the confidence.
    """))

    st.sidebar.subheader("Limitataions")
    st.sidebar.write(dedent("""\
        The model supports only one character per image.
        The character must be full length and located in the middle.
    """))

    st.sidebar.subheader("Author")
    st.sidebar.write("Kirill Brodt")


@st.cache
def process_image(img_raw):
    if not MODEL_FILEPATH.exists():
        urllib.request.urlretrieve(MODEL_URLPATH, MODEL_FILEPATH)

    model = cv2.dnn.readNetFromONNX(str(MODEL_FILEPATH))

    pose_input, img, center, scale = get_input(img_raw)
    model.setInput(pose_input[None])
    predicted_heatmap = model.forward()
    predicted_keypoints, confidence = get_final_preds(
        predicted_heatmap, center[None], scale[None], post_process=True
    )

    predicted_keypoints, confidence, predicted_heatmap = (
        predicted_keypoints[0],
        confidence[0],
        predicted_heatmap[0],
    )

    img = Image.fromarray(img[..., ::-1])
    original_img_size = img.size
    ratio = float(DESIRED_SIZE) / max(original_img_size)
    resized_img_size = tuple([int(x * ratio) for x in original_img_size])
    img = img.resize(resized_img_size, Image.ANTIALIAS)
    original_img_size = max(original_img_size)

    predicted_keypoints *= (DESIRED_SIZE - 1) / (original_img_size - 1)

    predicted_heatmap = predicted_heatmap.sum(0)
    predicted_heatmap_min = predicted_heatmap.min()
    predicted_heatmap_max = predicted_heatmap.max()
    predicted_heatmap = (predicted_heatmap - predicted_heatmap_min) / (
        predicted_heatmap_max - predicted_heatmap_min
    )
    predicted_heatmap = Image.fromarray((predicted_heatmap * 255).astype("uint8"))
    predicted_heatmap = predicted_heatmap.resize(resized_img_size, Image.ANTIALIAS)

    return img, predicted_keypoints, confidence, predicted_heatmap


def main():
    readme()

    inp = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if inp is None:
        st.warning("No file selected. Using default.")
        with open(DEFAULT_IMG_FILEPATH, "br") as inp:
            img_raw = inp.read()
    else:
        img_raw = inp.read()

    if st.checkbox("or put URL"):
        url = st.text_input("The URL link", value=DEFAULT_IMG_URLPATH)
        if url != "":
            img_raw = requests.get(url).content

    img, predicted_keypoints, confidence, predicted_heatmap = process_image(img_raw)

    thresh = st.slider("Confidence threshold", 0.0, 1.0, value=0.0, step=0.05)
    img_with_keyponts, kps_json = put_kps(
        img, predicted_keypoints, confidence, thresh=thresh
    )

    st.image(
        [
            img,
            img_with_keyponts,
            predicted_heatmap,
        ],
        caption=["original", "annotated", "heatmap"],
    )

    if st.checkbox("Show keypoints in json format {'JOINT_NAME': [x, y, confidence]}"):
        st.write(kps_json)


if __name__ == "__main__":
    main()
