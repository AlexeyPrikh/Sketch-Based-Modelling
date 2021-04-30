from pathlib import Path

ROOT_PATH = Path("g2pe/")
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

KPS_SPIN_MAP={
    'Head' : 'Head (H36M)', 
    'Neck' : 'OP Neck',
    'Right Shoulder' : 'OP RShoulder',
    'Right Arm' : 'OP RElbow',
    'Right Hand' : 'OP RWrist',
    'Left Shoulder' : 'OP LShoulder',
    'Left Arm' : 'OP LElbow',
    'Left Hand' : 'OP LWrist',
    'Spine' : 'Spine (H36M)',
    'Hips' : 'OP MidHip',
    'Right Upper Leg' : 'OP RHip',
    'Right Leg' : 'OP RKnee',
    'Right Foot' : 'OP RAnkle',
    'Left Upper Leg' : 'OP LHip',
    'Left Leg' : 'OP LKnee',
    'Left Foot' : 'OP LAnkle'
}

MODEL_HMR_PATH = 'spin/data/spin_model.pt'

LOAD_DIR = 'sketches/'
SAVE_DIR = 'results/'