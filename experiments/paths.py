import os

DEVICE_IDS = {
    '3':'0',
    '2':'1',
    '0':'2',
    '1':'3',
    '7':'4',
    '6':'5',
    '4':'6',
    '5':'7',
}

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = DEVICE_IDS[os.environ["CUDA_VISIBLE_DEVICES"]]
