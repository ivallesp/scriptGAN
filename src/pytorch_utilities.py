import os


def use_devices(devices):
    devices = [devices] if type(devices) is not list else devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in devices)