import os
import shutil

from tensorboardX import SummaryWriter


def use_devices(devices):
    devices = [devices] if type(devices) is not list else devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in devices)


def get_summary_writer(logs_path, project_id, version_id):
    path = os.path.join(logs_path, "{}_{}".format(project_id, version_id))
    if os.path.exists(path) and remove_if_exists:
        shutil.rmtree(path)
    summary_writer = SummaryWriter(log_dir=path)
    return summary_writer