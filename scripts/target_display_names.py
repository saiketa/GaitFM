
TARGET_DISPLAY_NAMES = {
    "gait_speed": "Gait speed",
    "step_length": "Step length",
    "step_time": "Step time",
}


def target_display_name(target):
    return TARGET_DISPLAY_NAMES.get(target, target)
