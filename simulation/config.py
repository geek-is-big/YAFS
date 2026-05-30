from enum import Enum


class TimeUnit(Enum):
    SECOND = 1
    MILLISECOND = 2


# What's bad
# Configs become outdated, when codex changes something, and it doesn't delete outdated configs.

# RULE: before refactor something, it should be covered with tests.


TIME_UNIT = TimeUnit.SECOND

USER_SPEED_MPS = 1.4
SIM_STEP_SECONDS = 1

ENABLE_ANIMATION = False
ANIMATION_STEP_STRIDE = 5
ANIMATION_MAX_FRAMES = 300
ANIMATION_FORMAT = "gif"  # "mp4" or "gif"

PRINT_NEAREST_DISTANCES = False
NEAREST_NODES_TO_PRINT = 3

RSSI_REFERENCE_DISTANCE_M = 1.0
RSSI_AT_REFERENCE_DBM = -20.0
RSSI_ENVIRONMENT_COEFF = 2.7

# Device-specific WiFi/BLE profiles are defined in simulation/device_profiles.py.


# TODO: Maybe, the task should be created be the class constructor or method,
# cause it's currently unclear what parameters should be used to create the task,
# and it may lead to code duplication in the future when we will add more tasks.
# Also, this config is a mess, mayby it's better to create static array of tasks with all parameters,
# and then just read them in the code when we need to create a task.

# Task1 profile (ECG classification)
TASK1_ID = "Task1"
TASK1_PERIOD_S = 10
TASK1_COMPLEXITY_MI = 500
TASK1_DATA_SIZE_KB = 36
TASK1_MAX_RESPONSE_TIME_S = 15
TASK1_CLASSIFICATION = "critical analysis"
TASK_EXECUTION_MODE = "EDGE"  # EDGE | FOG

OPTIMIZATION_METHOD = "LP"
ALPHA_LATENCY = 0.0
BETA_ENERGY = 1.0
OFFLOADING_DECISION_PERIOD_S = 1
TASK1_MIGRATION_STATE_SIZE_KB = 20.0

# Task2 profile (SmartWatch audio analysis)
TASK2_ID = "Task2"
TASK2_PERIOD_S = 10
TASK2_COMPLEXITY_MI = 1000
TASK2_DATA_SIZE_KB = 500
TASK2_TASK_SIZE_KB = 256
TASK2_RESULT_SIZE_KB = 4
TASK2_MAX_RESPONSE_TIME_S = None  # unlimited
TASK2_CLASSIFICATION = "data analysis"
TASK2_TRANSPORT = "UDP"

# Task3 profile (SmartWatch temperature/context management)
TASK3_ID = "Task3"
TASK3_PERIOD_S = 60
TASK3_COMPLEXITY_MI = 10
TASK3_DATA_SIZE_KB = 10
TASK3_TASK_SIZE_KB = 50
TASK3_RESULT_SIZE_B = 707
TASK3_MAX_RESPONSE_TIME_S = 0.5
TASK3_CLASSIFICATION = "context management"
TASK3_TRANSPORT = "TCP"

# Task4 profile (SmartWatch humidity/context management)
TASK4_ID = "Task4"
TASK4_PERIOD_S = 60
TASK4_COMPLEXITY_MI = 10
TASK4_DATA_SIZE_KB = 10
TASK4_TASK_SIZE_KB = 50
TASK4_RESULT_SIZE_B = 707
TASK4_MAX_RESPONSE_TIME_S = 0.5
TASK4_CLASSIFICATION = "context management"
TASK4_TRANSPORT = "TCP"

# Task5 profile (ECG compression/context management)
TASK5_ID = "Task5"
TASK5_PERIOD_S = 10
TASK5_COMPLEXITY_MI = 211
TASK5_DATA_SIZE_KB = 240
TASK5_TASK_SIZE_KB = 250
TASK5_RESULT_SIZE_KB = 16.8
TASK5_MAX_RESPONSE_TIME_S = 0.5
TASK5_CLASSIFICATION = "context management"
TASK5_TRANSPORT = "TCP"

# Static scenario link model (fixed RSSI for monitor-driven link updates).
STATIC_LINK_RSSI_DBM = -85.0
