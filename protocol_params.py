"""
Protocol parameters
"""

# Electrophysiology parameters

SAMPLING_RATE = 20000  # Hz
PRE_STIM_DURATION = 0.200  # seconds
ELEC_STIM_START_TIME = 0.200  # seconds, Independent optical pulse
STIM_ARTIFACT_DURATION = 0.018  # seconds
PAIRED_OPT_START_TIME = 0.250  # seconds, Independent optical pulse
IND_OPT_START_TIME = 5.265  # seconds, Independent optical pulse
RESPONSE_DURATION = 0.050  # seconds
STIM_NAMES = ["elec", "opt-paried", "opt-ind"]  # Stimulation types
PPF_RATIO_THRESHOLD = 1.0  # Threshold for detecting PPF
NO_PPF_RATIO_THRESHOLD = 0.8  # Lower threshold for detecting No PPF


# Optical stimulation parameters
GRID_SIZE = (24, 24)
N_SKIP_ROWS = 1  # Number of rows to skip in between two grid spots
N_SKIP_COLS = 1  # Number of cols to skip in between two grid spots
