

CSV_PATH = "data/BillionairePositionsMapped.csv"
OUTPUT_DIR = "output"

# CSV columns expected by the loader:
# Name, NetWorth_Billions, X, Y, Z, ForceBasedOnAge
MAX_PARTICLES = 50

# Initial velocity settings
# INWARD_BIAS = 1.0 means fully inward movement.
# INWARD_BIAS = 0.0 means fully random movement.
INWARD_BIAS = 0.51
VELOCITY_SCALE = 0.05
RANDOM_SEED = 42

# Physics settings
G = 0.005
DT = 0.02
STEPS = 500
SOFTENING = 2.0
SAVE_EVERY = 3

# Export settings
SAVE_GIF = True
SAVE_VVVV_CSV = True


# Output file names
OUTPUT_GIF_NAME = "billionaires_nbody_3d.gif"
OUTPUT_VVVV_CSV_NAME = "billionaires_nbody_for_vvvv.csv"

# Animation settings
FPS = 30
INTERVAL_MS = 30
DPI = 120

# Display settings
POINT_MIN_SIZE = 10
POINT_MAX_EXTRA_SIZE = 200
CAMERA_ELEVATION = 25
CAMERA_AZIMUTH = 45
CAMERA_ROTATION_SPEED = 0.0  
