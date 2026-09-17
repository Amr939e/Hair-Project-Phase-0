import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# 20 light -> deep skin-tone pairs (each becomes one colormap)
# ---------------------------------------------------------------------------
SKIN_PAIRS = [
    ("#FFF0E1", "#2A1205"),  # skin_01 balanced neutral
    ("#FCE3CE", "#3F1B08"),  # 02 warm peach -> warm brown
    ("#F8D6B8", "#52250B"),  # 03 golden -> rich brown
    ("#F3C9A2", "#642F0F"),  # 04 honey -> chestnut
    ("#EBB98C", "#753A13"),  # 05 olive -> deep tan
    ("#E2A876", "#864518"),  # 06 soft tan -> dark tan
    ("#D89861", "#96511E"),  # 07 golden tan -> warm brown
    ("#CC874F", "#A55D25"),  # 08 warm -> medium brown
    ("#BF763E", "#B36A2E"),  # 09 midrange warm
    ("#B16630", "#C0773A"),  # 10 midrange neutral
    ("#A25726", "#CB8448"),  # 11 brown -> golden
    ("#A25726", "#632810"),  # 12 brown -> dark
    ("#934A1E", "#D59158"),  # 13 deep -> light
    ("#833D18", "#DE9E6A"),  # 14 deep -> tan
    ("#733214", "#E5AB7C"),  # 15 deep -> soft tan
    ("#632810", "#ECB88E"),  # 16 dark -> light
    ("#53200C", "#F2C4A1"),  # 17 very dark -> pale
    ("#FDE7D6", "#4B2913"),  # 18 cool pale -> cool deep
    ("#F5CFB0", "#6D3C1C"),  # 19 neutral pair
    ("#EFC29B", "#8F5129"),  # 20 warm pair
    ("#E8B487", "#AE6A3B"),  # 21 warm midrange
]

# ---------------------------------------------------------------------------
# Build + register the 20 colormaps as skin_01 ... skin_20
# ---------------------------------------------------------------------------
SKIN_CMAPS = {}

for i, (light, deep) in enumerate(SKIN_PAIRS, start=1):
    name = f"skin_{i:02d}"
    cmap = LinearSegmentedColormap.from_list(name, [light, deep], N=256)
    SKIN_CMAPS[name] = cmap

    for cm in (cmap, cmap.reversed()):
        try:
            mpl.colormaps.register(cm)
        except ValueError:
            mpl.colormaps.register(cm, force=True)

# ---------------------------------------------------------------------------
# Named hair colours: hair_01 ... hair_20  (white -> black)
# ---------------------------------------------------------------------------
HAIR_COLORS_20 = [
    "#FFFFFF",  # hair_01  pure white
    "#F4F1EC",  # hair_02  warm white
    "#E9E2D6",  # hair_03  platinum / silver-white
    "#DDD2C0",  # hair_04  ash blonde (lightest)
    "#D1C1A8",  # hair_05  light ash blonde
    "#C5B08E",  # hair_06  light blonde
    "#B99E74",  # hair_07  golden blonde
    "#AD8C5B",  # hair_08  dark golden blonde
    "#A17A44",  # hair_09  honey blonde
    "#956830",  # hair_10  light brown
    "#89571F",  # hair_11  medium brown
    "#7D4714",  # hair_12  warm brown
    "#6F3A0E",  # hair_13  chestnut brown
    "#612F09",  # hair_14  dark brown
    "#532506",  # hair_15  deep brown
    "#451C04",  # hair_16  very dark brown
    "#371403",  # hair_17  near-black brown
    "#290D02",  # hair_18  soft black
    "#1A0701",  # hair_19  black
    "#000000",  # hair_20  true black
]

HAIR = {f"hair_{i:02d}": c for i, c in enumerate(HAIR_COLORS_20, start=1)}
