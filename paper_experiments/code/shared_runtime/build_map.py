"""
build_map.py
============
Defines a 10 × 10 m indoor map with two concrete walls that divide the space
into three equal thirds along the X-axis.

Wall layout
-----------
  Map size   : 10 m × 10 m (XY plane)
  Wall length: 6 m  (along Y)
  Wall thick : 0.5 m (along X)
  Wall height: 3 m  (along Z – tall enough for realistic ray tracing)

  The walls are centred in Y, leaving a 2 m gap at each end of the room.
  They are placed at the one-third and two-third positions along X:
    Wall 1 centre-X : 10/3 ≈ 3.33 m
    Wall 2 centre-X : 20/3 ≈ 6.67 m

Usage
-----
  from build_map import Map

  m = Map()
  scene = m.build()       # Sionna Scene with ray-tracing properties
  m.visualize()           # top-down 2-D plot of the room and walls
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Mitsuba 3 / Sionna imports
# ---------------------------------------------------------------------------
try:
    import sionna.rt as rt
except (ModuleNotFoundError, ImportError):  # Cached replay imports metadata only.
    rt = None  # type: ignore[assignment]


def _require_sionna_rt():
    if rt is None:
        raise RuntimeError("sionna.rt is required to build a live ray-tracing scene")
    return rt

# ---------------------------------------------------------------------------
# Map geometry constants
# ---------------------------------------------------------------------------
MAP_SIZE    = 10.0   # m – side length of the square room
WALL_LENGTH =  6.0   # m – extent of each wall along Y
WALL_THICK  =  0.5   # m – wall thickness along X
WALL_HEIGHT =  3.0   # m – wall height along Z

# Centre positions along X (one-third and two-thirds of the map)
_WALL1_X = MAP_SIZE / 3          # ≈ 3.33 m
_WALL2_X = 2 * MAP_SIZE / 3      # ≈ 6.67 m
# Walls are centred in Y
_WALL_Y_CENTER = MAP_SIZE / 2    # = 5.0 m
# Z: place base at z=0, extend to WALL_HEIGHT
_WALL_Z_CENTER = WALL_HEIGHT / 2 # = 1.5 m

LARGE_MAP_SIZE = 50.0


def _write_cube_ply(path: str):
    """
    Write a unit cube as a binary PLY file (the format Sionna's bundled
    scenes use).  Vertices span [-0.5, 0.5] on all axes.
    """
    import struct
    vertices = [
        (-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5), (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5), (-0.5,  0.5,  0.5),
    ]
    # Each face as two triangles
    faces = [
        (0,1,2), (0,2,3),  # bottom
        (4,6,5), (4,7,6),  # top
        (0,5,1), (0,4,5),  # front
        (1,6,2), (1,5,6),  # right
        (2,7,3), (2,6,7),  # back
        (3,4,0), (3,7,4),  # left
    ]

    n_verts = len(vertices)
    n_faces = len(faces)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for v in vertices:
            f.write(struct.pack("<fff", *v))
        for face in faces:
            f.write(struct.pack("<B3i", 3, *face))


def _build_scene_xml_generic(ply_path: str, walls: list) -> str:
    """
    Build a Mitsuba 3 XML scene string from a list of wall definitions.
    Each wall is (id_str, x_center, y_center, thickness, length, height).
    """
    xml_parts = [
        '<scene version="2.1.0">',
        '    <!-- ITU concrete material -->',
        '    <bsdf type="itu-radio-material" id="mat-concrete">',
        '        <string name="type" value="concrete"/>',
        '    </bsdf>'
    ]

    for (name, cx, cy, th, l, h) in walls:
        xml_parts.append(textwrap.dedent(f"""\
            <shape type="ply" id="{name}">
                <string name="filename" value="{ply_path}"/>
                <boolean name="face_normals" value="true"/>
                <transform name="to_world">
                    <scale x="{th}" y="{l}" z="{h}"/>
                    <translate x="{cx:.4f}" y="{cy:.4f}" z="{h/2:.4f}"/>
                </transform>
                <ref id="mat-concrete" name="bsdf"/>
            </shape>"""))

    xml_parts.append('</scene>')
    return "\n".join(xml_parts)


def _build_scene_xml(ply_path: str) -> str:
    """Legacy builder for the 10x10 map."""
    walls = [
        ("wall_1", _WALL1_X, _WALL_Y_CENTER, WALL_THICK, WALL_LENGTH, WALL_HEIGHT),
        ("wall_2", _WALL2_X, _WALL_Y_CENTER, WALL_THICK, WALL_LENGTH, WALL_HEIGHT),
    ]
    return _build_scene_xml_generic(ply_path, walls)


class Map:
    """
    A 10 × 10 m radio-propagation map with two concrete dividing walls.
    """

    def __init__(self, frequency: float = 3.5e9):
        self.frequency = frequency
        self.scene: rt.Scene | None = None
        self._xml_path: str | None = None
        self._ply_path: str | None = None

    def build(self) -> rt.Scene:
        rt = _require_sionna_rt()
        """
        Construct the Sionna ray-tracing scene.
        Geometry is defined via a PLY cube mesh; material via the
        Sionna-specific 'itu-radio-material' Mitsuba plugin.
        """
        tmp_dir = tempfile.gettempdir()

        # 1. Write the cube PLY file
        self._ply_path = os.path.join(tmp_dir, "sionna_wall_cube.ply")
        _write_cube_ply(self._ply_path)

        # 2. Write the XML scene file
        xml_tmp = tempfile.NamedTemporaryFile(
            suffix=".xml", delete=False, mode="w", encoding="utf-8"
        )
        xml_tmp.write(_build_scene_xml(self._ply_path))
        xml_tmp.close()
        self._xml_path = xml_tmp.name

        # 3. Load into Sionna
        scene = rt.load_scene(self._xml_path)
        scene.frequency = self.frequency

        # 4. Antennas
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        # 5. Reinforce Material (ensure objects are recognized for reflection)
        concrete = rt.RadioMaterial("itu_concrete", relative_permittivity=5.31, conductivity=0.0326)
        scene.add(concrete)
        for name in ["wall_1", "wall_2"]:
            obj = scene.get(name)
            if obj:
                obj.radio_material = concrete

        self.scene = scene
        return scene

    def visualize(self, ax: plt.Axes | None = None, show: bool = True) -> plt.Axes:
        """
        Render a top-down (XY-plane) 2-D floor-plan of the map.

        The room boundary, the two concrete walls, and the open gaps are
        drawn with labelled patches.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into.  A new figure is created if *None*.
        show : bool
            If *True* (default), ``plt.show()`` is called at the end.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # --- Room boundary -------------------------------------------------
        room = mpatches.Rectangle(
            (0, 0), MAP_SIZE, MAP_SIZE,
            linewidth=2, edgecolor="black", facecolor="#f0f0f0", zorder=1,
        )
        ax.add_patch(room)

        # --- Walls ---------------------------------------------------------
        wall_color = "#5a7d9a"
        gap = (MAP_SIZE - WALL_LENGTH) / 2  # 2 m gap at each end in Y

        for wx in (_WALL1_X, _WALL2_X):
            # The wall rectangle in the XY plane:
            #   x in [wx - WALL_THICK/2, wx + WALL_THICK/2]
            #   y in [gap, gap + WALL_LENGTH]
            wall_rect = mpatches.Rectangle(
                (wx - WALL_THICK / 2, gap),
                WALL_THICK, WALL_LENGTH,
                linewidth=1, edgecolor="black", facecolor=wall_color, zorder=2,
            )
            ax.add_patch(wall_rect)

        # --- Dimension annotations ----------------------------------------
        ax.annotate(
            "", xy=(_WALL1_X, -0.4), xytext=(0, -0.4),
            arrowprops=dict(arrowstyle="<->", color="gray"),
        )
        ax.text(
            _WALL1_X / 2, -0.65, f"{_WALL1_X:.2f} m",
            ha="center", va="top", fontsize=8, color="gray",
        )

        # --- Labels --------------------------------------------------------
        ax.text(
            _WALL1_X, _WALL_Y_CENTER + WALL_LENGTH / 2 + 0.15,
            "Wall 1", ha="center", va="bottom", fontsize=9, color=wall_color,
        )
        ax.text(
            _WALL2_X, _WALL_Y_CENTER + WALL_LENGTH / 2 + 0.15,
            "Wall 2", ha="center", va="bottom", fontsize=9, color=wall_color,
        )

        # --- Vertical dashed third-lines (for reference) ------------------
        for lx in (_WALL1_X, _WALL2_X):
            ax.axvline(lx, color="gray", linewidth=0.7, linestyle="--", zorder=0)

        # --- Gaps (open passages) annotation ------------------------------
        for wx in (_WALL1_X, _WALL2_X):
            for gy in (gap / 2, gap + WALL_LENGTH + gap / 2):
                ax.text(
                    wx, gy, "gap\n2 m",
                    ha="center", va="center", fontsize=7, color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )

        # --- Axes formatting ----------------------------------------------
        legend_patch = mpatches.Patch(facecolor=wall_color, edgecolor="black",
                                       label="Concrete wall (0.5 m thick, 6 m long)")
        ax.legend(handles=[legend_patch], loc="upper right", fontsize=8)

        ax.set_xlim(-0.8, MAP_SIZE + 0.3)
        ax.set_ylim(-1.0, MAP_SIZE + 0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(
            f"10 × 10 m Map — Top View\n"
            f"f = {self.frequency/1e9:.2f} GHz  |  "
            f"material: ITU concrete  |  wall height: {WALL_HEIGHT} m",
            fontsize=10,
        )
        ax.set_xticks(np.arange(0, MAP_SIZE + 1, 1))
        ax.set_yticks(np.arange(0, MAP_SIZE + 1, 1))
        ax.grid(True, linestyle=":", alpha=0.4)

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def cleanup(self):
        """Remove temporary files created by :meth:`build`."""
        for path_attr in ("_xml_path", "_ply_path"):
            path = getattr(self, path_attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
                setattr(self, path_attr, None)


class LargeMap:
    """
    A 50 × 50 m radio-propagation map with multiple concrete obstacles
    (12 in total) that hinder signal propagation.
    """

    def __init__(self, frequency: float = 3.5e9):
        self.frequency = frequency
        self.scene: rt.Scene | None = None
        self._xml_path: str | None = None
        self._ply_path: str | None = None
        
        # Define 12 walls (name, cx, cy, thickness, length, height)
        self.walls = [
            # Vertical walls
            ("v1", 10.0, 20.0, 0.5, 20.0, 3.0),
            ("v2", 20.0, 30.0, 0.5, 20.0, 3.0),
            ("v3", 30.0, 20.0, 0.5, 20.0, 3.0),
            ("v4", 40.0, 30.0, 0.5, 20.0, 3.0),
            # Horizontal walls
            ("h1", 20.0, 10.0, 15.0, 0.5, 3.0),
            ("h2", 30.0, 20.0, 15.0, 0.5, 3.0),
            ("h3", 20.0, 30.0, 15.0, 0.5, 3.0),
            ("h4", 30.0, 40.0, 15.0, 0.5, 3.0),
            # Smaller blocking blocks
            ("b1",  5.0,  5.0, 2.0, 2.0, 3.0),
            ("b2", 45.0,  5.0, 2.0, 2.0, 3.0),
            ("b3",  5.0, 45.0, 2.0, 2.0, 3.0),
            ("b4", 45.0, 45.0, 2.0, 2.0, 3.0),
        ]

    def build(self) -> rt.Scene:
        rt = _require_sionna_rt()
        tmp_dir = tempfile.gettempdir()
        self._ply_path = os.path.join(tmp_dir, "sionna_largemap_cube.ply")
        _write_cube_ply(self._ply_path)

        xml_tmp = tempfile.NamedTemporaryFile(
            suffix=".xml", delete=False, mode="w", encoding="utf-8"
        )
        xml_tmp.write(_build_scene_xml_generic(self._ply_path, self.walls))
        xml_tmp.close()
        self._xml_path = xml_tmp.name

        scene = rt.load_scene(self._xml_path)
        scene.frequency = self.frequency
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        concrete = rt.RadioMaterial("itu_concrete", relative_permittivity=5.31, conductivity=0.0326)
        scene.add(concrete)
        for name, *_ in self.walls:
            obj = scene.get(name)
            if obj:
                obj.radio_material = concrete

        self.scene = scene
        return scene

    def visualize(self, ax: plt.Axes | None = None, show: bool = True) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        room = mpatches.Rectangle(
            (0, 0), LARGE_MAP_SIZE, LARGE_MAP_SIZE,
            linewidth=2, edgecolor="black", facecolor="#f8f8f8", zorder=1,
        )
        ax.add_patch(room)

        wall_color = "#5a7d9a"
        for name, cx, cy, th, l, h in self.walls:
            wall_rect = mpatches.Rectangle(
                (cx - th / 2, cy - l / 2),
                th, l,
                linewidth=1, edgecolor="black", facecolor=wall_color, zorder=2,
            )
            ax.add_patch(wall_rect)

        ax.set_xlim(-2, LARGE_MAP_SIZE + 2)
        ax.set_ylim(-2, LARGE_MAP_SIZE + 2)
        ax.set_aspect("equal")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(f"50 × 50 m Large Map - 12 Obstacles\nf = {self.frequency/1e9:.2f} GHz", fontsize=11)
        ax.set_xticks(np.arange(0, LARGE_MAP_SIZE + 1, 10))
        ax.set_yticks(np.arange(0, LARGE_MAP_SIZE + 1, 10))
        ax.grid(True, linestyle=":", alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def cleanup(self):
        for path_attr in ("_xml_path", "_ply_path"):
            path = getattr(self, path_attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
                setattr(self, path_attr, None)



HUGE_MAP_SIZE = 100.0

class Complex100mMap:
    """
    A 100 × 100 m radio-propagation map with 4 distinct quarters:
    - Q1 (SW): Sparse pillars (Park-like)
    - Q2 (SE): Dense office-like layout (Thin walls, many corridors)
    - Q3 (NW): Warehouse-style (Long vertical aisles)
    - Q4 (NE): Urban canyon (Large square blocks)
    
    Trap-free design: Uses solid blocks or open passages logic.
    """

    def __init__(self, frequency: float = 3.5e9):
        self.quarter_walls = [[], [], [], [], []] # 0-3 for Hetero, 4 for common ground
        self.size = HUGE_MAP_SIZE
        self.frequency = frequency
        self.scene: rt.Scene | None = None
        self._xml_path: str | None = None
        self._ply_path: str | None = None
        
        # Initial build
        self.update_zone(0, 0)
        self.update_zone(1, 0)
        self.update_zone(2, 0)
        self.update_zone(3, 0)

    @property
    def walls(self):
        """Flattened wall list from all quarters."""
        return [w for q in self.quarter_walls for w in q]

    def update_zone(self, zone_idx: int, step: int):
        """Triggers a re-generation of a specific quarter with some randomness."""
        self.quarter_walls[zone_idx] = []
        if zone_idx == 0: self._build_q1_sparse(step)
        elif zone_idx == 1: self._build_q2_dense(step)
        elif zone_idx == 2: self._build_q3_warehouse(step)
        elif zone_idx == 3: self._build_q4_urban(step)

    def _build_q1_sparse(self, step=0):
        """SW Quarter (0-50, 0-50): A few heavy pillars and 2 diagonal barriers."""
        rng = np.random.default_rng(seed=step)
        q = self.quarter_walls[0]
        # 4 Central Pillars (randomly shifted slightly)
        offsets = rng.uniform(-4, 4, (4, 2))
        q.append(("q1_p1", 15+offsets[0,0], 15+offsets[0,1], 5, 5, 3))
        q.append(("q1_p2", 35+offsets[1,0], 15+offsets[1,1], 5, 5, 3))
        q.append(("q1_p3", 15+offsets[2,0], 35+offsets[2,1], 5, 5, 3))
        q.append(("q1_p4", 35+offsets[3,0], 35+offsets[3,1], 5, 5, 3))
        
        # Diagonal barriers
        for i in range(5):
            q.append((f"q1_d1_{i}", 5+i*8, 45-i*8, 4, 1, 3))

    def _build_q2_dense(self, step=0):
        """SE Quarter (50-100, 0-50): Open office layout with staggered partitions."""
        rng = np.random.default_rng(seed=step)
        q = self.quarter_walls[1]
        for x in range(60, 100, 15):
            for y in range(10, 50, 15):
                # Random Rotation toggle
                is_rotated = rng.choice([True, False])
                if is_rotated:
                    q.append((f"q2_v_{x}_{y}", x, y, 8, 0.5, 3))
                else:
                    q.append((f"q2_v_{x}_{y}", x, y, 0.5, 8, 3))
        
        # Base blocks
        for y in [5, 20, 35]:
            q.append((f"q2_extra_{y}", 80 + rng.uniform(-2,2), y, 10, 0.8, 3))

    def _build_q3_warehouse(self, step=0):
        """NW Quarter (0-50, 50-100): Long vertical storage aisles."""
        rng = np.random.default_rng(seed=step)
        q = self.quarter_walls[2]
        for x in [10, 20, 30, 40]:
            h_var = rng.uniform(0, 5)
            q.append((f"q3_w_{x}_1", x, 65 + h_var, 1.5, 20, 4))
            q.append((f"q3_w_{x}_2", x, 90, 1.5, 15 - h_var, 4))

    def _build_q4_urban(self, step=0):
        """NE Quarter (50-100, 50-100): Large solid blocks (city buildings)."""
        rng = np.random.default_rng(seed=step)
        q = self.quarter_walls[3]
        coords = [(60, 60), (85, 60), (60, 85), (85, 85), (72, 72)]
        for i, (cx, cy) in enumerate(coords):
            # Scale variation
            s = rng.uniform(8, 12)
            q.append((f"q4_block_{i}", cx + rng.uniform(-2,2), cy + rng.uniform(-2,2), s, s, 10))

    def build(self) -> rt.Scene:
        rt = _require_sionna_rt()
        tmp_dir = tempfile.gettempdir()
        self._ply_path = os.path.join(tmp_dir, f"sionna_huge_cube_{id(self)}.ply")
        _write_cube_ply(self._ply_path)

        xml_content = _build_scene_xml_generic(self._ply_path, self.walls)
        xml_tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
        xml_tmp.write(xml_content)
        xml_tmp.close()
        self._xml_path = xml_tmp.name

        scene = rt.load_scene(self._xml_path)
        scene.frequency = self.frequency
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        concrete = rt.RadioMaterial("itu_concrete", relative_permittivity=5.31, conductivity=0.0326)
        scene.add(concrete)
        for name, *_ in self.walls:
            obj = scene.get(name)
            if obj: obj.radio_material = concrete

        self.scene = scene
        return scene

    def visualize(self, ax: plt.Axes | None = None, show: bool = True) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Room floor
        room = mpatches.Rectangle((0, 0), HUGE_MAP_SIZE, HUGE_MAP_SIZE,
                                   linewidth=2, edgecolor="black", facecolor="#fdfdfd", zorder=1)
        ax.add_patch(room)

        # Quarters boundaries (dashed lines)
        ax.axvline(50, color="#ccc", linestyle="--", linewidth=1, zorder=0)
        ax.axhline(50, color="#ccc", linestyle="--", linewidth=1, zorder=0)

        # Walls
        wall_color = "#4682b4"
        for name, cx, cy, th, l, h in self.walls:
            rect = mpatches.Rectangle((cx - th/2, cy - l/2), th, l,
                                      linewidth=0.5, edgecolor="black", facecolor=wall_color, zorder=2)
            ax.add_patch(rect)

        # Annotations
        ax.text(25, 25, "Q1: Sparse", ha='center', alpha=0.5, fontsize=12)
        ax.text(75, 25, "Q2: Dense", ha='center', alpha=0.5, fontsize=12)
        ax.text(25, 75, "Q3: Warehouse", ha='center', alpha=0.5, fontsize=12)
        ax.text(75, 75, "Q4: Urban", ha='center', alpha=0.5, fontsize=12)

        ax.set_xlim(-5, 105); ax.set_ylim(-5, 105); ax.set_aspect("equal")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_title(f"100 × 100 m Complex Map - Heterogeneous Quarters")
        ax.grid(True, linestyle=":", alpha=0.2)

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def cleanup(self):
        for path_attr in ("_xml_path", "_ply_path"):
            path = getattr(self, path_attr, None)
            if path and os.path.exists(path):
                try: os.remove(path)
                except OSError: pass
                setattr(self, path_attr, None)


class Munich100mMap:
    """
    A 100 × 100 m region of the Munich city model (Marienplatz).
    Original bounds in Sionna: X [0, 100], Y [-200, -100].
    We translate the entire scene by +199m in Y to align with the [0, 100] sim space.
    """

    def __init__(self, frequency: float = 3.5e9):
        self.size = 100.0
        self.frequency = frequency
        self.scene: rt.Scene | None = None
        self.walls = [] # For collision detection, we would need a grid, but we'll stick to buildings

    def build(self) -> rt.Scene:
        # 1. Load built-in Munich
        scene = rt.load_scene(rt.scene.munich)
        scene.frequency = self.frequency
        
        # 2. Antennas
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        
        self.walls = []
        for name, obj in scene.objects.items():
            if "ground" in name.lower():
                continue
            
            # Translate Sionna object for the simulation
            obj.translate([0, 199, 0])
            
            try:
                mesh = obj.mi_mesh
                # vertices returned by Sionna are in local space? No, usually world after translation?
                # Actually, in Sionna rt, obj.translate() just adds a transform.
                # The vertex_positions_buffer() usually gives local coordinates.
                # Let's get them and apply the +199 manually for our wall logic.
                verts = np.array(mesh.vertex_positions_buffer()).reshape(-1, 3)
                x_min, y_min = np.min(verts[:, 0]), np.min(verts[:, 1]) + 199
                x_max, y_max = np.max(verts[:, 0]), np.max(verts[:, 1]) + 199
                
                # Check if BUILDING is in our 100m sim area
                if x_max < 0 or x_min > 100 or y_max < 0 or y_min > 100: continue
                
                # Format: (name, cx, cy, thickness/width_x, length/width_y, height)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                self.walls.append((name, cx, cy, x_max-x_min, y_max-y_min, 10.0))
            except: continue
            
        self.scene = scene
        return scene

    def visualize(self, ax: plt.Axes | None = None, show: bool = True) -> plt.Axes:
        if ax is None: fig, ax = plt.subplots(figsize=(10, 10))
        
        # Room floor
        room = mpatches.Rectangle((0, 0), 100, 100, linewidth=2, edgecolor="black", facecolor="#fdfdfd", zorder=1)
        ax.add_patch(room)
        
        # Buildings from our walls list
        for name, cx, cy, w, l, h in self.walls:
            rect = mpatches.Rectangle((cx - w/2, cy - l/2), w, l,
                                      linewidth=0.5, edgecolor="#333", facecolor="#555", alpha=0.8, zorder=2)
            ax.add_patch(rect)

        # Zone boundaries
        ax.axvline(50, color="#ccc", linestyle="--", linewidth=1, zorder=0)
        ax.axhline(50, color="#ccc", linestyle="--", linewidth=1, zorder=0)
        
        ax.set_xlim(-5, 105); ax.set_ylim(-5, 105); ax.set_aspect("equal")
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
        ax.set_title("100 × 100 m Munich Map (Static)")
        
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def cleanup(self):
        # built-in scenes don't need file cleanup usually
        pass


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing 100x100 Complex Map...")
    hm = Complex100mMap(frequency=3.5e9)
    s = hm.build()
    print(f"  Scene loaded with {len(hm.walls)} obstacles.")
    hm.visualize(show=True)
    hm.cleanup()
