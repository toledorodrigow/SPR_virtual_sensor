from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# BioNavis/Navi processed parameters.  The names and ordering are grounded in
# the user's DataViewer CSV export and the Navi sensorgram parameter list.
# The first twelve quantities are also numerically consistent with the shapes
# expected from the full angular curve (angles, intensities, peak width, etc.).
# Fields whose vendor names can be established directly from the supplied
# BioNavis DataViewer CSV. Two binary quantities are intentionally ignored rather
# than shown as metric_XX placeholders or assigned speculative names.
SENSORGRAM_FIELDS = [
    "PeakMinAngle",
    "PeakMinIntensity",
    "TirAngle",
    "TirIntensity",
    "PureKinetics",
    "PeakAmplitude",
    "PW_0.25_Intensity",
    "PW_0.25Height",
    "SteepestFallAngle",
    "SteepestFallIntensity",
    "Centroid",
    "Weighted Centroid",
    "Flow",
    "SensorTemp",
]

# The supplied BioNavis CSV uses these wavelength labels for device 326.
# read_spr2() also extracts the wavelength table from calibration P1500 when
# available, so this is only a conservative fallback.
DEFAULT_CHANNEL_WAVELENGTHS = {0: 670, 1: 785, 2: 670, 4: 785, 5: 850, 6: 980}


@dataclass
class Scan:
    rtime_ms: float
    step_len: float
    direction: str
    metadata: Dict[str, object]
    channels: Dict[int, np.ndarray]
    start_positions: Dict[int, float]


@dataclass
class SPRDataset:
    spr2_path: Path
    header: Dict[str, str] = field(default_factory=dict)
    calibration_info: Dict[str, str] = field(default_factory=dict)
    calibration_curves: Dict[int, np.ndarray] = field(default_factory=dict)
    calibration_channel_info: Dict[int, Dict[str, str]] = field(default_factory=dict)
    calibration_prm_sets: Dict[str, List[str]] = field(default_factory=dict)
    channel_wavelengths: Dict[int, int] = field(default_factory=dict)
    init_scan: Optional[Scan] = None
    scans: List[Scan] = field(default_factory=list)
    events: List[Dict[str, object]] = field(default_factory=list)
    binary_path: Optional[Path] = None
    binary_table: Optional[pd.DataFrame] = None
    angle_calibration: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    bulk_sensitivity: Dict[int, float] = field(default_factory=dict)
    bulk_sensitivity_source: str = ""
    bulk_reference: Dict[int, Dict[str, float]] = field(default_factory=dict)
    bulk_reference_source: str = "first processed scan"

    @property
    def channel_ids(self) -> List[int]:
        ids = set()
        for scan in self.scans[:10]:
            ids.update(scan.channels.keys())
        return sorted(ids)

    @property
    def scan_times_min(self) -> np.ndarray:
        if self.binary_table is not None and not self.binary_table.empty:
            # one time per scan
            return self.binary_table.groupby("scan_index", sort=True)["time_min"].first().to_numpy()
        return np.array([s.rtime_ms / 60000.0 for s in self.scans], dtype=float)

    def x_axis(self, scan_index: int, channel_id: int, mode: str = "angle") -> Tuple[np.ndarray, str]:
        scan = self.scans[scan_index]
        y = scan.channels[channel_id]
        n = len(y)
        sign = -1.0 if scan.direction.lower().startswith("back") else 1.0
        start = float(scan.start_positions[channel_id])
        motor = start + sign * float(scan.step_len) * np.arange(n, dtype=float)

        if mode == "index":
            return np.arange(n, dtype=float), "Point index"
        if mode == "motor":
            return motor, "Motor position (native units)"
        if mode == "angle" and channel_id in self.angle_calibration:
            slope, intercept, _rmse = self.angle_calibration[channel_id]
            return slope * motor + intercept, "Angle (deg, calibrated from processed binary)"
        return motor, "Motor position (native units)"

    def channel_label(self, channel_id: int) -> str:
        wl = self.channel_wavelengths.get(channel_id, DEFAULT_CHANNEL_WAVELENGTHS.get(channel_id))
        if wl is not None and 200 <= int(wl) <= 2500:
            return f"L{channel_id + 1} {int(wl)}nm"
        return f"L{channel_id + 1}"

    def set_bulk_sensitivity(self, channel_id: int, value: Optional[float]) -> None:
        """Set or clear the bulk-correction sensitivity for one channel."""
        cid = int(channel_id)
        if value is None or not np.isfinite(float(value)):
            self.bulk_sensitivity.pop(cid, None)
        else:
            self.bulk_sensitivity[cid] = float(value)
        self._refresh_pure_kinetics_column()

    def _ensure_bulk_reference(self) -> None:
        """Create a per-channel reference from the first processed scan if absent."""
        if self.binary_table is None or self.binary_table.empty:
            return
        for cid, g in self.binary_table.groupby("channel_id", sort=False):
            cid = int(cid)
            if cid in self.bulk_reference or g.empty:
                continue
            row = g.sort_values("scan_index").iloc[0]
            self.bulk_reference[cid] = {
                "PeakMinAngle": float(row["PeakMinAngle"]),
                "TirAngle": float(row["TirAngle"]),
            }

    def set_bulk_reference_window(self, start_min: float, end_min: float) -> None:
        """Use the mean of a time window as the bulk-correction reference."""
        if self.binary_table is None or self.binary_table.empty:
            return
        lo, hi = sorted((float(start_min), float(end_min)))
        refs: Dict[int, Dict[str, float]] = {}
        for cid, g in self.binary_table.groupby("channel_id"):
            w = g[(g["time_min"] >= lo) & (g["time_min"] <= hi)]
            if w.empty:
                continue
            refs[int(cid)] = {
                "PeakMinAngle": float(w["PeakMinAngle"].mean()),
                "TirAngle": float(w["TirAngle"].mean()),
            }
        if not refs:
            raise ValueError("The selected baseline interval contains no processed scans.")
        self.bulk_reference = refs
        self.bulk_reference_source = f"mean over {lo:.6g} to {hi:.6g} min"
        self._refresh_pure_kinetics_column()

    def _refresh_pure_kinetics_column(self) -> None:
        """Update the bulk-corrected signal from ANGLE VARIATIONS.

        For each channel and reference state 0:
            DeltaPeak = PeakMinAngle - PeakMinAngle_0
            DeltaTIR  = TirAngle     - TirAngle_0
            PureKinetics = DeltaPeak - S_bulk * DeltaTIR

        PureKinetics is the corrected VARIATION itself. No absolute/vendor
        PureKinetics offset is added back. Therefore the selected reference state
        is zero by construction.
        """
        if self.binary_table is None or self.binary_table.empty:
            return
        self._ensure_bulk_reference()
        bt = self.binary_table
        sens = bt["channel_id"].map(self.bulk_sensitivity).astype(float)
        peak0 = bt["channel_id"].map(lambda c: self.bulk_reference.get(int(c), {}).get("PeakMinAngle", np.nan)).astype(float)
        tir0 = bt["channel_id"].map(lambda c: self.bulk_reference.get(int(c), {}).get("TirAngle", np.nan)).astype(float)
        bt["BulkSensitivity"] = sens
        bt["PeakMinAngleReference"] = peak0
        bt["TirAngleReference"] = tir0
        bt["DeltaPeakMinAngle"] = bt["PeakMinAngle"] - peak0
        bt["DeltaTirAngle"] = bt["TirAngle"] - tir0
        bt["BulkCorrection"] = sens * bt["DeltaTirAngle"]
        bt["PureKineticsDelta"] = bt["DeltaPeakMinAngle"] - bt["BulkCorrection"]
        bt["PureKinetics"] = bt["PureKineticsDelta"]

    def pure_kinetics(self, channel_id: int) -> pd.DataFrame:
        """Return time/PureKinetics for a channel when a sensitivity is available."""
        if self.binary_table is None:
            return pd.DataFrame(columns=["time_min", "PureKinetics"])
        self._refresh_pure_kinetics_column()
        g = self.binary_table[self.binary_table["channel_id"] == int(channel_id)].copy()
        return g[["time_min", "PureKinetics"]].sort_values("time_min")

    def processed_sensorgram(self, field: str = "PeakMinAngle") -> pd.DataFrame:
        if self.binary_table is None:
            return pd.DataFrame()
        if field == "PureKinetics":
            self._refresh_pure_kinetics_column()
        if field not in self.binary_table.columns:
            raise KeyError(field)
        return self.binary_table.pivot(index="time_min", columns="channel_id", values=field).sort_index()


def _text(elem: Optional[ET.Element]) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _parse_tctrl(parent: ET.Element) -> Dict[str, Dict[str, str]]:
    result = {}
    for tc in parent.findall("t_ctrl"):
        key = tc.attrib.get("ch", "unknown")
        result[key] = {child.tag: _text(child) for child in tc}
    return result


def _parse_scan_element(elem: ET.Element) -> Scan:
    channels: Dict[int, np.ndarray] = {}
    starts: Dict[int, float] = {}
    for ch in elem.findall("ch"):
        cid = int(ch.attrib["number"])
        arr = np.fromstring(ch.text or "", sep=";", dtype=np.float32)
        channels[cid] = arr
        starts[cid] = float(ch.attrib.get("start_pos", elem.findtext("scan_meta/stepper_pos", default="0")))

    meta_elem = elem.find("scan_meta")
    metadata: Dict[str, object] = {}
    if meta_elem is not None:
        for child in meta_elem:
            if child.tag != "t_ctrl":
                metadata[child.tag] = _text(child)
        metadata["t_ctrl"] = _parse_tctrl(meta_elem)

    return Scan(
        rtime_ms=float(elem.attrib.get("rtime", "0")),
        step_len=float(elem.attrib.get("step_len", "1")),
        direction=elem.attrib.get("dir", "Forward"),
        metadata=metadata,
        channels=channels,
        start_positions=starts,
    )


def read_spr2(path: str | Path) -> SPRDataset:
    path = Path(path)
    ds = SPRDataset(spr2_path=path)

    # Streaming parser: the source can be tens or hundreds of MB.
    for event, elem in ET.iterparse(path, events=("end",)):
        tag = elem.tag

        if tag == "header":
            for child in elem:
                if child.tag == "t_ctrl":
                    continue
                ds.header[child.tag] = _text(child)
            tctrl = _parse_tctrl(elem)
            for key, val in tctrl.items():
                for subkey, subval in val.items():
                    ds.header[f"t_ctrl.{key}.{subkey}"] = subval
            elem.clear()

        elif tag == "calibration":
            ds.calibration_info = dict(elem.attrib)
            for prm in elem.findall("prm_set"):
                key = prm.attrib.get("set", "")
                ds.calibration_prm_sets[key] = (_text(prm)).split(";") if _text(prm) else []

            # In this SPR2 generation, P1500 contains the eight laser wavelength
            # slots at indices 15..22 in hardware order [L5..L8,L1..L4].
            # Rotate them back to L1..L8.  The value 111 is an unused-slot marker.
            p1500 = ds.calibration_prm_sets.get("P1500", [])
            if len(p1500) >= 23:
                try:
                    hw = [int(float(v)) for v in p1500[15:23]]
                    ordered = hw[4:] + hw[:4]
                    for cid, wl in enumerate(ordered):
                        if 200 <= wl <= 2500:
                            ds.channel_wavelengths[cid] = wl
                except (TypeError, ValueError):
                    pass

            for ch in elem.findall("ch"):
                cid = int(ch.attrib.get("number", "-1"))
                data_elem = ch.find("data")
                if data_elem is not None:
                    ds.calibration_curves[cid] = np.fromstring(data_elem.text or "", sep=";", dtype=np.float32)
                info = {}
                fc = ch.find("fc")
                laser = ch.find("laser")
                if fc is not None:
                    info.update({f"fc.{x.tag}": _text(x) for x in fc})
                if laser is not None:
                    info.update({f"laser.{x.tag}": _text(x) for x in laser})
                ds.calibration_channel_info[cid] = info
            elem.clear()

        elif tag == "init_scan":
            ds.init_scan = _parse_scan_element(elem)
            elem.clear()

        elif tag == "scan":
            ds.scans.append(_parse_scan_element(elem))
            elem.clear()

        elif tag == "event":
            row: Dict[str, object] = {
                "type": elem.attrib.get("type", ""),
                "elapsed_s": float(elem.attrib.get("elapsed", "nan")),
                "timestamp": elem.attrib.get("ts", ""),
            }
            conc = elem.find("concentration")
            if conc is not None:
                row["concentration_value"] = _text(conc.find("value"))
                row["concentration_unit"] = _text(conc.find("unit"))
            row["sample"] = _text(elem.find("sample"))
            row["comment"] = _text(elem.find("comment"))
            row["flowulmin"] = _text(elem.find("flowulmin"))
            row["flowchannels"] = _text(elem.find("flowchannels"))
            ds.events.append(row)
            elem.clear()

    return ds


def read_companion_bin(path: str | Path) -> pd.DataFrame:
    """Decode the companion processed-data binary seen in SPR Navi exports.

    Structure reverse-engineered from the supplied example:
      uint32 BE: number of scans
      repeated scan records:
        float64 BE: elapsed time in minutes
        uint32 BE: number of active channels
        per channel:
          13 x float32 BE processed metrics
          uint16 BE channel id
          5 x float32 BE auxiliary values
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 4:
        raise ValueError("Binary file is too short.")

    n_scans = struct.unpack_from(">I", data, 0)[0]
    offset = 4
    rows = []

    for scan_index in range(n_scans):
        if offset + 12 > len(data):
            raise ValueError(f"Unexpected end of binary file before scan {scan_index}.")
        time_min = struct.unpack_from(">d", data, offset)[0]
        offset += 8
        n_channels = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        for _ in range(n_channels):
            if offset + 74 > len(data):
                raise ValueError(f"Unexpected end of binary file in scan {scan_index}.")
            metrics = struct.unpack_from(">13f", data, offset)
            offset += 52
            channel_id = struct.unpack_from(">H", data, offset)[0]
            offset += 2
            aux = struct.unpack_from(">5f", data, offset)
            offset += 20

            row = {
                "scan_index": scan_index,
                "time_min": time_min,
                "channel_id": channel_id,
            }
            # Map only quantities whose vendor names are supported by the user's
            # DataViewer CSV. metrics[4] equals PeakMinAngle - TirAngle numerically,
            # but no vendor field name for that stored quantity is present in the
            # reference export, so it is deliberately not exposed. metrics[12] and
            # aux[1:4] are likewise hidden rather than receiving metric_XX names.
            row["PeakMinAngle"] = metrics[0]
            row["PeakMinIntensity"] = metrics[1]
            row["TirAngle"] = metrics[2]
            row["TirIntensity"] = metrics[3]
            row["PeakAmplitude"] = metrics[5]
            row["PW_0.25_Intensity"] = metrics[6]
            row["PW_0.25Height"] = metrics[7]
            row["SteepestFallAngle"] = metrics[8]
            row["SteepestFallIntensity"] = metrics[9]
            row["Centroid"] = metrics[10]
            row["Weighted Centroid"] = metrics[11]
            row["Flow"] = aux[0]
            row["SensorTemp"] = aux[4]
            rows.append(row)

    if offset != len(data):
        # Some versions may append extra bytes. Keep parsing result but expose count.
        trailing = len(data) - offset
    else:
        trailing = 0

    df = pd.DataFrame(rows)
    df.attrs["declared_scans"] = n_scans
    df.attrs["trailing_bytes"] = trailing
    return df


def attach_binary(ds: SPRDataset, path: str | Path) -> None:
    path = Path(path)
    table = read_companion_bin(path)
    ds.binary_path = path
    ds.binary_table = table
    ds.angle_calibration = estimate_angle_calibration(ds)
    ds._refresh_pure_kinetics_column()


def auto_find_companion_bin(spr2_path: str | Path) -> Optional[Path]:
    p = Path(spr2_path)
    candidates = sorted(p.parent.glob(p.name + ".*.bin"))
    if candidates:
        return candidates[0]
    # Renamed experiments retain their acquisition timestamp in both filenames.
    # Only attach a uniquely matching timestamp AND sample suffix.
    parts = p.stem.split('_')
    if len(parts) >= 3:
        candidates = [q for q in p.parent.glob(parts[0] + '_*.bin')
                      if q.stem.lower().endswith('_' + '_'.join(parts[2:]).lower())]
        if len(candidates) == 1:
            return candidates[0]
    return None


def estimate_angle_calibration(ds: SPRDataset) -> Dict[int, Tuple[float, float, float]]:
    """Fit angle = slope * motor_position + intercept for each channel.

    Uses the raw curve minimum as a motor-coordinate anchor and PeakMinAngle as
    the processed resonance-angle anchor. Returned tuple is (slope, intercept, RMSE).
    This is an empirical calibration, not a claim about the vendor's internal formula.
    """
    if ds.binary_table is None or len(ds.scans) == 0:
        return {}

    anchors: Dict[int, List[Tuple[float, float]]] = {}
    bt = ds.binary_table.set_index(["scan_index", "channel_id"])

    for i, scan in enumerate(ds.scans):
        for cid, y in scan.channels.items():
            key = (i, cid)
            if key not in bt.index or len(y) == 0:
                continue
            theta = float(bt.loc[key, "PeakMinAngle"])
            if not np.isfinite(theta):
                continue
            k = int(np.nanargmin(y))
            sign = -1.0 if scan.direction.lower().startswith("back") else 1.0
            motor = float(scan.start_positions[cid]) + sign * float(scan.step_len) * k
            anchors.setdefault(cid, []).append((motor, theta))

    result = {}
    for cid, pairs in anchors.items():
        arr = np.asarray(pairs, dtype=float)
        if len(arr) < 2 or np.ptp(arr[:, 0]) == 0:
            continue
        slope, intercept = np.polyfit(arr[:, 0], arr[:, 1], 1)
        pred = slope * arr[:, 0] + intercept
        rmse = float(np.sqrt(np.mean((arr[:, 1] - pred) ** 2)))
        result[cid] = (float(slope), float(intercept), rmse)
    return result


def metadata_dataframe(ds: SPRDataset) -> pd.DataFrame:
    rows = []
    for key, value in ds.header.items():
        rows.append(("header", key, value))
    for key, value in ds.calibration_info.items():
        rows.append(("calibration", key, value))
    rows.append(("dataset", "spr2_path", str(ds.spr2_path)))
    rows.append(("dataset", "scan_count", len(ds.scans)))
    rows.append(("dataset", "channels", ", ".join(map(str, ds.channel_ids))))
    if ds.binary_path:
        rows.append(("dataset", "binary_path", str(ds.binary_path)))
    for cid, (slope, intercept, rmse) in ds.angle_calibration.items():
        rows.append(("angle_calibration", f"channel_{cid}", f"theta={slope:.10g}*motor+{intercept:.10g}; RMSE={rmse:.6g} deg"))
    for cid in sorted(ds.bulk_sensitivity):
        rows.append(("bulk_correction", ds.channel_label(cid), ds.bulk_sensitivity[cid]))
    if ds.bulk_sensitivity_source:
        rows.append(("bulk_correction", "sensitivity_source", ds.bulk_sensitivity_source))
    rows.append(("bulk_correction", "formula", "PureKinetics = DeltaPeakMinAngle - S_bulk * DeltaTirAngle"))
    rows.append(("bulk_correction", "reference_source", ds.bulk_reference_source))
    for cid in sorted(ds.bulk_reference):
        ref = ds.bulk_reference[cid]
        rows.append(("bulk_reference", ds.channel_label(cid),
                     f"Peak0={ref.get('PeakMinAngle', float('nan')):.10g}; "
                     f"TIR0={ref.get('TirAngle', float('nan')):.10g}"))
    return pd.DataFrame(rows, columns=["section", "field", "value"])


def events_dataframe(ds: SPRDataset) -> pd.DataFrame:
    return pd.DataFrame(ds.events)


def current_scan_dataframe(ds: SPRDataset, scan_index: int, x_mode: str = "angle") -> pd.DataFrame:
    scan = ds.scans[scan_index]
    frames = []
    for cid in sorted(scan.channels):
        y = scan.channels[cid]
        x, xlabel = ds.x_axis(scan_index, cid, mode=x_mode)
        frames.append(pd.DataFrame({
            "scan_index": scan_index,
            "time_min": scan.rtime_ms / 60000.0,
            "channel_id": cid,
            "x": x,
            "x_label": xlabel,
            "signal": y,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def _fmt_bionavis_number(value: object) -> str:
    """Format one number like the supplied Navi DataViewer CSV."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.6f}".replace(".", ",")


def bionavis_series(ds: SPRDataset) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Build x/y sensorgram series in the same column order as the supplied CSV.

    The BioNavis reference export contains six traces for most parameters and four
    FixedAngleIntensity traces.  FixedAngleIntensity is not stored in the supplied
    companion BIN, so those four y-columns are left blank instead of inventing data.
    """
    if ds.binary_table is None or ds.binary_table.empty:
        return []

    bt = ds.binary_table
    active = [int(x) for x in sorted(bt["channel_id"].unique())]
    time_by_channel = {
        cid: bt.loc[bt["channel_id"] == cid].sort_values("scan_index")["time_min"].to_numpy(dtype=float)
        for cid in active
    }

    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    # Exact parameter order seen in the user's DataViewer CSV.
    # First four stored fields.
    for field in ["PeakMinAngle", "PeakMinIntensity", "TirAngle", "TirIntensity"]:
        for cid in active:
            g = bt.loc[bt["channel_id"] == cid].sort_values("scan_index")
            out.append((f"{field} {ds.channel_label(cid)}", time_by_channel[cid], g[field].to_numpy(dtype=float)))

    # PureKinetics is derived from angle variations relative to a reference:
    #     DeltaPure = DeltaPeakMinAngle - S_bulk * DeltaTirAngle
    # The supplied BioNavis header reserves all active channels.  When a channel
    # has no verified S_bulk value, its y column is kept blank rather than guessed.
    ds._refresh_pure_kinetics_column()
    for cid in active:
        g = bt.loc[bt["channel_id"] == cid].sort_values("scan_index")
        t = time_by_channel[cid]
        if cid in ds.bulk_sensitivity:
            y = g["PureKinetics"].to_numpy(dtype=float)
        else:
            y = np.full_like(t, np.nan, dtype=float)
        out.append((f"PureKinetics {ds.channel_label(cid)}", t, y))

    for field in [
        "PeakAmplitude", "PW_0.25_Intensity", "PW_0.25Height",
        "SteepestFallAngle", "SteepestFallIntensity", "Centroid",
        "Weighted Centroid", "Flow", "SensorTemp",
    ]:
        for cid in active:
            g = bt.loc[bt["channel_id"] == cid].sort_values("scan_index")
            out.append((f"{field} {ds.channel_label(cid)}", time_by_channel[cid], g[field].to_numpy(dtype=float)))

    # Match the supplied DataViewer CSV header exactly: FixedAngleIntensity is
    # exported there for L1, L2, L3 and L5. The fixed-angle selection itself is
    # a DataViewer/UI setting not recovered from these two source files, so the
    # y values remain blank instead of being guessed.
    for cid in active[:4]:
        t = time_by_channel[cid]
        out.append((f"FixedAngleIntensity {ds.channel_label(cid)}", t, np.full_like(t, np.nan, dtype=float)))

    return out

def infer_bulk_sensitivities_from_bionavis_csv(ds: SPRDataset, csv_path: str | Path) -> Dict[int, float]:
    """Infer S_bulk from VARIATIONS in a BioNavis CSV containing PureKinetics.

    The fitted relationship is
        DeltaPureKinetics = DeltaPeakMinAngle - S_bulk * DeltaTirAngle

    The vendor PureKinetics trace is used only to infer S_bulk from its variations.
    Its absolute/initial PureKinetics value is NOT retained. In this reader,
    PureKinetics is defined as the bulk-corrected variation and is zero at the
    selected reference state.
    """
    import csv
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter=";")
        try:
            header1 = next(r)
            _header2 = next(r)
        except StopIteration:
            raise ValueError("BioNavis CSV must contain two header rows.")
        rows = list(r)

    result: Dict[int, float] = {}
    inferred_refs: Dict[int, Dict[str, float]] = {}
    for cid in ds.channel_ids:
        label = ds.channel_label(cid)
        names = [f"PeakMinAngle {label}", f"TirAngle {label}", f"PureKinetics {label}"]
        try:
            idx = [header1.index(name) for name in names]
        except ValueError:
            continue

        pma, tir, pure = [], [], []
        for row in rows:
            vals = []
            ok = True
            for i in idx:
                if i + 1 >= len(row) or not row[i + 1].strip():
                    ok = False
                    break
                try:
                    vals.append(float(row[i + 1].replace(",", ".")))
                except ValueError:
                    ok = False
                    break
            if ok:
                pma.append(vals[0]); tir.append(vals[1]); pure.append(vals[2])

        if len(pma) < 3:
            continue
        pma = np.asarray(pma, dtype=float)
        tir = np.asarray(tir, dtype=float)
        pure = np.asarray(pure, dtype=float)
        dp = pma - pma[0]
        dt = tir - tir[0]
        du = pure - pure[0]

        # DeltaPeak - DeltaPure = S_bulk * DeltaTIR
        rhs = dp - du
        denom = float(np.dot(dt, dt))
        if denom <= 0:
            continue
        s_bulk = float(np.dot(dt, rhs) / denom)
        pred = dp - s_bulk * dt
        rmse = float(np.sqrt(np.mean((pred - du) ** 2)))
        scale = max(float(np.nanstd(du)), 1e-6)
        if np.isfinite(s_bulk) and 0.1 < s_bulk < 5.0 and rmse < max(0.02, 0.25 * scale):
            result[cid] = s_bulk
            # Apply the inferred coefficient to this dataset's processed angles.
            # The Peak/TIR reference must therefore come from the same data source.
            # The vendor PureKinetics absolute value is intentionally ignored.
            if ds.binary_table is not None:
                dg = ds.binary_table[ds.binary_table["channel_id"] == int(cid)].sort_values("scan_index")
            else:
                dg = pd.DataFrame()
            if not dg.empty:
                inferred_refs[cid] = {
                    "PeakMinAngle": float(dg.iloc[0]["PeakMinAngle"]),
                    "TirAngle": float(dg.iloc[0]["TirAngle"]),
                }
            else:
                inferred_refs[cid] = {
                    "PeakMinAngle": float(pma[0]),
                    "TirAngle": float(tir[0]),
                }

    if result:
        ds.bulk_sensitivity.update(result)
        ds.bulk_sensitivity_source = f"variation fit from {csv_path.name}"
        # Reference angles come from this dataset. The vendor PureKinetics level
        # is never copied into the result.
        ds.bulk_reference.update(inferred_refs)
        ds.bulk_reference_source = f"first processed scan (S_bulk inferred from {csv_path.name})"
        ds._refresh_pure_kinetics_column()
    return result

def load_bulk_sensitivity_profile(ds: SPRDataset, profile_path: str | Path) -> Dict[int, float]:
    """Load a JSON mapping of channel id/label to bulk sensitivity."""
    profile_path = Path(profile_path)
    obj = json.loads(profile_path.read_text(encoding="utf-8"))
    raw = obj.get("bulk_sensitivity", obj)
    result: Dict[int, float] = {}
    for key, value in raw.items():
        cid = None
        try:
            cid = int(key)
        except (TypeError, ValueError):
            for c in ds.channel_ids:
                if str(key).strip() == ds.channel_label(c):
                    cid = c
                    break
        if cid is None or cid not in ds.channel_ids:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            result[cid] = v
    ds.bulk_sensitivity.update(result)
    # A sensitivity profile is reusable across experiments, but Peak/TIR reference
    # angles are measurement-specific. Never import baseline angles from a profile.
    ds.bulk_reference = {}
    ds.bulk_reference_source = "first processed scan"
    ds.bulk_sensitivity_source = f"loaded from {profile_path.name}"
    ds._refresh_pure_kinetics_column()
    return result


def save_bulk_sensitivity_profile(ds: SPRDataset, profile_path: str | Path) -> None:
    profile_path = Path(profile_path)
    obj = {
        "formula": "PureKinetics = DeltaPeakMinAngle - BulkSensitivity * DeltaTirAngle",
        "device_serial": ds.header.get("device_serial", ""),
        "bulk_sensitivity": {str(cid): float(v) for cid, v in sorted(ds.bulk_sensitivity.items())},
        "channel_labels": {str(cid): ds.channel_label(cid) for cid in ds.channel_ids},
        "note": "Reference PeakMin/TIR angles are experiment-specific and are intentionally not stored in a reusable sensitivity profile.",
    }
    profile_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def export_bionavis_csv(ds: SPRDataset, output: str | Path) -> None:
    """Write a semicolon/decimal-comma, two-header-row Navi-style sensorgram CSV."""
    series = bionavis_series(ds)
    if not series:
        raise ValueError("The companion BIN is required for BioNavis-style sensorgram export.")

    max_len = max(len(y) for _, _x, y in series)
    output = Path(output)
    with output.open("w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f, delimiter=";", lineterminator="\r\n")
        header1: List[str] = []
        header2: List[str] = []
        for name, _x, _y in series:
            header1.extend([name, ""])
            header2.extend(["x", "y"])
        w.writerow(header1)
        w.writerow(header2)
        for i in range(max_len):
            row: List[str] = []
            for _name, x, y in series:
                row.append(_fmt_bionavis_number(x[i]) if i < len(x) else "")
                row.append(_fmt_bionavis_number(y[i]) if i < len(y) else "")
            w.writerow(row)


def export_all_scans_csv(ds: SPRDataset, output_dir: str | Path, x_mode: str = "angle") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dataframe(ds).to_csv(output_dir / "metadata.csv", index=False)
    events_dataframe(ds).to_csv(output_dir / "events.csv", index=False)
    if ds.binary_table is not None:
        ds.binary_table.to_csv(output_dir / "processed_sensorgram_long.csv", index=False)
        export_bionavis_csv(ds, output_dir / "sensorgram_bionavis_style.csv")
    scans_dir = output_dir / "angular_scans"
    scans_dir.mkdir(exist_ok=True)
    for i in range(len(ds.scans)):
        current_scan_dataframe(ds, i, x_mode=x_mode).to_csv(scans_dir / f"scan_{i:04d}.csv", index=False)


def export_excel(ds: SPRDataset, output: str | Path, current_scan: Optional[int] = None) -> None:
    output = Path(output)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        metadata_dataframe(ds).to_excel(writer, sheet_name="Metadata", index=False)
        events_dataframe(ds).to_excel(writer, sheet_name="Events", index=False)
        if ds.binary_table is not None:
            ds.binary_table.to_excel(writer, sheet_name="Processed_Sensorgram", index=False)
        if current_scan is not None and ds.scans:
            current_scan_dataframe(ds, current_scan).to_excel(writer, sheet_name="Current_Angular_Scan", index=False)


def export_json(ds: SPRDataset, output: str | Path) -> None:
    obj = {
        "header": ds.header,
        "calibration_info": ds.calibration_info,
        "events": ds.events,
        "scan_count": len(ds.scans),
        "channel_ids": ds.channel_ids,
        "angle_calibration": {
            str(cid): {"slope": a, "intercept": b, "rmse_deg": r}
            for cid, (a, b, r) in ds.angle_calibration.items()
        },
        "processed_binary": None if ds.binary_table is None else ds.binary_table.to_dict(orient="records"),
    }
    Path(output).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def export_npz(ds: SPRDataset, output: str | Path) -> None:
    payload = {}
    for cid in ds.channel_ids:
        curves = [s.channels[cid] for s in ds.scans if cid in s.channels]
        if curves and len({len(x) for x in curves}) == 1:
            payload[f"channel_{cid}_curves"] = np.stack(curves).astype(np.float32)
        else:
            payload[f"channel_{cid}_curves"] = np.array(curves, dtype=object)
    payload["scan_rtime_ms"] = np.array([s.rtime_ms for s in ds.scans], dtype=float)
    payload["channel_ids"] = np.array(ds.channel_ids, dtype=int)
    np.savez_compressed(output, **payload)


def export_mat(ds: SPRDataset, output: str | Path) -> None:
    from scipy.io import savemat
    data = {
        "scan_rtime_ms": np.array([s.rtime_ms for s in ds.scans], dtype=float),
        "channel_ids": np.array(ds.channel_ids, dtype=int),
    }
    for cid in ds.channel_ids:
        curves = [s.channels[cid] for s in ds.scans if cid in s.channels]
        if curves and len({len(x) for x in curves}) == 1:
            data[f"channel_{cid}_curves"] = np.stack(curves).astype(np.float32)
    if ds.binary_table is not None:
        for col in ds.binary_table.select_dtypes(include=[np.number]).columns:
            data[f"processed_{col}"] = ds.binary_table[col].to_numpy()
    savemat(output, data, do_compression=True)
