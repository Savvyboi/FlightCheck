"""Thread-safe per-scene diagnostic cache, internal mutation guard, and debounced rescan."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class Severity(str, Enum):
    OK = "OK"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Category(str, Enum):
    ALBEDO = "ALBEDO"
    ROUGHNESS = "ROUGHNESS"
    COLORSPACE = "COLORSPACE"
    TEXEL = "TEXEL"
    SCALE = "SCALE"


class FlightStatus(str, Enum):
    GO = "GO"
    CAUTION = "CAUTION"
    NO_GO = "NO-GO"


@dataclass(frozen=True)
class DiagnosticIssue:
    code: str
    category: Category
    severity: Severity
    message: str
    object_name: str = ""
    material_name: str = ""
    image_name: str = ""
    measured_value: Optional[float] = None
    expected_range: str = ""
    fixable: bool = True
    is_shared_image: bool = False


@dataclass
class ObjectScaleInfo:
    object_name: str
    scale: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    determinant: float
    is_uniform: bool
    is_applied: bool
    is_inverted: bool
    has_dimension_outlier: bool
    severity: Severity = Severity.OK


@dataclass
class TexelObjectStats:
    object_name: str
    avg_density_px_per_m: float
    min_density_px_per_m: float
    max_density_px_per_m: float
    target_px_per_m: float
    face_count: int
    out_of_spec_faces: int
    has_uvs: bool
    severity: Severity = Severity.OK


@dataclass
class DiagnosticReport:
    timestamp: float = field(default_factory=time.time)
    issues: List[DiagnosticIssue] = field(default_factory=list)
    scale_info: Dict[str, ObjectScaleInfo] = field(default_factory=dict)
    texel_stats: Dict[str, TexelObjectStats] = field(default_factory=dict)
    scanned_objects: int = 0
    scanned_materials: int = 0
    scanned_images: int = 0
    scan_duration_ms: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == Severity.INFO)

    @property
    def status(self) -> FlightStatus:
        if self.critical_count > 0:
            return FlightStatus.NO_GO
        if self.warning_count > 0:
            return FlightStatus.CAUTION
        return FlightStatus.GO

    def issues_by_category(self, category: Category) -> List[DiagnosticIssue]:
        return [issue for issue in self.issues if issue.category == category]

    def category_status(self, category: Category) -> Severity:
        cat_issues = self.issues_by_category(category)
        if any(issue.severity == Severity.CRITICAL for issue in cat_issues):
            return Severity.CRITICAL
        if any(issue.severity == Severity.WARNING for issue in cat_issues):
            return Severity.WARNING
        return Severity.OK

    def gauge_scores(self) -> Dict[str, Tuple[Severity, float]]:
        """Compute compliance percentages (0..100) and severities for HUD gauges."""
        mat_denom = max(self.scanned_materials, 1)
        obj_denom = max(self.scanned_objects, 1)

        def _issue_penalty(issue: DiagnosticIssue) -> float:
            if issue.severity == Severity.CRITICAL:
                return 35.0
            if issue.severity == Severity.WARNING:
                return 15.0
            return 0.0

        def _score_for(cat: Category, denom: int) -> Tuple[Severity, float]:
            cat_issues = self.issues_by_category(cat)
            if not cat_issues:
                return (Severity.OK, 100.0)
            sev = self.category_status(cat)
            penalty = sum(_issue_penalty(i) for i in cat_issues)
            score = max(0.0, 100.0 - (penalty / max(denom, 1)))
            return (sev, score)

        alb_sev, alb_score = _score_for(Category.ALBEDO, mat_denom)
        cs_issues = self.issues_by_category(Category.COLORSPACE)
        if cs_issues:
            cs_sev = self.category_status(Category.COLORSPACE)
            if cs_sev == Severity.CRITICAL or (cs_sev == Severity.WARNING and alb_sev == Severity.OK):
                alb_sev = cs_sev
            cs_penalty = sum(_issue_penalty(i) for i in cs_issues)
            alb_score = max(0.0, alb_score - (cs_penalty / max(mat_denom, 1)))

        rgh_sev, rgh_score = _score_for(Category.ROUGHNESS, mat_denom)
        scl_sev, scl_score = _score_for(Category.SCALE, obj_denom)

        if self.texel_stats:
            total_faces = sum(s.face_count for s in self.texel_stats.values())
            bad_faces = sum(s.out_of_spec_faces for s in self.texel_stats.values())
            txl_sev = self.category_status(Category.TEXEL)
            if total_faces > 0:
                txl_score = max(0.0, 100.0 * (1.0 - (bad_faces / total_faces)))
            else:
                txl_score = 100.0 if txl_sev == Severity.OK else 0.0
        else:
            txl_sev, txl_score = _score_for(Category.TEXEL, obj_denom)

        return {
            "ALB": (alb_sev, alb_score),
            "RGH": (rgh_sev, rgh_score),
            "TXL": (txl_sev, txl_score),
            "SCL": (scl_sev, scl_score),
        }


def _resolve_scene_key(scene: Any = None) -> int:
    """Return a unique integer key for ``scene`` (using context.scene when omitted)."""
    target = scene
    if target is None:
        try:
            import bpy

            ctx = getattr(bpy, "context", None)
            target = getattr(ctx, "scene", None) if ctx is not None else None
        except Exception:
            target = None
    if target is None:
        return 0
    as_ptr = getattr(target, "as_pointer", None)
    if callable(as_ptr):
        try:
            return int(as_ptr())
        except Exception:
            pass
    return id(target)


class DiagnosticCache:
    """Thread-safe per-scene container storing diagnostic reports and HUD draw timing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reports: Dict[int, DiagnosticReport] = {}
        self._dirty: Dict[int, bool] = {}
        self._last_scene_key: int = 0
        self._draw_time_ms: float = 0.0

    def _key_or_last(self, scene: Any = None) -> int:
        if scene is not None:
            return _resolve_scene_key(scene)
        key = _resolve_scene_key(None)
        if key in self._reports or key in self._dirty:
            return key
        return self._last_scene_key if self._last_scene_key != 0 else key

    def get_report(self, scene: Any = None) -> Optional[DiagnosticReport]:
        with self._lock:
            key = self._key_or_last(scene)
            return self._reports.get(key)

    def set_report(self, report: DiagnosticReport, scene: Any = None) -> None:
        with self._lock:
            key = _resolve_scene_key(scene)
            self._reports[key] = report
            self._dirty[key] = False
            self._last_scene_key = key

    def clear(self, scene: Any = None) -> None:
        with self._lock:
            if scene is None:
                self._reports.clear()
                self._dirty.clear()
                self._last_scene_key = 0
                self._draw_time_ms = 0.0
            else:
                key = _resolve_scene_key(scene)
                self._reports.pop(key, None)
                self._dirty[key] = True

    def mark_dirty(self, scene: Any = None) -> None:
        with self._lock:
            if scene is None:
                key = _resolve_scene_key(None)
                if key:
                    self._dirty[key] = True
                for existing_key in list(self._reports.keys()):
                    self._dirty[existing_key] = True
                self._dirty[self._last_scene_key] = True
            else:
                key = _resolve_scene_key(scene)
                self._dirty[key] = True
                self._last_scene_key = key

    def is_dirty(self, scene: Any = None) -> bool:
        with self._lock:
            key = self._key_or_last(scene)
            return self._dirty.get(key, True)

    def set_draw_time_ms(self, ms: float) -> None:
        with self._lock:
            self._draw_time_ms = float(ms)

    def get_draw_time_ms(self) -> float:
        with self._lock:
            return self._draw_time_ms


CACHE = DiagnosticCache()

_MUTATION_LOCK = threading.RLock()
_INTERNAL_MUTATION_DEPTH = 0

_RESCAN_CALLBACK: Optional[Callable[[Any], None]] = None
_PENDING_SCENE_PTR: Optional[int] = None


def is_internal_mutation() -> bool:
    """Return True while FlightCheck itself is writing attributes or splicing shaders."""
    return _INTERNAL_MUTATION_DEPTH > 0


@contextmanager
def internal_mutation() -> Iterator[None]:
    """Suppress depsgraph dirty/rescan recursion around FlightCheck's own writes."""
    global _INTERNAL_MUTATION_DEPTH
    with _MUTATION_LOCK:
        _INTERNAL_MUTATION_DEPTH += 1
    try:
        yield
    finally:
        with _MUTATION_LOCK:
            _INTERNAL_MUTATION_DEPTH = max(0, _INTERNAL_MUTATION_DEPTH - 1)


def register_rescan_callback(callback: Optional[Callable[[Any], None]]) -> None:
    """Register the scene resync/rescan callback triggered by debounced timers."""
    global _RESCAN_CALLBACK
    _RESCAN_CALLBACK = callback


def _find_scene_by_ptr(scene_ptr: Optional[int]) -> Any:
    try:
        import bpy

        if scene_ptr is not None:
            for sc in bpy.data.scenes:
                if sc.as_pointer() == scene_ptr:
                    return sc
        ctx = getattr(bpy, "context", None)
        return getattr(ctx, "scene", None) if ctx is not None else None
    except Exception:
        return None


def _timer_rescan_entry() -> Optional[float]:
    global _PENDING_SCENE_PTR
    target_ptr = _PENDING_SCENE_PTR
    _PENDING_SCENE_PTR = None
    scene = _find_scene_by_ptr(target_ptr)
    if scene is not None and _RESCAN_CALLBACK is not None:
        _RESCAN_CALLBACK(scene)
    return None


def cancel_pending_rescan() -> None:
    """Cancel any queued debounced rescan timer."""
    global _PENDING_SCENE_PTR
    _PENDING_SCENE_PTR = None
    try:
        import bpy

        timers = getattr(bpy.app, "timers", None)
        if timers is not None and timers.is_registered(_timer_rescan_entry):
            timers.unregister(_timer_rescan_entry)
    except Exception:
        pass


def schedule_rescan(scene: Any = None, delay: float = 0.35) -> None:
    """Schedule a debounced rescan + overlay resync for ``scene``."""
    global _PENDING_SCENE_PTR
    _PENDING_SCENE_PTR = _resolve_scene_key(scene) if scene is not None else _resolve_scene_key(None)
    try:
        import bpy

        timers = getattr(bpy.app, "timers", None)
        if timers is None:
            return
        if timers.is_registered(_timer_rescan_entry):
            timers.unregister(_timer_rescan_entry)
        timers.register(_timer_rescan_entry, first_interval=max(0.05, float(delay)), persistent=True)
    except Exception:
        pass


def flush_pending_rescan(scene: Any = None) -> None:
    """Execute any pending debounced rescan immediately (useful in synchronous tests)."""
    target_ptr = _PENDING_SCENE_PTR
    cancel_pending_rescan()
    target_scene = scene if scene is not None else _find_scene_by_ptr(target_ptr)
    if target_scene is not None and _RESCAN_CALLBACK is not None:
        _RESCAN_CALLBACK(target_scene)
