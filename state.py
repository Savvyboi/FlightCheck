"""Thread-safe diagnostic report dataclasses and global cache."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Severity(str, Enum):
    OK = "OK"
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

        def _score_for(cat: Category, denom: int) -> Tuple[Severity, float]:
            cat_issues = self.issues_by_category(cat)
            if not cat_issues:
                return (Severity.OK, 100.0)
            sev = self.category_status(cat)
            penalty = sum(35.0 if i.severity == Severity.CRITICAL else 15.0 for i in cat_issues)
            score = max(0.0, 100.0 - (penalty / max(denom, 1)))
            return (sev, score)

        alb_sev, alb_score = _score_for(Category.ALBEDO, mat_denom)
        cs_issues = self.issues_by_category(Category.COLORSPACE)
        if cs_issues and alb_sev == Severity.OK:
            alb_sev = self.category_status(Category.COLORSPACE)

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


class DiagnosticCache:
    """Thread-safe container storing the active diagnostic report and draw timing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._report: Optional[DiagnosticReport] = None
        self._dirty: bool = True
        self._draw_time_ms: float = 0.0

    def get_report(self) -> Optional[DiagnosticReport]:
        with self._lock:
            return self._report

    def set_report(self, report: DiagnosticReport) -> None:
        with self._lock:
            self._report = report
            self._dirty = False

    def clear(self) -> None:
        with self._lock:
            self._report = None
            self._dirty = True
            self._draw_time_ms = 0.0

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def set_draw_time_ms(self, ms: float) -> None:
        with self._lock:
            self._draw_time_ms = float(ms)

    def get_draw_time_ms(self) -> float:
        with self._lock:
            return self._draw_time_ms


CACHE = DiagnosticCache()
