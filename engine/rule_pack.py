"""
PatientTriage.ai — Rule Pack Loader
===================================

Loads the versioned clinical safety policy from YAML and makes it queryable.

The split this enforces: rule *logic* is Python (testable, reviewable in a
pull request), rule *policy* is YAML (readable and changeable by the clinical
governance committee that actually owns the risk). A site that wants tighter
paediatric thresholds edits a file and bumps a version; nobody redeploys code
and nobody re-reads a diff full of unrelated changes to find the one number
that moved.

Every decision the safety engine makes records the pack id and version that
produced it, so an audit two years later can reconstruct exactly which policy
was in force when a given patient was triaged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Dict, List, Optional

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
DEFAULT_PACK = os.path.join(CONFIG_DIR, "rules_default.yaml")


class RulePack:
    """A loaded, validated clinical rule pack."""

    def __init__(self, data: Dict, source_path: Optional[str] = None):
        self.data = data
        self.source_path = source_path
        self._validate()

    # ── Loading ──
    @classmethod
    def load(cls, path: Optional[str] = None) -> "RulePack":
        path = path or DEFAULT_PACK
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data, source_path=path)

    @classmethod
    def load_site(cls, site: str) -> "RulePack":
        """
        Load a site pack, layered over the default.

        Site packs are deltas, not copies. A rural ED's file contains only what
        it changes, so reviewing a site's policy means reading ten lines rather
        than diffing three hundred — and an improvement to a shared rule
        reaches every site without being re-applied by hand.
        """
        base = cls.load(DEFAULT_PACK)
        site_path = os.path.join(CONFIG_DIR, f"rules_{site}.yaml")
        if not os.path.isfile(site_path):
            raise FileNotFoundError(f"No rule pack for site '{site}' at {site_path}")
        with open(site_path, "r", encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        merged = _deep_merge(copy.deepcopy(base.data), overlay)
        return cls(merged, source_path=site_path)

    # ── Validation ──
    def _validate(self):
        required = ["pack_id", "version", "thresholds", "rules"]
        missing = [k for k in required if k not in self.data]
        if missing:
            raise ValueError(f"Rule pack missing required keys: {missing}")

        for band in ("pediatric", "adult", "geriatric"):
            if band not in self.data["thresholds"]:
                raise ValueError(f"Rule pack missing threshold band: {band}")

        seen = set()
        for rule in self.data["rules"]:
            if "id" not in rule:
                raise ValueError(f"Rule without an id: {rule}")
            if rule["id"] in seen:
                raise ValueError(f"Duplicate rule id: {rule['id']}")
            seen.add(rule["id"])
            target = rule.get("escalate_to")
            # A rule that could lower urgency would silently invert the whole
            # safety model, so the loader refuses to accept one.
            if target is not None and not (1 <= int(target) <= 5):
                raise ValueError(
                    f"Rule {rule['id']} has escalate_to={target}, outside 1 to 5."
                )

    # ── Queries ──
    @property
    def pack_id(self) -> str:
        return self.data["pack_id"]

    @property
    def version(self) -> str:
        return str(self.data["version"])

    @property
    def jurisdiction(self) -> str:
        return self.data.get("jurisdiction", "unspecified")

    def content_hash(self) -> str:
        """
        Stable hash of the pack's content.

        Recorded alongside every triage decision. Version strings can be
        forgotten on edit; a content hash cannot, so this is what actually
        proves which policy text produced a given decision.
        """
        payload = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def thresholds_for(self, age_group: str) -> Dict:
        return self.data["thresholds"].get(age_group, self.data["thresholds"]["adult"])

    def rule(self, rule_id: str) -> Optional[Dict]:
        for r in self.data["rules"]:
            if r["id"] == rule_id:
                return r
        return None

    def is_enabled(self, rule_id: str) -> bool:
        r = self.rule(rule_id)
        return bool(r and r.get("enabled", True))

    def escalation_target(self, rule_id: str, default: Optional[int] = None) -> Optional[int]:
        r = self.rule(rule_id)
        if not r:
            return default
        target = r.get("escalate_to", default)
        return None if target is None else int(target)

    def lexicon(self, name: str) -> List[str]:
        return [str(t).lower() for t in self.data.get("lexicons", {}).get(name, [])]

    def reassessment_interval(self, level: int) -> int:
        intervals = self.data.get("reassessment_intervals", {})
        return int(intervals.get(level, intervals.get(str(level), 60)))

    def enabled_rules(self) -> List[Dict]:
        return [r for r in self.data["rules"] if r.get("enabled", True)]

    def provenance(self) -> Dict:
        """Stamped onto every decision this pack contributes to."""
        return {
            "pack_id": self.pack_id,
            "version": self.version,
            "content_hash": self.content_hash(),
            "jurisdiction": self.jurisdiction,
            "effective_date": str(self.data.get("effective_date", "")),
            "n_rules_enabled": len(self.enabled_rules()),
            "n_rules_total": len(self.data["rules"]),
            "source": os.path.basename(self.source_path or "inline"),
        }

    def summary_table(self) -> List[Dict]:
        """Rule inventory for the governance view in the UI."""
        return [
            {
                "id": r["id"],
                "enabled": r.get("enabled", True),
                "category": r.get("category", "unspecified"),
                "escalates_to": r.get("escalate_to"),
                "description": r.get("description", "").strip(),
                "rationale": " ".join(r.get("clinical_rationale", "").split()),
                "citation": r.get("citation", "not cited"),
            }
            for r in self.data["rules"]
        ]


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """
    Merge an overlay into a base pack.

    Rules merge by id rather than by list position, so a site overlay that
    disables one rule does not have to restate the other fourteen — and cannot
    accidentally drop them.
    """
    for key, value in overlay.items():
        if key == "rules" and isinstance(value, list):
            by_id = {r["id"]: r for r in base.get("rules", [])}
            for override in value:
                rid = override.get("id")
                if rid in by_id:
                    by_id[rid] = {**by_id[rid], **override}
                else:
                    by_id[rid] = override
            base["rules"] = list(by_id.values())
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def available_site_packs() -> List[str]:
    if not os.path.isdir(CONFIG_DIR):
        return []
    return sorted(
        f[len("rules_"):-len(".yaml")]
        for f in os.listdir(CONFIG_DIR)
        if f.startswith("rules_") and f.endswith(".yaml")
    )
