"""Frozen dataclass configuration. ALL tunables live here (§3)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env.local", override=True)
except ImportError:
    pass


@dataclass(frozen=True)
class PomConfig:
    # Environment & Chains
    chain: str = field(default_factory=lambda: os.environ.get("POM_CHAIN", "amoy"))
    rpc_amoy: str = field(default_factory=lambda: os.environ.get("POM_RPC_AMOY", "https://rpc-amoy.polygon.technology"))
    rpc_base_sepolia: str = field(default_factory=lambda: os.environ.get("POM_RPC_BASE_SEPOLIA", "https://sepolia.base.org"))
    contract_amoy: str = field(default_factory=lambda: os.environ.get("POM_CONTRACT_AMOY", "0x0000000000000000000000000000000000000000"))
    contract_base_sepolia: str = field(default_factory=lambda: os.environ.get("POM_CONTRACT_BASE_SEPOLIA", "0x0000000000000000000000000000000000000000"))
    private_key: str = field(default_factory=lambda: os.environ.get("POM_PRIVATE_KEY", ""))
    imagehost: str = field(default_factory=lambda: os.environ.get("POM_IMAGEHOST", "pinata"))
    
    # API Keys
    serpapi_key: str = field(default_factory=lambda: os.environ.get("SERPAPI_API_KEY", ""))
    pinata_jwt: str = field(default_factory=lambda: os.environ.get("PINATA_JWT", ""))
    gcv_credentials: str = field(default_factory=lambda: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    
    # Thresholds & Biometric Tunables (§14 Measured Calibration)
    arcface_match_threshold: float = 0.40
    adaface_match_threshold: float = 0.30
    min_face_size_px: int = 40
    blur_min_variance: float = 50.0
    max_roll_deg: float = 20.0
    max_yaw_score: float = 0.30
    max_pitch_score: float = 0.55
    occlusion_min_crop_std: float = 10.0
    
    # Search & Network Limits
    search_timeout_seconds: float = 15.0
    hop2_max_seed_pages: int = 8
    hop2_max_queries: int = 6
    max_image_download_bytes: int = 15 * 1024 * 1024  # 15 MB
    max_upload_bytes: int = 500_000                  # 500 KB
    
    # Directories
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    out_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "out")
    probes_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "probes")
    contracts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "contracts")

    def get_active_rpc(self) -> str:
        if self.chain == "base_sepolia":
            return self.rpc_base_sepolia
        return self.rpc_amoy

    def get_active_contract(self) -> str:
        if self.chain == "base_sepolia":
            return self.contract_base_sepolia
        return self.contract_amoy


CONFIG = PomConfig()
