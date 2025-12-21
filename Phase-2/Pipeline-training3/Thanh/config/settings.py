import json
import os
from pathlib import Path
from typing import Any, Dict

# 1. Định nghĩa thư mục gốc của Project
# File này nằm trong config/, nên parent của nó là root project
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Đường dẫn đến file params.json
CFG_PROFILE = os.getenv("CFG_PROFILE", "full").lower().strip()
# Supported: "smoke", "full"
if CFG_PROFILE not in {"smoke", "full"}:
    CFG_PROFILE = "full"
PARAMS_PATH = BASE_DIR / "config" / f"params_{CFG_PROFILE}.json"
# Backward compatibility: if profile file missing, fall back to params.json
if not PARAMS_PATH.exists():
    PARAMS_PATH = BASE_DIR / "config" / "params.json"

class Settings:
    """
    Class này load config từ JSON và cho phép truy cập attribute bằng dấu chấm.
    Ví dụ: settings.stage1.n_neighbors
    """
    def __init__(self, json_path: Path):
        self._config = self._load_json(json_path)
        
        # Tự động gán các keys trong json thành attributes của class
        for key, value in self._config.items():
            if isinstance(value, dict):
                # Nếu value là dict con, chuyển nó thành object để gọi dấu chấm được
                setattr(self, key, DictWrapper(value))
            else:
                setattr(self, key, value)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __repr__(self):
        return str(self._config)

class DictWrapper:
    """Helper class để truy cập dictionary bằng dấu chấm (.)"""
    def __init__(self, data):
        self.__dict__.update(data)
    
    def __getitem__(self, item):
        return self.__dict__[item]
    
    def get(self, item, default=None):
        return self.__dict__.get(item, default)

# Khởi tạo singleton instance để import ở nơi khác
try:
    CFG = Settings(PARAMS_PATH)
except Exception as e:
    print(f"Warning: Could not load settings. Error: {e}")
    CFG = None

# Helper để lấy đường dẫn tuyệt đối dựa trên project root
def get_path(relative_path: str) -> str:
    """
    Chuyển đường dẫn tương đối (vd: 'artifacts/model.pkl')
    thành đường dẫn tuyệt đối (vd: '/home/user/project/artifacts/model.pkl')
    """
    return str(BASE_DIR / relative_path)