from typing import Any, Dict, Optional

from data.data_object import DataObject


class AdultCFVAEData(DataObject):
    def __init__(
        self,
        data_path: str,
        config_path: str = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(data_path, config_path, config_override)
