from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from config.k_factors import DEFAULT_K_FACTOR, get_k_factor  # noqa: E402


class KFactorTests(unittest.TestCase):
    def test_get_k_factor_returns_configured_tournament_weight(self):
        self.assertEqual(get_k_factor("FIFA World Cup"), 60)
        self.assertEqual(get_k_factor("FIFA World Cup qualification"), 40)
        self.assertEqual(get_k_factor("Friendly"), 20)

    def test_get_k_factor_handles_accented_tournament_names(self):
        self.assertEqual(get_k_factor("Copa America"), 50)
        self.assertEqual(get_k_factor("Copa Am\u00e9rica"), 50)

    def test_get_k_factor_uses_default_for_unknown_or_missing_tournament(self):
        self.assertEqual(get_k_factor("Made Up Cup"), DEFAULT_K_FACTOR)
        self.assertEqual(get_k_factor(None), DEFAULT_K_FACTOR)


if __name__ == "__main__":
    unittest.main()