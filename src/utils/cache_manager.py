"""
CacheManager: cache de cohérence avec TTL pour AIPROD
"""
import time
from typing import Any, Dict, Optional

class CacheManager:
	"""
	Gère un cache clé/valeur avec expiration TTL (par défaut 168h).
	"""
	def __init__(self, ttl_hours: int = 168):
		self._cache: Dict[str, Any] = {}
		self._expiry: Dict[str, float] = {}
		self.ttl = ttl_hours * 3600  # secondes

	def set(self, key: str, value: Any) -> None:
		self._cache[key] = value
		self._expiry[key] = time.time() + self.ttl

	def get(self, key: str) -> Optional[Any]:
		if key in self._cache:
			if time.time() < self._expiry.get(key, 0):
				return self._cache[key]
			else:
				self.delete(key)
		return None

	def delete(self, key: str) -> None:
		self._cache.pop(key, None)
		self._expiry.pop(key, None)

	def clear(self) -> None:
		self._cache.clear()
		self._expiry.clear()

	def keys(self):
		return [k for k in self._cache if time.time() < self._expiry.get(k, 0)]
