"""Search type configuration for vector databases."""

from enum import Enum


class SearchType(str, Enum):
  vector = "vector"
  keyword = "keyword"
  hybrid = "hybrid"
