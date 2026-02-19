from definable.vectordb.redis.redisdb import RedisDB
from definable.vectordb.search import SearchType

# Backward compatibility alias
RedisVectorDb = RedisDB

__all__ = [
  "RedisVectorDb",
  "RedisDB",
  "SearchType",
]
