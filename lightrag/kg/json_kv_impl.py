import os
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import (
    BaseKVStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
)
from .shared_storage import (
    get_namespace_data,
    get_storage_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)


@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

    async def initialize(self):
        """Initialize storage data"""
        self._storage_lock = get_storage_lock()
        self.storage_updated = await get_update_flag(self.namespace)
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(self.namespace)
            self._data = await get_namespace_data(self.namespace)
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    self._data.update(loaded_data)

                    # Calculate data count based on namespace
                    if self.namespace.endswith("cache"):
                        # For cache namespaces, sum the cache entries across all cache types
                        data_count = sum(
                            len(first_level_dict)
                            for first_level_dict in loaded_data.values()
                            if isinstance(first_level_dict, dict)
                        )
                    else:
                        # For non-cache namespaces, use the original count method
                        data_count = len(loaded_data)

                    logger.info(
                        f"Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )

    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )

                # Calculate data count based on namespace
                if self.namespace.endswith("cache"):
                    # # For cache namespaces, sum the cache entries across all cache types
                    data_count = sum(
                        len(first_level_dict)
                        for first_level_dict in data_dict.values()
                        if isinstance(first_level_dict, dict)
                    )
                else:
                    # For non-cache namespaces, use the original count method
                    data_count = len(data_dict)

                logger.debug(
                    f"Process {os.getpid()} KV writting {data_count} records to {self.namespace}"
                )
                write_json(data_dict, self._file_name)
                await clear_all_update_flags(self.namespace)

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """
        async with self._storage_lock:
            return dict(self._data)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            return self._data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            return [
                (
                    {k: v for k, v in self._data[id].items()}
                    if self._data.get(id, None)
                    else None
                )
                for id in ids
            ]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """
        if not data:
            return
        logger.debug(f"Inserting {len(data)} records to {self.namespace}")
        async with self._storage_lock:
            self._data.update(data)
            await set_all_update_flags(self.namespace)

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        async with self._storage_lock:
            any_deleted = False
            for doc_id in ids:
                result = self._data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace)

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Delete specific records from storage by by cache mode

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Args:
            ids (list[str]): List of cache mode to be drop from storage

        Returns:
             True: if the cache drop successfully
             False: if the cache drop failed
        """
        if not modes:
            return False

        try:
            await self.delete(modes)
            return True
        except Exception:
            return False

    # async def drop_cache_by_chunk_ids(self, chunk_ids: list[str] | None = None) -> bool:
    #     """Delete specific cache records from storage by chunk IDs

    #     Importance notes for in-memory storage:
    #     1. Changes will be persisted to disk during the next index_done_callback
    #     2. update flags to notify other processes that data persistence is needed

    #     Args:
    #         chunk_ids (list[str]): List of chunk IDs to be dropped from storage

    #     Returns:
    #          True: if the cache drop successfully
    #          False: if the cache drop failed
    #     """
    #     if not chunk_ids:
    #         return False

    #     try:
    #         async with self._storage_lock:
    #             # Iterate through all cache modes to find entries with matching chunk_ids
    #             for mode_key, mode_data in list(self._data.items()):
    #                 if isinstance(mode_data, dict):
    #                     # Check each cached entry in this mode
    #                     for cache_key, cache_entry in list(mode_data.items()):
    #                         if (
    #                             isinstance(cache_entry, dict)
    #                             and cache_entry.get("chunk_id") in chunk_ids
    #                         ):
    #                             # Remove this cache entry
    #                             del mode_data[cache_key]
    #                             logger.debug(
    #                                 f"Removed cache entry {cache_key} for chunk {cache_entry.get('chunk_id')}"
    #                             )

    #                     # If the mode is now empty, remove it entirely
    #                     if not mode_data:
    #                         del self._data[mode_key]

    #             # Set update flags to notify persistence is needed
    #             await set_all_update_flags(self.namespace)

    #         logger.info(f"Cleared cache for {len(chunk_ids)} chunk IDs")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Error clearing cache by chunk IDs: {e}")
    #         return False

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources
           This action will persistent the data to disk immediately.

        This method will:
        1. Clear all data from memory
        2. Update flags to notify other processes
        3. Trigger index_done_callback to save the empty state

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.namespace)

            await self.index_done_callback()
            logger.info(f"Process {os.getpid()} drop {self.namespace}")
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def finalize(self):
        """Finalize storage resources
        Persistence cache data to disk before exiting
        """
        if self.namespace.endswith("cache"):
            await self.index_done_callback()
