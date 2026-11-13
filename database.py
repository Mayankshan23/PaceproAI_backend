"""
PacePro AI - Hybrid Database Manager (MongoDB Atlas + Resilient Local Persistence)
==================================================================================
Handles connection to MongoDB Atlas and provides bulletproof data persistence for:
- Athlete profiles (synced with Clerk / custom inputs)
- Biomechanics delivery analysis reports
- Generated diet and nutrition protocols
- 7-day workload training plans
- Performance History Timeline (with 3-generation auto-retention)

Resilience:
If MongoDB Atlas is unreachable or offline, automatically persists everything to a local
file-based JSON store (backend/data/pacepro_storage.json). When online, operations sync
to both systems.
"""

import os
import json
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    _PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    ConnectionFailure = Exception
    ServerSelectionTimeoutError = Exception
    _PYMONGO_AVAILABLE = False

import config

logger = logging.getLogger("pacepro.db")


class LocalStorageFallback:
    """Thread-safe JSON file store ensuring zero data loss if MongoDB is offline."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.storage_path.exists():
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump({"users": {}, "analyses": [], "plans": [], "history": []}, f, indent=2)
        except Exception as e:
            logger.error(f"[LocalStorage] Failed to init storage file: {e}")

    def read_store(self) -> Dict[str, Any]:
        with self._lock:
            try:
                if self.storage_path.exists():
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"[LocalStorage] Read error: {e}")
            return {"users": {}, "analyses": [], "plans": [], "history": []}

    def write_store(self, data: Dict[str, Any]):
        with self._lock:
            try:
                temp_file = self.storage_path.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                temp_file.replace(self.storage_path)
            except Exception as e:
                logger.error(f"[LocalStorage] Write error: {e}")


class DatabaseManager:
    def __init__(self):
        self._client = None
        self._db = None
        self._connected = False
        self._last_error = None
        
        # Local fallback store
        data_dir = Path(__file__).resolve().parent / "data"
        self._local = LocalStorageFallback(data_dir / "pacepro_storage.json")

    def connect(self) -> bool:
        """Initializes client and tests connection."""
        if not _PYMONGO_AVAILABLE:
            self._last_error = "pymongo package not installed."
            return False

        uri = config.get_mongodb_uri()
        if not uri:
            self._last_error = "MONGODB_URI not configured."
            return False

        if "<db_password>" in uri:
            self._last_error = "MONGODB_URI contains '<db_password>' placeholder."
            return False

        try:
            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000
            )
            # Quick ping
            self._client.admin.command('ping')
            db_name = config.get_mongodb_db_name()
            self._db = self._client[db_name]
            self._connected = True
            self._last_error = None
            logger.info(f"[MongoDB] Successfully connected to database: '{db_name}'")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            self._connected = False
            self._last_error = str(e)
            return False

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._client is None:
            return self.connect()
        return self._connected

    def get_status(self) -> Dict[str, Any]:
        """Returns connection diagnostics for health check."""
        connected = self.is_connected
        return {
            "configured": bool(config.get_mongodb_uri()),
            "connected": connected,
            "local_store_active": True,
            "local_store_path": str(self._local.storage_path),
            "database": config.get_mongodb_db_name(),
            "has_password_placeholder": "<db_password>" in (config.get_mongodb_uri() or ""),
            "error": self._last_error if not connected else None
        }

    # =========================================================================
    # Collections
    # =========================================================================
    @property
    def users_col(self):
        return self._db["users"] if self._db is not None else None

    @property
    def analyses_col(self):
        return self._db["analyses"] if self._db is not None else None

    @property
    def plans_col(self):
        return self._db["plans"] if self._db is not None else None

    @property
    def history_col(self):
        return self._db["history"] if self._db is not None else None

    # =========================================================================
    # Athlete Profile Data Access
    # =========================================================================
    def save_athlete_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        uid = user_id or "default_athlete"
        doc = {
            **profile_data,
            "user_id": uid,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # 1. Always update local store
        try:
            store = self._local.read_store()
            if "users" not in store:
                store["users"] = {}
            store["users"][uid] = doc
            self._local.write_store(store)
        except Exception as le:
            logger.warning(f"[LocalStorage] Profile write warning: {le}")

        # 2. Sync to MongoDB if connected
        if self.is_connected and self.users_col is not None:
            try:
                self.users_col.update_one(
                    {"user_id": uid},
                    {"$set": {**profile_data, "user_id": uid, "updated_at": datetime.utcnow()}},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"[MongoDB] Profile write error: {e}")

        return True

    def get_athlete_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        uid = user_id or "default_athlete"
        # 1. Check MongoDB first if connected
        if self.is_connected and self.users_col is not None:
            try:
                doc = self.users_col.find_one({"user_id": uid}, {"_id": 0})
                if doc:
                    return doc
            except Exception as e:
                logger.warning(f"[MongoDB] Profile read error: {e}")

        # 2. Fallback to local store
        store = self._local.read_store()
        users = store.get("users", {})
        return users.get(uid) or users.get("default_athlete")

    # =========================================================================
    # Biomechanics Analysis Records with Auto-Deletion (Max 3 retained)
    # =========================================================================
    def save_analysis_record(self, record: Dict[str, Any], user_id: Optional[str] = None, max_retained: int = 3) -> Optional[str]:
        uid = user_id or record.get("user_id", "default_athlete")
        record_id = record.get("id") or f"ana_{int(datetime.utcnow().timestamp()*1000)}"
        record_copy = dict(record)
        record_copy["id"] = record_id
        record_copy["user_id"] = uid
        record_copy["created_at"] = datetime.utcnow().isoformat()

        # 1. Always save to local store with pruning
        try:
            store = self._local.read_store()
            analyses = store.get("analyses", [])
            analyses = [a for a in analyses if a.get("id") != record_id]
            analyses.insert(0, record_copy)
            user_analyses = [a for a in analyses if a.get("user_id") == uid]
            if len(user_analyses) > max_retained:
                stale_ids = {a.get("id") for a in user_analyses[max_retained:]}
                analyses = [a for a in analyses if a.get("id") not in stale_ids]
            store["analyses"] = analyses
            self._local.write_store(store)
        except Exception as le:
            logger.warning(f"[LocalStorage] Save analysis error: {le}")

        # 2. Sync to MongoDB if connected
        if self.is_connected and self.analyses_col is not None:
            try:
                mongo_copy = dict(record)
                mongo_copy["user_id"] = uid
                mongo_copy["created_at"] = datetime.utcnow()
                res = self.analyses_col.insert_one(mongo_copy)
                all_records = list(self.analyses_col.find({"user_id": uid}, {"_id": 1}).sort("created_at", -1))
                if len(all_records) > max_retained:
                    stale_ids = [d["_id"] for d in all_records[max_retained:]]
                    self.analyses_col.delete_many({"_id": {"$in": stale_ids}})
                return str(res.inserted_id)
            except Exception as e:
                logger.warning(f"[MongoDB] Save analysis error: {e}")

        return record_id

    def get_recent_analyses(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        uid = user_id or "default_athlete"
        # 1. Check MongoDB
        if self.is_connected and self.analyses_col is not None:
            try:
                query = {"user_id": uid} if uid != "all" else {}
                cursor = self.analyses_col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
                results = list(cursor)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"[MongoDB] Fetch analyses error: {e}")

        # 2. Fallback to local store
        store = self._local.read_store()
        analyses = store.get("analyses", [])
        if uid != "all":
            analyses = [a for a in analyses if a.get("user_id") == uid]
        return analyses[:limit]

    # =========================================================================
    # Workout & Nutrition Protocols with Auto-Deletion (Max 3 retained)
    # =========================================================================
    def save_plan_record(self, plan_type: str, plan_data: Dict[str, Any], user_id: Optional[str] = None, max_retained: int = 3) -> Optional[str]:
        uid = user_id or "default_athlete"
        plan_id = f"plan_{int(datetime.utcnow().timestamp()*1000)}"
        doc = {
            "id": plan_id,
            "type": plan_type,
            "user_id": uid,
            "plan": plan_data,
            "created_at": datetime.utcnow().isoformat()
        }

        # 1. Always save to local store with pruning
        try:
            store = self._local.read_store()
            plans = store.get("plans", [])
            plans.insert(0, doc)
            user_plans_for_type = [p for p in plans if p.get("user_id") == uid and p.get("type") == plan_type]
            if len(user_plans_for_type) > max_retained:
                stale_ids = {p.get("id") for p in user_plans_for_type[max_retained:]}
                plans = [p for p in plans if p.get("id") not in stale_ids]
            store["plans"] = plans
            self._local.write_store(store)
        except Exception as le:
            logger.warning(f"[LocalStorage] Save plan error: {le}")

        # 2. Sync to MongoDB
        if self.is_connected and self.plans_col is not None:
            try:
                mongo_doc = {
                    "type": plan_type,
                    "user_id": uid,
                    "plan": plan_data,
                    "created_at": datetime.utcnow()
                }
                res = self.plans_col.insert_one(mongo_doc)
                all_plans = list(self.plans_col.find({"user_id": uid, "type": plan_type}, {"_id": 1}).sort("created_at", -1))
                if len(all_plans) > max_retained:
                    stale_ids = [d["_id"] for d in all_plans[max_retained:]]
                    self.plans_col.delete_many({"_id": {"$in": stale_ids}})
                return str(res.inserted_id)
            except Exception as e:
                logger.warning(f"[MongoDB] Save plan error: {e}")

        return plan_id

    def get_recent_plans(self, plan_type: Optional[str] = None, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        uid = user_id or "default_athlete"
        if self.is_connected and self.plans_col is not None:
            try:
                query = {}
                if plan_type:
                    query["type"] = plan_type
                if uid and uid != "all":
                    query["user_id"] = uid
                cursor = self.plans_col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
                results = list(cursor)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"[MongoDB] Fetch plans error: {e}")

        store = self._local.read_store()
        plans = store.get("plans", [])
        if uid and uid != "all":
            plans = [p for p in plans if p.get("user_id") == uid]
        if plan_type:
            plans = [p for p in plans if p.get("type") == plan_type]
        return plans[:limit]

    # =========================================================================
    # History Timeline with Auto-Deletion (Max 3 generations retained per type)
    # =========================================================================
    def save_history_record(self, item: Dict[str, Any], user_id: Optional[str] = None, max_retained: int = 3) -> Dict[str, Any]:
        """
        Saves a performance history record (locally and to MongoDB) with automatic
        pruning of older generations beyond max_retained (default 3) for that type.
        """
        uid = user_id or item.get("user_id") or "default_athlete"
        h_type = item.get("type", "general")
        hist_id = item.get("id") or item.get("history_id") or f"h_{int(datetime.utcnow().timestamp()*1000)}"

        doc = {
            "id": hist_id,
            "history_id": hist_id,
            "user_id": uid,
            "type": h_type,
            "title": item.get("title", "Performance Log"),
            "subtitle": item.get("subtitle", ""),
            "status": item.get("status", "Active"),
            "date": item.get("date") or datetime.utcnow().strftime("%b %d, %Y"),
            "detail": item.get("detail", {}),
            "created_at": datetime.utcnow().isoformat()
        }

        deleted_count = 0

        # 1. Always save to local store with pruning
        try:
            store = self._local.read_store()
            history = store.get("history", [])
            # Remove any with same ID
            history = [h for h in history if h.get("id") != hist_id and h.get("history_id") != hist_id]
            history.insert(0, doc)

            # Prune older items for this user & type
            type_items = [h for h in history if h.get("user_id") == uid and h.get("type") == h_type]
            if len(type_items) > max_retained:
                stale_ids = {h.get("id") for h in type_items[max_retained:]}
                deleted_count = len(stale_ids)
                history = [h for h in history if h.get("id") not in stale_ids]

            store["history"] = history
            self._local.write_store(store)
            logger.info(f"[LocalStorage] Saved history item '{hist_id}' ({h_type}). Total in store: {len(history)}")
        except Exception as le:
            logger.error(f"[LocalStorage] History write error: {le}")

        # 2. Sync to MongoDB if connected
        if self.is_connected and self.history_col is not None:
            try:
                mongo_doc = dict(doc)
                mongo_doc["created_at"] = datetime.utcnow()
                self.history_col.update_one(
                    {"id": hist_id},
                    {"$set": mongo_doc},
                    upsert=True
                )
                type_query = {"user_id": uid, "type": h_type}
                existing_docs = list(self.history_col.find(type_query, {"_id": 1, "id": 1}).sort("created_at", -1))
                if len(existing_docs) > max_retained:
                    stale_docs = existing_docs[max_retained:]
                    stale_ids = [d["_id"] for d in stale_docs]
                    del_res = self.history_col.delete_many({"_id": {"$in": stale_ids}})
                    deleted_count = max(deleted_count, del_res.deleted_count)
            except Exception as me:
                logger.warning(f"[MongoDB] History sync error: {me}")

        return {
            "status": "ok",
            "saved": True,
            "history_id": hist_id,
            "pruned_count": deleted_count,
            "retained_max": max_retained,
            "source": "mongodb" if self.is_connected else "local_store"
        }

    def get_history(self, user_id: Optional[str] = None, item_type: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieves history records sorted from newest to oldest."""
        uid = user_id or "default_athlete"

        # 1. Try MongoDB if connected
        if self.is_connected and self.history_col is not None:
            try:
                query = {}
                if uid and uid != "all":
                    query["user_id"] = uid
                if item_type and item_type != "all":
                    query["type"] = item_type
                cursor = self.history_col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
                items = []
                for doc in cursor:
                    if "created_at" in doc and isinstance(doc["created_at"], datetime):
                        doc["created_at"] = doc["created_at"].isoformat()
                    items.append(doc)
                if items:
                    return items
            except Exception as e:
                logger.warning(f"[MongoDB] Fetch history error: {e}")

        # 2. Resilient Fallback to Local Store
        try:
            store = self._local.read_store()
            history = store.get("history", [])

            if uid and uid != "all":
                history = [h for h in history if h.get("user_id") == uid or not h.get("user_id")]

            if item_type and item_type != "all":
                if item_type == "workout":
                    history = [h for h in history if h.get("type") in ["workout", "combined_plan"]]
                else:
                    history = [h for h in history if h.get("type") == item_type]

            # Sort by created_at desc
            history.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            return history[:limit]
        except Exception as le:
            logger.error(f"[LocalStorage] History read error: {le}")
            return []

    def delete_history_item(self, history_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes a specific history record from both local store and MongoDB."""
        deleted = False

        # 1. Local Store deletion
        try:
            store = self._local.read_store()
            orig_len = len(store.get("history", []))
            store["history"] = [
                h for h in store.get("history", [])
                if h.get("id") != history_id and h.get("history_id") != history_id
            ]
            if len(store["history"]) < orig_len:
                deleted = True
                self._local.write_store(store)
        except Exception as le:
            logger.warning(f"[LocalStorage] Delete history warning: {le}")

        # 2. MongoDB deletion
        if self.is_connected and self.history_col is not None:
            try:
                query = {"$or": [{"id": history_id}, {"history_id": history_id}]}
                if user_id:
                    query["user_id"] = user_id
                res = self.history_col.delete_one(query)
                if res.deleted_count > 0:
                    deleted = True
            except Exception as me:
                logger.warning(f"[MongoDB] Delete history error: {me}")

        return deleted


# Global Database Singleton
db_manager = DatabaseManager()
