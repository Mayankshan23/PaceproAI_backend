"""
PacePro AI - MongoDB Database Manager
=====================================
Handles connection to MongoDB Atlas and provides data access helpers for:
- Athlete profiles (synced with Clerk / custom inputs)
- Biomechanics delivery analysis reports
- Generated diet and nutrition protocols
- 7-day workload training plans
- History logs
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
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


class DatabaseManager:
    def __init__(self):
        self._client = None
        self._db = None
        self._connected = False
        self._last_error = None

    def connect(self) -> bool:
        """Initializes client and tests connection."""
        if not _PYMONGO_AVAILABLE:
            self._last_error = "pymongo package not installed."
            logger.warning("[MongoDB] pymongo package is not available.")
            return False

        uri = config.get_mongodb_uri()
        if not uri:
            self._last_error = "MONGODB_URI not configured."
            logger.warning("[MongoDB] No MONGODB_URI specified.")
            return False

        if "<db_password>" in uri:
            self._last_error = "MONGODB_URI contains '<db_password>' placeholder. Please set your actual database password in backend/.env"
            logger.warning(f"[MongoDB] {self._last_error}")
            return False

        try:
            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=3500,
                connectTimeoutMS=3500
            )
            # Trigger quick server ping
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
            logger.error(f"[MongoDB] Connection failed: {e}")
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
            "database": config.get_mongodb_db_name(),
            "has_password_placeholder": "<db_password>" in config.get_mongodb_uri(),
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
        if not self.is_connected or self.users_col is None:
            return False
        try:
            doc = {
                **profile_data,
                "user_id": user_id,
                "updated_at": datetime.utcnow()
            }
            self.users_col.update_one(
                {"user_id": user_id},
                {"$set": doc},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Error saving athlete profile: {e}")
            return False

    def get_athlete_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected or self.users_col is None:
            return None
        try:
            doc = self.users_col.find_one({"user_id": user_id}, {"_id": 0})
            return doc
        except Exception as e:
            logger.error(f"[MongoDB] Error reading athlete profile: {e}")
            return None

    # =========================================================================
    # Biomechanics Analysis Records with Auto-Deletion (Max 3 retained)
    # =========================================================================
    def save_analysis_record(self, record: Dict[str, Any], user_id: Optional[str] = None, max_retained: int = 3) -> Optional[str]:
        if not self.is_connected or self.analyses_col is None:
            return None
        try:
            uid = user_id or record.get("user_id", "default_athlete")
            record_copy = dict(record)
            record_copy["user_id"] = uid
            record_copy["created_at"] = datetime.utcnow()
            res = self.analyses_col.insert_one(record_copy)

            # Auto-delete older analysis records beyond max_retained (2-3 generations)
            try:
                all_records = list(self.analyses_col.find({"user_id": uid}, {"_id": 1}).sort("created_at", -1))
                if len(all_records) > max_retained:
                    stale_ids = [d["_id"] for d in all_records[max_retained:]]
                    self.analyses_col.delete_many({"_id": {"$in": stale_ids}})
                    logger.info(f"[MongoDB] Auto-deleted {len(stale_ids)} older analysis generations. Retained latest {max_retained}.")
            except Exception as del_err:
                logger.warning(f"[MongoDB] Auto-deletion warning in analyses: {del_err}")

            return str(res.inserted_id)
        except Exception as e:
            logger.error(f"[MongoDB] Error saving analysis: {e}")
            return None

    def get_recent_analyses(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.is_connected or self.analyses_col is None:
            return []
        try:
            query = {"user_id": user_id} if user_id else {}
            cursor = self.analyses_col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"[MongoDB] Error fetching analyses: {e}")
            return []

    # =========================================================================
    # Workout & Nutrition Protocols with Auto-Deletion (Max 3 retained)
    # =========================================================================
    def save_plan_record(self, plan_type: str, plan_data: Dict[str, Any], user_id: Optional[str] = None, max_retained: int = 3) -> Optional[str]:
        if not self.is_connected or self.plans_col is None:
            return None
        try:
            uid = user_id or "default_athlete"
            doc = {
                "type": plan_type,  # 'combined', 'diet', 'workout'
                "user_id": uid,
                "plan": plan_data,
                "created_at": datetime.utcnow()
            }
            res = self.plans_col.insert_one(doc)

            # Auto-delete older plan records of this type beyond max_retained (2-3 generations)
            try:
                all_plans = list(self.plans_col.find({"user_id": uid, "type": plan_type}, {"_id": 1}).sort("created_at", -1))
                if len(all_plans) > max_retained:
                    stale_ids = [d["_id"] for d in all_plans[max_retained:]]
                    self.plans_col.delete_many({"_id": {"$in": stale_ids}})
                    logger.info(f"[MongoDB] Auto-deleted {len(stale_ids)} older {plan_type} plans. Retained latest {max_retained}.")
            except Exception as del_err:
                logger.warning(f"[MongoDB] Auto-deletion warning in plans: {del_err}")

            return str(res.inserted_id)
        except Exception as e:
            logger.error(f"[MongoDB] Error saving plan record: {e}")
            return None

    def get_recent_plans(self, plan_type: Optional[str] = None, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.is_connected or self.plans_col is None:
            return []
        try:
            query = {}
            if plan_type:
                query["type"] = plan_type
            if user_id:
                query["user_id"] = user_id
            cursor = self.plans_col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"[MongoDB] Error fetching plans: {e}")
            return []

    # =========================================================================
    # History Timeline with Auto-Deletion (Max 3 generations retained per type)
    # =========================================================================
    def save_history_record(self, item: Dict[str, Any], user_id: Optional[str] = None, max_retained: int = 3) -> Dict[str, Any]:
        """
        Saves a performance history record to MongoDB and automatically prunes
        older generations beyond max_retained (default 3) for the user and item type.
        """
        if not self.is_connected or self.history_col is None:
            return {"status": "offline", "saved": False}
        try:
            uid = user_id or item.get("user_id") or "default_athlete"
            h_type = item.get("type", "general")
            hist_id = item.get("id") or f"h_{int(datetime.utcnow().timestamp()*1000)}"

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
                "created_at": datetime.utcnow()
            }

            self.history_col.update_one(
                {"id": hist_id},
                {"$set": doc},
                upsert=True
            )

            # Auto-delete older history items of this category beyond max_retained
            deleted_count = 0
            try:
                type_query = {"user_id": uid, "type": h_type}
                existing_docs = list(self.history_col.find(type_query, {"_id": 1, "id": 1}).sort("created_at", -1))
                if len(existing_docs) > max_retained:
                    stale_docs = existing_docs[max_retained:]
                    stale_ids = [d["_id"] for d in stale_docs]
                    del_res = self.history_col.delete_many({"_id": {"$in": stale_ids}})
                    deleted_count = del_res.deleted_count
                    logger.info(f"[MongoDB] Auto-pruned {deleted_count} older history items for '{h_type}'. Retaining latest {max_retained}.")
            except Exception as prune_err:
                logger.warning(f"[MongoDB] Auto-deletion error in history: {prune_err}")

            return {
                "status": "ok",
                "saved": True,
                "history_id": hist_id,
                "pruned_count": deleted_count,
                "retained_max": max_retained
            }
        except Exception as e:
            logger.error(f"[MongoDB] Error saving history record: {e}")
            return {"status": "error", "saved": False, "error": str(e)}

    def get_history(self, user_id: Optional[str] = None, item_type: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieves history records sorted from newest to oldest."""
        if not self.is_connected or self.history_col is None:
            return []
        try:
            query = {}
            if user_id and user_id != "all":
                query["user_id"] = user_id
            if item_type and item_type != "all":
                query["type"] = item_type
            cursor = self.history_col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
            items = []
            for doc in cursor:
                if "created_at" in doc and isinstance(doc["created_at"], datetime):
                    doc["created_at"] = doc["created_at"].isoformat()
                items.append(doc)
            return items
        except Exception as e:
            logger.error(f"[MongoDB] Error fetching history records: {e}")
            return []

    def delete_history_item(self, history_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes a specific history record."""
        if not self.is_connected or self.history_col is None:
            return False
        try:
            query = {"$or": [{"id": history_id}, {"history_id": history_id}]}
            if user_id:
                query["user_id"] = user_id
            res = self.history_col.delete_one(query)
            return res.deleted_count > 0
        except Exception as e:
            logger.error(f"[MongoDB] Error deleting history item {history_id}: {e}")
            return False


# Global Database Singleton
db_manager = DatabaseManager()
